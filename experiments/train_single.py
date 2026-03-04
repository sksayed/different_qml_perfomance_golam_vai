"""
Train a single model. Usage:
  python -m experiments.train_single pennylane_qsvm
  python -m experiments.train_single pennylane_vqc
  python -m experiments.train_single qsvm   (requires Qiskit)
"""
import sys
import time
from pathlib import Path

import json
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.train_single <model_name>")
        print("  Model names: pennylane_qsvm, pennylane_vqc, pennylane_qmlp, pennylane_qnn")
        print("  With Qiskit: qsvm, vqc, qmlp, qnn")
        return 1
    name = sys.argv[1].strip().lower()

    from data.dataset import get_dataset
    from models import get_models

    with open(ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    framework = cfg.get("framework", "pennylane").lower()
    data_dir = ROOT / cfg.get("data_dir", "CIC_iomt_dataset")
    # Test set: always use the original hold-out file (Parquet preferred, CSV fallback)
    test_path = data_dir / cfg.get("test_file", "CIC_IoMT_2024_WiFi_MQTT_test.parquet")
    if not test_path.exists():
        test_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

    # Pre-generated training datasets:
    # - CIC_IoMT_2024_Variational.parquet: for VQC/QMLP/QNN (larger, capped per class)
    # - CIC_IoMT_2024_QSVM.parquet:       for QSVM (smaller, stratified ~5k rows)
    var_train_path = data_dir / "CIC_IoMT_2024_Variational.parquet"
    qsvm_train_path = data_dir / "CIC_IoMT_2024_QSVM.parquet"

    model_classes = get_models(framework)
    n_qubits = cfg.get("n_qubits", 4)
    model_map = {cl(n_qubits=n_qubits).name: cl for cl in model_classes}
    if name not in model_map:
        print(f"Unknown model: {name}. Available: {list(model_map.keys())}")
        return 1
    ModelClass = model_map[name]

    # Choose the appropriate training dataset based on the requested model.
    if framework == "pennylane":
        if name == "pennylane_qsvm":
            train_path = qsvm_train_path
        else:
            train_path = var_train_path
    else:
        # For future Qiskit models:
        #   - "qsvm"  -> QSVM dataset
        #   - others  -> Variational dataset
        if name == "qsvm":
            train_path = qsvm_train_path
        else:
            train_path = var_train_path

    if not train_path.exists():
        print(
            f"Training dataset not found at {train_path}. "
            "Run `python -m data.generate_variational_qsvm_datasets` first."
        )
        return 1

    X_train, y_train, X_test, y_test, _ = get_dataset(
        train_path=train_path,
        test_path=test_path,
        # Use the full pre-generated training datasets (Variational / QSVM).
        # They are already sized and balanced appropriately, so we ignore
        # any n_train limit from the config here.
        n_train=None,
        n_test=cfg.get("n_test"),
        n_components=cfg.get("n_components"),
        random_state=cfg.get("random_state", 42),
    )
    reps = cfg.get("reps", 2)
    shots = cfg.get("shots", 1024)
    max_iter = cfg.get("max_iter", 100)
    step_size = cfg.get("step_size", 0.1)
    optimizer = cfg.get("optimizer", "COBYLA")
    random_state = cfg.get("random_state", 42)

    # Create per-run timestamped directory under results/
    base_results = ROOT / cfg.get("results_dir", "results")
    base_results.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_results / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if framework == "pennylane":
        model = ModelClass(
            n_qubits=n_qubits,
            reps=reps,
            shots=shots,
            max_iter=max_iter,
            step_size=step_size,
            random_state=random_state,
        )
    else:
        model = ModelClass(
            n_qubits=n_qubits,
            reps=reps,
            shots=shots,
            max_iter=max_iter,
            optimizer=optimizer,
            random_state=random_state,
        )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_test)
    infer_time = time.perf_counter() - t0

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="weighted"))
    prec = float(precision_score(y_test, pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, pred, average="weighted", zero_division=0))
    print(name, "accuracy:", acc, "f1:", f1, "recall:", rec)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": name,
        "framework": framework,
        "accuracy": acc,
        "f1_weighted": f1,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(infer_time, 2),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    out = metrics_dir / f"{name}_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", out)

    ckpt_path = checkpoints_dir / f"{name}.pkl"
    try:
        model.save(ckpt_path)
        print("Saved checkpoint to", ckpt_path)
    except Exception as exc:
        print("Warning: could not save checkpoint:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
