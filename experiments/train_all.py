"""
Train QSVM, VQC, QMLP, QNN on the same CIC IoMT 2024 data.
Uses framework from config: "pennylane" or "qiskit".
"""
import json
import sys
import time
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataset
from models import get_models


def load_config():
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    framework = cfg.get("framework", "pennylane").lower()
    data_dir = ROOT / cfg.get("data_dir", "CIC_iomt_dataset")
    # Test set: always use the original hold-out file (Parquet preferred, CSV fallback)
    test_file = cfg.get("test_file", "CIC_IoMT_2024_WiFi_MQTT_test.parquet")
    test_path = data_dir / test_file
    if not test_path.exists():
        test_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

    # Pre-generated training datasets:
    # - CIC_IoMT_2024_Variational.parquet: for VQC/QMLP/QNN (larger, capped per class)
    # - CIC_IoMT_2024_QSVM.parquet:       for QSVM (smaller, stratified ~5k rows)
    var_train_path = data_dir / "CIC_IoMT_2024_Variational.parquet"
    qsvm_train_path = data_dir / "CIC_IoMT_2024_QSVM.parquet"

    n_train = cfg.get("n_train")
    n_test = cfg.get("n_test")
    n_components = cfg.get("n_components")
    n_qubits = cfg.get("n_qubits", 4)
    reps = cfg.get("reps", 2)
    shots = cfg.get("shots", 1024)
    max_iter = cfg.get("max_iter", 100)
    optimizer = cfg.get("optimizer", "COBYLA")
    step_size = cfg.get("step_size", 0.1)
    random_state = cfg.get("random_state", 42)

    # Create per-run timestamped directory under results/
    base_results = ROOT / cfg.get("results_dir", "results")
    base_results.mkdir(parents=True, exist_ok=True)
    # Timestamp format: date_hour_minute (24h) for readability, e.g. run_20260304_2310
    run_id = time.strftime("run_%Y%m%d_%H%M")
    run_dir = base_results / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_classes = get_models(framework)
    for ModelClass in model_classes:
        try:
            # Decide which training dataset to use for this model.
            # For PennyLane:
            #   - PennyLaneQSVM -> QSVM dataset (~5k, kernel-based)
            #   - PennyLaneVQC/QMLP/QNN -> Variational dataset (~O(80k))
            if framework == "pennylane":
                model_name = ModelClass.__name__
                is_qsvm = model_name == "PennyLaneQSVM"
            else:
                # For future Qiskit models, match by class name if available.
                model_name = ModelClass.__name__
                is_qsvm = model_name.lower().startswith("qsvm")

            train_path_for_model = qsvm_train_path if is_qsvm else var_train_path

            if not train_path_for_model.exists():
                raise FileNotFoundError(
                    f"Training dataset not found at {train_path_for_model}. "
                    "Run `python -m data.generate_variational_qsvm_datasets` first."
                )

            print(f"\nLoading data for {ModelClass.__name__} from {train_path_for_model.name} (framework={framework})...")
            X_train, y_train, X_test, y_test, le = get_dataset(
                train_path=train_path_for_model,
                test_path=test_path,
                # Use the full pre-generated training datasets (Variational / QSVM).
                # They are already sized and balanced appropriately, so we ignore
                # any n_train limit from the config here.
                n_train=None,
                n_test=n_test,
                n_components=n_components,
                random_state=random_state,
            )
            print(
                f"  Data: Train {X_train.shape[0]}, Test {X_test.shape[0]}, "
                f"features {X_train.shape[1]}, classes {len(le.classes_)}"
            )

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
            name = model.name
            print(f"\n=== Model - {name} ===")
            print(f"Training {name}...")
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t0
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="weighted"))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
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
            print(f"  {name}: acc={acc:.4f} f1={f1:.4f} train_time={train_time:.1f}s")

            # Save model checkpoint, if supported
            ckpt_path = checkpoints_dir / f"{name}.pkl"
            try:
                model.save(ckpt_path)
                print(f"  {name}: checkpoint saved to {ckpt_path}")
            except Exception as save_exc:
                print(f"  {name}: warning - could not save checkpoint: {save_exc}")
        except Exception as exc:
            # Log the error but continue with remaining models
            name = ModelClass.__name__
            print(f"Error while training {name} ({framework}): {exc}")

    print("Done. Metrics in", metrics_dir)
    print("Checkpoints in", checkpoints_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
