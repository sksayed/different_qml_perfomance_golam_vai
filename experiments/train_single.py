"""
Train a single model. Usage:
  python -m experiments.train_single pennylane_qsvm
  python -m experiments.train_single pennylane_vqc
  python -m experiments.train_single qsvm   (requires Qiskit)
"""
import sys
from pathlib import Path

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

    import yaml
    from data.dataset import get_dataset
    from models import get_models

    with open(ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    framework = cfg.get("framework", "pennylane").lower()
    data_dir = ROOT / cfg.get("data_dir", "CIC_iomt_dataset")
    train_path = data_dir / cfg.get("train_file", "CIC_IoMT_2024_WiFi_MQTT_train.parquet")
    test_path = data_dir / cfg.get("test_file", "CIC_IoMT_2024_WiFi_MQTT_test.parquet")
    if not train_path.exists():
        train_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_train.csv"
        test_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

    model_classes = get_models(framework)
    n_qubits = cfg.get("n_qubits", 4)
    model_map = {cl(n_qubits=n_qubits).name: cl for cl in model_classes}
    if name not in model_map:
        print(f"Unknown model: {name}. Available: {list(model_map.keys())}")
        return 1
    ModelClass = model_map[name]

    X_train, y_train, X_test, y_test, _ = get_dataset(
        train_path=train_path,
        test_path=test_path,
        n_train=cfg.get("n_train"),
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
    model.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score, f1_score
    pred = model.predict(X_test)
    print(name, "accuracy:", accuracy_score(y_test, pred), "f1:", f1_score(y_test, pred, average="weighted"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
