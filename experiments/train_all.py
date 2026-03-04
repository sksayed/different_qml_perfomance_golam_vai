"""
Train QSVM, VQC, QMLP, QNN on the same CIC IoMT 2024 data.
Uses framework from config: "pennylane" or "qiskit".
"""
import json
import sys
import time
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score

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
    train_file = cfg.get("train_file", "CIC_IoMT_2024_WiFi_MQTT_train.parquet")
    test_file = cfg.get("test_file", "CIC_IoMT_2024_WiFi_MQTT_test.parquet")
    train_path = data_dir / train_file
    test_path = data_dir / test_file
    if not train_path.exists():
        train_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_train.csv"
        test_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

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
    metrics_dir = Path(cfg.get("metrics_dir", "results/metrics"))
    metrics_dir = ROOT / metrics_dir if not metrics_dir.is_absolute() else metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data (framework={framework})...")
    X_train, y_train, X_test, y_test, le = get_dataset(
        train_path=train_path,
        test_path=test_path,
        n_train=n_train,
        n_test=n_test,
        n_components=n_components,
        random_state=random_state,
    )
    print(f"Train {X_train.shape[0]}, Test {X_test.shape[0]}, features {X_train.shape[1]}, classes {len(le.classes_)}")

    model_classes = get_models(framework)
    for ModelClass in model_classes:
        try:
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
            print(f"Training {name}...")
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            infer_time = time.perf_counter() - t0
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="weighted"))
            metrics = {
                "model": name,
                "framework": framework,
                "accuracy": acc,
                "f1_weighted": f1,
                "train_time_sec": round(train_time, 2),
                "inference_time_sec": round(infer_time, 2),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
            out = metrics_dir / f"{name}_metrics.json"
            with open(out, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  {name}: acc={acc:.4f} f1={f1:.4f} train_time={train_time:.1f}s")
        except Exception as exc:
            # Log the error but continue with remaining models
            name = ModelClass.__name__
            print(f"Error while training {name} ({framework}): {exc}")

    print("Done. Metrics in", metrics_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
