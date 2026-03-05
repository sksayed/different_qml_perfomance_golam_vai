"""
Re-evaluate saved checkpoints on a larger test subset.

This script:
  - Finds the most recent results/run_*/ directory.
  - Loads each saved model checkpoint from run_*/checkpoints.
  - Builds a larger test set (without retraining!) and computes metrics:
      accuracy, F1 (weighted), precision (weighted), recall (weighted).
  - Saves new evaluation metrics JSON files alongside the originals.

Usage (from project root):
  # Evaluate all checkpoints from the latest run directory
  python -m experiments.evaluate_checkpoints

  # Evaluate a specific checkpoint file
  python -m experiments.evaluate_checkpoints results/run_YYYYMMDD_HHMM/checkpoints/pennylane_vqc.pkl

By default:
  - Variational models (VQC, QMLP, QNN) are evaluated on 10,000 test samples.
  - QSVM is evaluated on 1,000 test samples (kernel cost scales with n_train * n_test).
"""

import sys
import json
from pathlib import Path
from typing import Dict

import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataset  # noqa: E402
from models import get_models  # noqa: E402


# Evaluation subset sizes
N_TEST_VARIATIONAL = 10_000
N_TEST_QSVM = 1_000


def load_config() -> Dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def find_latest_run(results_dir: Path) -> Path:
    """Return the most recent run_* directory under results_dir."""
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory at {results_dir}")

    run_dirs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {results_dir}")
    return run_dirs[-1]


def main() -> int:
    cfg = load_config()
    framework = cfg.get("framework", "pennylane").lower()
    data_dir = ROOT / cfg.get("data_dir", "CIC_iomt_dataset")

    # Base test file (Parquet preferred, CSV fallback)
    test_file = cfg.get("test_file", "CIC_IoMT_2024_WiFi_MQTT_test.parquet")
    test_path = data_dir / test_file
    if not test_path.exists():
        test_path = data_dir / "CIC_IoMT_2024_WiFi_MQTT_test.csv"

    # Pre-generated training datasets
    var_train_path = data_dir / "CIC_IoMT_2024_Variational.parquet"
    qsvm_train_path = data_dir / "CIC_IoMT_2024_QSVM.parquet"

    if not var_train_path.exists():
        raise FileNotFoundError(
            f"Variational train parquet not found at {var_train_path}. "
            "Run `python -m data.generate_variational_qsvm_datasets` first."
        )
    if not qsvm_train_path.exists():
        raise FileNotFoundError(
            f"QSVM train parquet not found at {qsvm_train_path}. "
            "Run `python -m data.generate_variational_qsvm_datasets` first."
        )

    # Determine which checkpoints to evaluate:
    # - If a .pkl path is provided on the command line, evaluate only that file.
    # - Otherwise, evaluate all checkpoints from the latest run directory.
    if len(sys.argv) > 1:
        ckpt_arg = Path(sys.argv[1])
        ckpt_path = ckpt_arg if ckpt_arg.is_absolute() else (ROOT / ckpt_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        checkpoints_dir = ckpt_path.parent
        latest_run = checkpoints_dir.parent
        ckpt_paths = [ckpt_path]
    else:
        base_results = ROOT / cfg.get("results_dir", "results")
        latest_run = find_latest_run(base_results)
        checkpoints_dir = latest_run / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No checkpoints directory in latest run: {latest_run}")
        ckpt_paths = sorted(checkpoints_dir.glob("*.pkl"))

    metrics_dir = latest_run / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating checkpoints from: {latest_run}")
    print(f"  Checkpoints dir: {checkpoints_dir}")

    # Pre-load datasets once to avoid redundant scaling/PCA work per model
    print("Pre-loading evaluation datasets (variational and QSVM)...")
    var_data = get_dataset(
        train_path=var_train_path,
        test_path=test_path,
        n_train=None,
        n_test=N_TEST_VARIATIONAL,
        n_components=cfg.get("n_components"),
        random_state=cfg.get("random_state", 42),
    )
    qsvm_data = get_dataset(
        train_path=qsvm_train_path,
        test_path=test_path,
        n_train=None,
        n_test=N_TEST_QSVM,
        n_components=cfg.get("n_components"),
        random_state=cfg.get("random_state", 42),
    )

    # Build a mapping from model name -> ModelClass
    model_classes = get_models(framework)
    name_to_class: Dict[str, type] = {}
    for ModelClass in model_classes:
        # Instantiate briefly to get its .name property
        try:
            model = ModelClass()
        except TypeError:
            # Some constructors may require args; fall back to n_qubits=cfg value
            model = ModelClass(n_qubits=cfg.get("n_qubits", 4))
        name_to_class[model.name] = ModelClass

    # Evaluate each checkpoint found
    for ckpt_path in ckpt_paths:
        model_name = ckpt_path.stem  # e.g. "pennylane_vqc"
        if model_name not in name_to_class:
            print(f"Skipping unknown checkpoint {ckpt_path.name} (no matching ModelClass).")
            continue

        ModelClass = name_to_class[model_name]

        is_qsvm = model_name == "pennylane_qsvm" or model_name.lower().startswith("qsvm")
        if is_qsvm:
            X_train, y_train, X_test, y_test, le = qsvm_data
            n_test_eval = N_TEST_QSVM
        else:
            X_train, y_train, X_test, y_test, le = var_data
            n_test_eval = N_TEST_VARIATIONAL

        print(
            f"\n=== Evaluating {model_name} from {ckpt_path.name} "
            f"on {n_test_eval} test samples ==="
        )

        # Instantiate a fresh model and load the checkpoint
        try:
            model = ModelClass()
        except TypeError:
            model = ModelClass(n_qubits=cfg.get("n_qubits", 4))

        try:
            model = model.load(ckpt_path)
        except Exception as exc:
            print(f"  Error loading checkpoint {ckpt_path.name}: {exc}")
            continue

        # Evaluate
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        print(
            f"  Results on {n_test_eval} test samples — "
            f"acc={acc:.4f}, f1={f1:.4f}, precision={prec:.4f}, recall={rec:.4f}"
        )

        # Save a separate evaluation metrics file so we don't overwrite the original
        eval_metrics = {
            "model": model_name,
            "framework": framework,
            "accuracy": acc,
            "f1_weighted": f1,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "n_train_used_for_scaling": int(len(y_train)),
            "n_test_eval": int(len(y_test)),
            "label_classes": list(map(str, le.classes_)),
        }
        out = metrics_dir / f"{model_name}_metrics_eval_large_test.json"
        with open(out, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"  Saved evaluation metrics to {out}")

    print("\nDone evaluating all checkpoints in", latest_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

