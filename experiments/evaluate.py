"""Load metrics from results/metrics and print comparison table (Qiskit and PennyLane)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    metrics_dir = ROOT / "results" / "metrics"
    if not metrics_dir.exists():
        print("No results/metrics dir. Run train_all.py first.")
        return 1
    rows = []
    for p in metrics_dir.glob("*_metrics.json"):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print("No *_metrics.json files found.")
        return 1
    rows.sort(key=lambda x: (-x.get("accuracy", 0), x.get("model", "")))
    print("Model                    | Framework  | Accuracy | F1 (weighted) | Train (s)")
    print("-" * 75)
    for r in rows:
        m = r.get("model", "")
        fw = r.get("framework", "")
        acc = r.get("accuracy", 0)
        f1 = r.get("f1_weighted", 0)
        t = r.get("train_time_sec", 0)
        print(f"{m:24} | {fw:10} | {acc:.4f}   | {f1:.4f}         | {t:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
