from .base import BaseQuantumClassifier
from .pennylane_models import (
    PennyLaneQSVM,
    PennyLaneVQC,
    PennyLaneQMLP,
    PennyLaneQNN,
)

# Qiskit models (optional - require qiskit-machine-learning)
try:
    from .qsvm import QSVMModel
    from .vqc import VQCModel
    from .qmlp import QMLPModel
    from .qnn import QNNModel
    _QISKIT_AVAILABLE = True
except ImportError:
    QSVMModel = VQCModel = QMLPModel = QNNModel = None
    _QISKIT_AVAILABLE = False


def get_models(framework: str = "pennylane"):
    """Return list of [QSVM, VQC, QMLP, QNN] for the given framework."""
    if framework.lower() == "pennylane":
        return [PennyLaneQSVM, PennyLaneVQC, PennyLaneQMLP, PennyLaneQNN]
    if framework.lower() == "qiskit" and _QISKIT_AVAILABLE:
        return [QSVMModel, VQCModel, QMLPModel, QNNModel]
    if framework.lower() == "qiskit" and not _QISKIT_AVAILABLE:
        raise ImportError("Qiskit models require qiskit-machine-learning. Install with: pip install qiskit-machine-learning")
    raise ValueError("framework must be 'pennylane' or 'qiskit'")


__all__ = [
    "BaseQuantumClassifier",
    "PennyLaneQSVM",
    "PennyLaneVQC",
    "PennyLaneQMLP",
    "PennyLaneQNN",
    "QSVMModel",
    "VQCModel",
    "QMLPModel",
    "QNNModel",
    "get_models",
]
