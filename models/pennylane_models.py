"""
PennyLane implementations of QSVM, VQC, QMLP, QNN.
Same BaseQuantumClassifier interface as Qiskit models.
Uses PennyLane's differentiable numpy so the autograd graph is preserved for VQC/QMLP/QNN.
"""
import warnings
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np_standard  # for sklearn and final .astype(np.int64)
import pennylane as qml
from pennylane import numpy as np  # differentiable; required for GradientDescentOptimizer
from sklearn.svm import SVC

from .base import BaseQuantumClassifier

# Kernel matrix: prefer qml.kernels, fallback to manual for compatibility
try:
    from pennylane.kernels import kernel_matrix as _kernel_matrix
except ImportError:
    def _kernel_matrix(X1, X2, kernel_fn):
        return np.array([[float(kernel_fn(x1, x2)) for x2 in X2] for x1 in X1])


def _default_device(n_wires: int, shots: Optional[int] = None):
    """Create a PennyLane device.

    Prefer the high-performance C++ backend `lightning.qubit` (requires
    `pip install pennylane-lightning`), and fall back to `default.qubit`
    if it is not available.
    """
    dev_name = "lightning.qubit"
    try:
        if shots and shots > 0:
            return qml.device(dev_name, wires=n_wires, shots=shots)
        return qml.device(dev_name, wires=n_wires)
    except Exception:
        if shots and shots > 0:
            return qml.device("default.qubit", wires=n_wires, shots=shots)
        return qml.device("default.qubit", wires=n_wires)


def _mse_loss(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean squared error between preds and one-hot (-1/1) labels.

    preds: (n_samples, n_classes) in [-1, 1] from Pauli-Z expectations.
    y:     (n_samples,) integer labels (plain numpy indices, no gradients).
    """
    y_idx = np_standard.asarray(y, dtype=np_standard.int64)
    n = preds.shape[0]
    n_classes = preds.shape[1]

    # Initialize to -1.0 to match the lower bound of Pauli-Z expectations
    targets_std = np_standard.full((n, n_classes), -1.0, dtype=float)

    # Vectorized assignment of 1.0 to the correct class index
    targets_std[np_standard.arange(n), y_idx] = 1.0

    # Wrap in PennyLane numpy to preserve the computational graph
    targets = np.array(targets_std)
    return np.mean((preds - targets) ** 2)


# --- PennyLane QSVM: quantum kernel + SVC ---


class PennyLaneQSVM(BaseQuantumClassifier):
    """QSVM using PennyLane kernel (feature map) + sklearn SVC with precomputed kernel."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_features: Optional[int] = None,
        reps: int = 2,
        shots: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        self.n_qubits = n_qubits
        self.n_features = n_features or min(n_qubits, 2**n_qubits)
        self.reps = reps
        self.shots = shots
        self.random_state = random_state
        # Drop training-only kwargs that SVC doesn't accept
        kwargs.pop("step_size", None)
        kwargs.pop("max_iter", None)
        self._kwargs = kwargs
        self._dev = None
        self._kernel_fn = None
        self._svc = None
        self._X_train = None

    @property
    def name(self) -> str:
        return "pennylane_qsvm"

    def _feature_map(self, x):
        for _ in range(self.reps):
            for i in range(min(len(x), self.n_qubits)):
                qml.RY(x[i % len(x)], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

    def _build_kernel(self, n_features: int):
        n_wires = self.n_qubits
        n_feat = min(n_features, self.n_features, 2**n_wires)
        self._dev = _default_device(n_wires, self.shots)

        @qml.qnode(self._dev)
        def kernel_circuit(x1, x2):
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            return qml.probs(wires=list(range(n_wires)))

        def kernel_fn(x1, x2):
            probs = kernel_circuit(x1, x2)
            return float(probs[0])  # |<phi(x1)|phi(x2)>|^2 = prob of |0..0>

        self._kernel_fn = kernel_fn

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "PennyLaneQSVM":
        n_features = X.shape[1]
        self._build_kernel(n_features)
        n = X.shape[0]
        n_evals = n * (n - 1) // 2  # upper triangle only; diagonal = 1.0
        print(f"  {self.name}: computing {n}x{n} symmetric kernel ({n_evals} circuits + diagonal) — may take hours...", flush=True)
        K_train = np_standard.zeros((n, n))
        report_every = max(1, n // 20)
        for i in range(n):
            K_train[i, i] = 1.0  # self-similarity is always 1.0 for normalized states
            if i % report_every == 0 or i == n - 1:
                print(f"  {self.name}: kernel row {i + 1}/{n} ({100 * (i + 1) / n:.0f}%)", flush=True)
            for j in range(i + 1, n):
                val = float(self._kernel_fn(X[i], X[j]))
                K_train[i, j] = val
                K_train[j, i] = val
        K_train = np_standard.asarray(K_train)  # sklearn expects plain numpy
        self._svc = SVC(kernel="precomputed", random_state=self.random_state, **{**self._kwargs, **kwargs})
        self._svc.fit(K_train, y)
        self._X_train = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._svc is None or self._X_train is None:
            raise RuntimeError("Model not fitted.")
        K_test = _kernel_matrix(X, self._X_train, self._kernel_fn)
        K_test = np_standard.asarray(K_test)
        return np_standard.asarray(self._svc.predict(K_test), dtype=np_standard.int64)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_features": self.n_features,
            "reps": self.reps,
            "shots": self.shots,
            "random_state": self.random_state,
            **self._kwargs,
        }


class _BasePennyLaneVariational(BaseQuantumClassifier, ABC):
    """Shared training logic for PennyLane variational classifiers (VQC, QMLP, QNN)."""

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
        shots: Optional[int] = None,
        max_iter: int = 100,
        step_size: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        self.n_qubits = n_qubits
        self.reps = reps
        self.shots = shots
        self.max_iter = max_iter
        self.step_size = step_size
        self.random_state = random_state
        self._kwargs = kwargs
        self._dev = None
        self._circuit = None
        self._weights = None
        self._n_classes = None
        self._n_features = None

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - defined in subclasses
        ...

    @abstractmethod
    def _build_qnode(self, n_wires: int, n_feat: int, n_classes: int):
        """Return a QNode: (inputs, weights) -> list of (batch,) expvals."""
        ...

    @abstractmethod
    def _init_weights(self, n_wires: int) -> np.ndarray:
        """Return initial trainable weights (PennyLane numpy, requires_grad=True)."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "_BasePennyLaneVariational":
        self._n_classes = int(np_standard.max(y)) + 1
        self._n_features = X.shape[1]
        n_wires = max(self.n_qubits, self._n_classes)
        self._dev = _default_device(n_wires, self.shots)
        n_feat = min(self._n_features, n_wires)

        if self._n_features > n_wires:
            warnings.warn(
                f"Only {n_wires} of {self._n_features} features are used (n_qubits/n_wires). "
                "Set config n_components <= n_qubits to avoid dropping features.",
                UserWarning,
                stacklevel=2,
            )

        self._circuit = self._build_qnode(n_wires, n_feat, self._n_classes)
        self._weights = self._init_weights(n_wires)

        batch_size = min(32, len(X))
        rng = np_standard.random.default_rng(self.random_state)
        opt = qml.GradientDescentOptimizer(stepsize=self.step_size)

        for step in range(self.max_iter):
            batch_idx = rng.integers(0, len(X), size=batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            def cost(weights):
                preds_list = [self._circuit(x, weights) for x in X_batch]
                preds = np.stack(preds_list, axis=0)
                return _mse_loss(preds, y_batch)

            self._weights, curr_loss = opt.step_and_cost(cost, self._weights)
            print(f"{self.name}: step {step + 1}/{self.max_iter}, loss={float(curr_loss):.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._circuit is None or self._weights is None:
            raise RuntimeError("Model not fitted.")
        preds_list = [self._circuit(x, self._weights) for x in X]
        preds = np.stack(preds_list, axis=0)
        return np_standard.asarray(np.argmax(preds, axis=1), dtype=np_standard.int64)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "shots": self.shots,
            "max_iter": self.max_iter,
            "step_size": self.step_size,
            "random_state": self.random_state,
            **self._kwargs,
        }


# --- PennyLane VQC: variational circuit, multi-class via n_classes expectations ---


class PennyLaneVQC(_BasePennyLaneVariational):
    """VQC: variational quantum classifier with PennyLane."""

    @property
    def name(self) -> str:
        return "pennylane_vqc"

    def _build_qnode(self, n_wires: int, n_feat: int, n_classes: int):
        dev = self._dev
        reps = self.reps

        @qml.qnode(dev)
        def circuit(inputs, weights):
            # inputs[..., i] supports both (n_feat,) and (batch, n_feat)
            for i in range(n_feat):
                qml.RY(inputs[..., i], wires=i)
            for i in range(n_feat, n_wires):
                qml.RY(0.0, wires=i)
            for layer in range(reps):
                for i in range(n_wires):
                    qml.Rot(*weights[layer, i], wires=i)
                for i in range(n_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

        return circuit

    def _init_weights(self, n_wires: int) -> np.ndarray:
        rng = np_standard.random.default_rng(self.random_state)
        init_std = rng.uniform(-0.5, 0.5, (self.reps, n_wires, 3))
        return np.array(init_std, requires_grad=True)


# --- PennyLane QMLP: deeper variational circuit (more layers) ---


class PennyLaneQMLP(_BasePennyLaneVariational):
    """QMLP: multi-layer variational circuit (StronglyEntanglingLayers-style)."""

    @property
    def name(self) -> str:
        return "pennylane_qmlp"

    def _build_qnode(self, n_wires: int, n_feat: int, n_classes: int):
        dev = self._dev
        reps = self.reps

        @qml.qnode(dev)
        def circuit(inputs, weights):
            for i in range(n_feat):
                qml.RY(inputs[..., i], wires=i)
            for i in range(n_feat, n_wires):
                qml.RY(0.0, wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

        return circuit

    def _init_weights(self, n_wires: int) -> np.ndarray:
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.reps, n_wires=n_wires)
        rng = np_standard.random.default_rng(self.random_state)
        init_std = rng.uniform(-0.5, 0.5, shape)
        return np.array(init_std, requires_grad=True)


# --- PennyLane QNN: Neural-network style (BasicEntanglerLayers + multi-output) ---


class PennyLaneQNN(_BasePennyLaneVariational):
    """QNN: quantum neural network (BasicEntanglerLayers + measured expectations)."""

    @property
    def name(self) -> str:
        return "pennylane_qnn"

    def _build_qnode(self, n_wires: int, n_feat: int, n_classes: int):
        dev = self._dev

        @qml.qnode(dev)
        def circuit(inputs, weights):
            for i in range(n_feat):
                qml.RY(inputs[..., i], wires=i)
            for i in range(n_feat, n_wires):
                qml.RY(0.0, wires=i)
            qml.BasicEntanglerLayers(weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]

        return circuit

    def _init_weights(self, n_wires: int) -> np.ndarray:
        shape = qml.BasicEntanglerLayers.shape(n_layers=self.reps, n_wires=n_wires)
        rng = np_standard.random.default_rng(self.random_state)
        init_std = rng.uniform(-np.pi, np.pi, shape)
        return np.array(init_std, requires_grad=True)
