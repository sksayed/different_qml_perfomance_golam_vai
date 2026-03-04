"""Common interface for all quantum classifiers."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BaseQuantumClassifier(ABC):
    """Base class for QSVM, VQC, QMLP, QNN - same interface for train_all / evaluate."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging and saving (e.g. 'qsvm', 'vqc')."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseQuantumClassifier":
        """Train the model. Returns self."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Return dict of constructor params for reproducibility."""
        return {}

    def save(self, path: Path) -> None:
        """Optional: save model state. Override in subclasses that support it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Default: no-op; subclasses can persist state
        pass

    def load(self, path: Path) -> "BaseQuantumClassifier":
        """Optional: load model state. Override in subclasses."""
        return self
