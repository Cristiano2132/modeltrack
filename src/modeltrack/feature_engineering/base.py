# feature_engineering/base.py
from abc import ABC, abstractmethod
from typing import Any

class Transformer(ABC):
    @abstractmethod
    def fit(self, X, y=None) -> Any:
        """Treina/ajusta o transformer. Pode retornar info útil (ex: cuts)."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class BaseBinner(Transformer):
    """Binner especializado; convention: fit retorna as regras de binagem."""
    @abstractmethod
    def fit(self, X, y=None) -> Any:
        """Deve retornar os cuts (por ex.: list[float]) e armazená-los em self.bins"""
        raise NotImplementedError