# feature_engineering/encoder.py
import numpy as np
import pandas as pd
from .base import Transformer
from typing import Dict, Any

class WOEEncoder(Transformer):
    def __init__(self, regularization: float = 1.0, unseen_value: float = 0.0):
        self.regularization = regularization
        self.woe_map: Dict[Any, float] = {}
        self.unseen_value = unseen_value

    def fit(self, X: pd.Series, y: pd.Series) -> Dict[Any, float]:
        # assume y binária (0/1)
        if len(y.unique()) != 2:
            raise ValueError("Target deve ser binária")

        total_events = y.sum()
        total_non_events = len(y) - total_events
        woe_map = {}
        for cat in X.dropna().unique():
            mask = X == cat
            events = int(y[mask].sum())
            non_events = int(mask.sum()) - events

            event_rate = (events + self.regularization) / (total_events + 2 * self.regularization)
            non_event_rate = (non_events + self.regularization) / (total_non_events + 2 * self.regularization)
            woe_map[cat] = float(np.log(event_rate / non_event_rate))

        self.woe_map = woe_map
        return self.woe_map

    def transform(self, X: pd.Series) -> pd.Series:
        # categorias não vistas durante o fit recebem unseen_value
        return X.map(lambda v: self.woe_map.get(v, self.unseen_value))