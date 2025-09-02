# feature_engineering/feature_pipeline.py
import pandas as pd
from .base import BaseBinner

class FeaturePipeline:
    def __init__(self, transformations: dict):
        """
        transformations: dict[col_name] -> list(instances of Transformers)
        ex:
        {
            "idade": [TreeBinner(max_depth=2), WOEEncoder()],
            "renda": [CutBinner(bins=[2000,5000]), WOEEncoder()],
            "sexo": [WOEEncoder()],
        }
        """
        self.transformations = transformations
        self.fitted_transformers = {}
        self.binning_rules = {}  # col -> bins (se houver)
        self.woe_maps = {}       # col -> woe_map (se houver)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fitted_transformers = {}
        self.binning_rules = {}
        self.woe_maps = {}

        for col, steps in self.transformations.items():
            series = X[col]
            fitted_steps = []
            for step in steps:
                # fit pode retornar info (ex.: bins, woe_map)
                info = step.fit(series, y)
                # se for um binner, capture explicitamente os cuts
                if isinstance(step, BaseBinner):
                    self.binning_rules[col] = getattr(step, "bins", info)
                    # após fit, atualiza a série com os valores binned
                    series = step.transform(series)
                else:
                    # genérico (ex: WOEEncoder)
                    # se o fit retornou um map (woe_map), pode armazenar
                    if isinstance(info, dict):
                        # assume que é woe_map
                        self.woe_maps[col] = info
                    series = step.transform(series)
                fitted_steps.append(step)

            self.fitted_transformers[col] = fitted_steps
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, steps in self.fitted_transformers.items():
            series = X_out[col]
            for step in steps:
                series = step.transform(series)
            X_out[col] = series
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)