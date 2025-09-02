import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
from pyspark.sql import DataFrame as SparkDataFrame

from modeltrack.feature_engineering.binning import TreeBinner, CutBinner
from modeltrack.feature_engineering.encoder import WOEEncoder


class FeaturePipelineDev:
    """
    Pipeline de feature engineering para ambiente de desenvolvimento.
    Faz o fit nos dados, aplica transformações e salva os metadados em JSON.
    """

    def __init__(self, binning_method: str = "tree", max_depth: int = 3):
        if binning_method == "tree":
            self.binner = TreeBinner(max_depth=max_depth)
        elif binning_method == "cut":
            self.binner = CutBinner()
        else:
            raise ValueError("Método de binning inválido. Use 'tree' ou 'cut'.")
        self.encoder = WOEEncoder()
        self.fitted_features: Dict[str, Any] = {}

    def fit_transform(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        features_binning: List[str],
        features_encoding: List[str],
        target: str,
        suffix_binning: str = "_binned",
        suffix_encoding: str = "_woe"
    ):
        # Binning
        for f in features_binning:
            self.binner.fit(
                X=df[f].toPandas() if isinstance(df, SparkDataFrame) else df[f],
                y=df[target].toPandas() if isinstance(df, SparkDataFrame) else df[target],
                col_name=f
            )
        df = self.binner.transform_dataframe(df, features_binning, suffix=suffix_binning)

        # WOE
        for f in features_encoding:
            self.encoder.fit(
                X=df[f].toPandas() if isinstance(df, SparkDataFrame) else df[f],
                y=df[target].toPandas() if isinstance(df, SparkDataFrame) else df[target],
                col_name=f
            )
        if isinstance(df, SparkDataFrame):
            df = self.encoder.transform(df, columns=features_encoding, suffix=suffix_encoding)
        else:
            df = self.encoder.transform_dataframe(df, features=features_encoding, suffix=suffix_encoding)

        return df

    def save(self, path: Union[str, Path]):
        """Salva os parâmetros do binner e encoder em JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        binner_path = path / "binner.json"
        # salvar bins
        with open(binner_path, "w") as f:
            json.dump(self.binner.features_bins_labels, f, indent=2)

        # salvar WOE
        encoder_path = path / "encoder.json"
        with open(encoder_path, "w") as f:
            json.dump(self.encoder.woe_map, f, indent=2)


class FeaturePipelinePrd:
    """
    Pipeline de produção que carrega parâmetros já treinados
    e aplica as transformações automaticamente.
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path)
        self.binner = TreeBinner()
        self.encoder = WOEEncoder()

        # Carregar metadados
        with open(path / "binner.json", "r") as f:
            self.binner.import_config(json.load(f))
        with open(path / "encoder.json", "r") as f:
            self.encoder.import_config(json.load(f))

    def transform(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        features_binning: List[str],
        features_encoding: List[str],
        suffix_binning: str = "_binned",
        suffix_encoding: str = "_woe"
    ):
        # Aplicar binning
        df = self.binner.transform_dataframe(df, features_binning, suffix=suffix_binning)

        # Aplicar WOE
        if isinstance(df, SparkDataFrame):
            df = self.encoder.transform(df, columns=features_encoding, suffix=suffix_encoding)
        else:
            df = self.encoder.transform_dataframe(df, features=features_encoding, suffix=suffix_encoding)

        return df