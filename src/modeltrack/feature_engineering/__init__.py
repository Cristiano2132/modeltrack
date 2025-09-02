"""
Feature Engineering
===================

Este módulo oferece classes para transformação de variáveis,
incluindo binagem, encoding e pipelines de features.

Suporta tanto Pandas quanto PySpark DataFrames.

Submódulos
----------
- binning: classes para discretização de variáveis contínuas
    (e.g., TreeBinner, CutBinner)
- encoder: classes para encoders estatísticos
    (e.g., WOEEncoder)
- feature_pipeline: pipeline para orquestrar transformações múltiplas em features

Exemplo de uso
--------------
>>> import pandas as pd
>>> from feature_engineering.binning import TreeBinner, CutBinner
>>> from feature_engineering.encoder import WOEEncoder
>>> from feature_engineering.feature_pipeline import FeaturePipeline
>>>
>>> df = pd.DataFrame({
...     "idade": [22, 25, 45, 33, 40, 50, 60, 35],
...     "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200],
...     "sexo": ["M", "F", "M", "F", "M", "M", "F", "M"],
...     "target": [0, 0, 1, 0, 1, 1, 0, 1]
... })
>>>
>>> transformations = {
...     "idade": [TreeBinner(max_depth=2), WOEEncoder()],
...     "renda": [CutBinner(bins=[2000, 5000, 8000]), WOEEncoder()],
...     "sexo": [WOEEncoder()],
... }
>>> pipeline = FeaturePipeline(transformations)
>>> df_transformed = pipeline.fit_transform(df.drop(columns="target"), df["target"])
"""

from .binning import TreeBinner, CutBinner
from .encoder import WOEEncoder
from .feature_pipeline import FeaturePipeline

__all__ = [
    "TreeBinner",
    "CutBinner",
    "WOEEncoder",
    "FeaturePipeline",
]