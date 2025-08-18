import pandas as pd
from pyspark.sql import functions as F
from modeltrack.features.base import BaseFeatureTransformer
from modeltrack.utils.dataframe_adapter import DataFrameAdapter

class QuantileBinning(BaseFeatureTransformer):
    def __init__(self, column, n_bins=5):
        super().__init__()
        self.column = column
        self.n_bins = n_bins
        self.bin_edges = None

    def fit(self, df):
        df_adapter = DataFrameAdapter(df)
        if df_adapter.backend == "pandas":
            self.bin_edges = pd.qcut(df_adapter.df[self.column], q=self.n_bins, retbins=True, duplicates='drop')[1]
        else:
            quantiles = [i / self.n_bins for i in range(self.n_bins + 1)]
            self.bin_edges = df_adapter.df.approxQuantile(self.column, quantiles, 0.01)
        self.fitted = True

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Transformer not fitted yet")
        df_adapter = DataFrameAdapter(df)
        if df_adapter.backend == "pandas":
            df_adapter.df[self.column + "_binned"] = pd.cut(df_adapter.df[self.column], bins=self.bin_edges, include_lowest=True)
        else:
            expr = None
            for i in range(len(self.bin_edges) - 1):
                condition = (F.col(self.column) >= self.bin_edges[i]) & (F.col(self.column) <= self.bin_edges[i + 1])
                expr = F.when(condition, f"bin_{i}") if expr is None else expr.when(condition, f"bin_{i}")
            expr = expr.otherwise(None)
            df_adapter.df = df_adapter.df.withColumn(self.column + "_binned", expr)
        return df_adapter.df