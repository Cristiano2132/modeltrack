import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

class DataFrameAdapter:
    def __init__(self, df):
        if isinstance(df, pd.DataFrame):
            self.backend = "pandas"
        elif isinstance(df, SparkDataFrame):
            self.backend = "spark"
        else:
            raise ValueError("Unsupported dataframe type")
        self.df = df

    def head(self, n=5):
        if self.backend == "pandas":
            return self.df.head(n)
        else:
            return self.df.limit(n).toPandas()

    def select_columns(self, columns):
        if self.backend == "pandas":
            return self.df[columns]
        else:
            return self.df.select(*columns)

    def add_column(self, column_name, series_or_expr):
        if self.backend == "pandas":
            self.df[column_name] = series_or_expr
        else:
            self.df = self.df.withColumn(column_name, series_or_expr)
        return self

    def to_pandas(self):
        if self.backend == "pandas":
            return self.df
        else:
            return self.df.toPandas()