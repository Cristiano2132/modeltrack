# feature_engineering/encoder.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame, Column
from .base import Transformer

class WOEEncoder(Transformer):
    """
    Weight of Evidence (WOE) encoder for binary classification.

    Can encode both Pandas Series/DataFrame and Spark Column/DataFrame.

    Attributes
    ----------
    regularization : float
        Laplace smoothing to avoid division by zero.
    unseen_value : float
        Value assigned to unseen categories during transform.
    woe_map : dict
        Dictionary storing WOE mapping for each column: {col_name: {category: woe}}.
    """

    def __init__(self, regularization: float = 1.0, unseen_value: float = 0.0):
        """
        Initialize WOEEncoder.

        Parameters
        ----------
        regularization : float, default=1.0
            Laplace smoothing factor to avoid division by zero.
        unseen_value : float, default=0.0
            Value assigned to unseen categories during transform.
        """
        self.regularization = regularization
        self.unseen_value = unseen_value
        self.woe_map: Dict[str, Dict[Any, float]] = {}

    def import_config(self, woe_map: dict):
        """
        Import WOE mapping for the encoder.

        Parameters
        ----------
        woe_map : dict
            Mapping of categories to WOE values for each feature.
        """
        self.woe_map = woe_map

    def fit(self, X: pd.Series, y: pd.Series, col_name: str) -> Dict[Any, float]:
        """
        Fit WOE mapping for a single column.

        Parameters
        ----------
        X : pd.Series
            Feature column to encode.
        y : pd.Series
            Binary target column (0/1).
        col_name : str
            Name of the column to store mapping in woe_map.

        Returns
        -------
        dict
            Mapping from category to WOE values.
        """
        if len(y.unique()) != 2:
            raise ValueError("Target must be binary (0/1)")

        total_events = y.sum()
        total_non_events = len(y) - total_events
        mapping = {}

        for cat in X.dropna().unique():
            mask = X == cat
            events = int(y[mask].sum())
            non_events = int(mask.sum()) - events
            event_rate = (events + self.regularization) / (total_events + 2 * self.regularization)
            non_event_rate = (non_events + self.regularization) / (total_non_events + 2 * self.regularization)
            mapping[cat] = float(np.log(event_rate / non_event_rate))

        self.woe_map[col_name] = mapping
        return mapping

    def _transform_pandas(self, X: pd.Series, col_name: str) -> pd.Series:
        """Apply WOE mapping to a Pandas Series."""
        return X.map(lambda v: self.woe_map.get(col_name, {}).get(v, self.unseen_value))

    def _transform_spark(self, col: Column, col_name: str) -> Column:
        """Apply WOE mapping to a Spark Column."""
        if col_name not in self.woe_map:
            raise ValueError(f"WOEEncoder has not been fit for column '{col_name}' yet")
        expr = None
        for cat, woe in self.woe_map[col_name].items():
            condition = col == cat
            expr = F.when(condition, F.lit(woe)) if expr is None else expr.when(condition, F.lit(woe))
        expr = expr.otherwise(F.lit(self.unseen_value))
        return expr

    def transform(
        self,
        X: Union[pd.Series, pd.DataFrame, Column, SparkDataFrame],
        columns: Optional[Union[str, list[str]]] = None,
        suffix: str = "_woe"
    ) -> Union[pd.Series, pd.DataFrame, Column, SparkDataFrame]:
        """
        Transform one or multiple columns using WOE encoding.

        Parameters
        ----------
        X : pd.Series, pd.DataFrame, Spark Column, or Spark DataFrame
            Feature(s) to transform.
        columns : str or list of str, optional
            Columns to transform (required for DataFrames).
        suffix : str, optional
            Suffix to add to transformed columns (default is "_woe").

        Returns
        -------
        Transformed Series, DataFrame, or Column(s)
        """
        # Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            if columns is None:
                raise ValueError("columns must be provided for Pandas DataFrame")
            if isinstance(columns, str):
                columns = [columns]
            df_out = X.copy()
            for col in columns:
                df_out[col + suffix] = self._transform_pandas(X[col], col)
            return df_out

        # Pandas Series (use col_name as mandatory)
        if isinstance(X, pd.Series):
            raise ValueError("For Pandas Series, use transform with col_name via fit")

        # Spark DataFrame
        if "pyspark.sql.dataframe.DataFrame" in str(type(X)):
            if columns is None:
                raise ValueError("columns must be provided for Spark DataFrame")
            if isinstance(columns, str):
                columns = [columns]
            df_out = X
            for col in columns:
                df_out = df_out.withColumn(col + suffix, self._transform_spark(F.col(col), col))
            return df_out

        # Spark Column (use col_name as mandatory)
        if isinstance(X, SparkDataFrame):
            raise ValueError("For Spark Column, use transform with col_name via fit")

        raise TypeError(f"X must be a Pandas Series, Pandas DataFrame, Spark Column, or Spark DataFrame. The current type is {type(X)}")

    def transform_dataframe(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        features: list[str],
        suffix: str = "_woe"
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        Transform multiple columns of a DataFrame using WOE encoding.

        Parameters
        ----------
        df : pd.DataFrame or Spark DataFrame
            DataFrame to transform.
        features : list of str
            List of columns to transform.
        suffix : str
            Suffix to add to transformed columns.

        Returns
        -------
        pd.DataFrame or Spark DataFrame
            DataFrame with additional *_woe columns.
        """
        if isinstance(df, pd.DataFrame):
            df_out = df.copy()
            for col in features:
                df_out[col + suffix] = self._transform_pandas(df[col], col)
            return df_out


        if isinstance(df, pd.DataFrame):
            exprs = [df[col] for col in df.columns]  # mantém todas as colunas originais
            exprs += [
                self._transform_spark(F.col(col), col).alias(col + suffix)
                for col in features
            ]
            return df.select(*exprs)

        raise TypeError(f"df must be a Pandas DataFrame or Spark DataFrame. The current type is {type(df)}")