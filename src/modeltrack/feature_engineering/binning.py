import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from typing import List, Optional
from .base import BaseBinner
from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Union


class Binner(BaseBinner):
    """
    Base class for feature binning.

    This class provides common functionality for binning numerical features, 
    including handling of missing values and generating category labels.

    Parameters
    ----------
    labels : list of str, optional
        Custom labels for the bins. If None, labels are generated automatically.
    return_labels : bool, default=True
        If True, `transform` returns category labels.
        If False, `transform` returns numeric bin indices.
    """

    def __init__(self, labels: Optional[List[str]] = None, return_labels: bool = True):
        self.bins: List[float] = []
        self.labels: Optional[List[str]] = None
        self.return_labels = return_labels

    def fit(self, X: pd.Series, y: pd.Series) -> None:
        """
        Fit the binner to the data.

        Subclasses should implement this method to determine bin edges.

        Parameters
        ----------
        X : pd.Series
            Feature values to bin.
        y : pd.Series
            Target values used to guide binning (may be ignored for unsupervised binners).

        Returns
        -------
        None
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _make_labels(self, bins: List[float]) -> List[str]:
        """
        Generate default bin labels based on bin edges.

        Parameters
        ----------
        bins : list of float
            Bin edges.

        Returns
        -------
        labels : list of str
            List of string labels for each bin.
        """
        if not bins:
            return []
        labels = [f"<{bins[0]:.6g}"]
        for i in range(len(bins) - 1):
            labels.append(f"[{bins[i]:.6g}, {bins[i+1]:.6g})")
        labels.append(f">={bins[-1]:.6g}")
        return labels

    def _transform_pandas(self, X: pd.Series) -> pd.Series:
        """
        Apply binning to a feature series of pandas.

        Parameters
        ----------
        X : pd.Series
            Feature values to transform.

        Returns
        -------
        pd.Series
            Binned feature, either as labels or numeric indices.
            Missing values are replaced with "N/A".
        """
        if not self.bins:
            return X.fillna("N/A") if self.return_labels else X

        bins = [-float("inf")] + self.bins + [float("inf")]
        labels = self._make_labels(self.bins) if self.return_labels else list(range(len(bins)-1))

        def categorize(x):
            if pd.isna(x):
                return "N/A" if self.return_labels else "N/A"
            for i in range(len(bins)-1):
                if bins[i] <= x < bins[i+1]:
                    return labels[i]
            return labels[-1]

        return X.map(categorize)
    
    def _transform_spark(self, col):
        """
        Transform a Spark DataFrame column using the defined bins.
        """
        if not self.bins:
            return F.when(F.isnan(col), F.lit("N/A")).otherwise(col)

        bins = [-float("inf")] + self.bins + [float("inf")]
        labels = self._make_labels(self.bins) if self.return_labels else list(range(len(bins)-1))

        expr = None
        for i in range(len(bins)-1):
            lower, upper = bins[i], bins[i+1]
            label = labels[i]
            condition = (col >= lower) & (col < upper)
            expr = F.when(condition, F.lit(label)) if expr is None else expr.when(condition, F.lit(label))

        # missing values
        expr = expr.when(F.isnan(col), F.lit("N/A")).otherwise(F.lit(labels[-1]))
        return expr

    def transform(self, X)->Union[pd.Series, SparkDataFrame]:
        """
        Transform the feature series using the defined bins.

        Parameters
        ----------
        X : pd.Series or pyspark.sql.Column
            Input feature to bin.

        Returns
        -------
        pd.Series or pyspark.sql.Column
            Binned feature.
        """
        if isinstance(X, pd.Series):
            return self._transform_pandas(X)
        
        elif "pyspark.sql" in str(type(X)):
            return self._transform_spark(X)
        else:
            raise TypeError("X must be a pandas Series or PySpark Column")

    def transform_dataframe(self, df: Union[pd.DataFrame, SparkDataFrame], features: List[str], suffix: str = "_binned"
                            ) -> Union[pd.DataFrame, SparkDataFrame]:
        if isinstance(df, pd.DataFrame):
            out = df.copy()
            for col in features:
                out[col + suffix] = self._transform_pandas(out[col])
            return out
        elif isinstance(df, SparkDataFrame):
            new_cols = [
                self._transform_spark(F.col(c)).alias(c + suffix) for c in features
            ]
            return df.select("*", *new_cols)
        else:
            raise TypeError("df must be a pandas or Spark DataFrame")


class TreeBinner(Binner):
    """
    Decision tree-based binner.

    Uses a decision tree classifier to determine optimal splits based on a target variable.

    Parameters
    ----------
    max_depth : int, default=2
        Maximum depth of the decision tree.
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes for the tree.
    random_state : int, default=1234
        Random seed for reproducibility.
    return_labels : bool, default=True
        If True, `transform` returns category labels; otherwise numeric indices.
    """

    def __init__(self, max_depth: int = 2, max_leaf_nodes: Optional[int] = None, random_state: int = 1234, return_labels: bool = True):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.bins: List[float] = []
        self.labels: Optional[List[str]] = None
        self.return_labels = return_labels

    def fit(self, X: pd.Series, y: pd.Series) -> List[float]:
        """
        Fit a decision tree to determine optimal bin edges.

        Parameters
        ----------
        X : pd.Series
            Feature values to bin.
        y : pd.Series
            Target variable guiding the splits.

        Returns
        -------
        bins : list of float
            List of determined bin edges.
        """
        mask = X.notna()
        X_train = X.loc[mask].values.reshape(-1, 1)
        y_train = y.loc[mask]

        if len(X_train) == 0:
            self.bins = []
            return self.bins

        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)

        tree = model.tree_
        internal_node_mask = tree.feature != -2
        splits = tree.threshold[internal_node_mask]
        self.bins = sorted(set(float(s) for s in splits if s != -2.0))
        return self.bins


class CutBinner(Binner):
    """
    Fixed-cut binner.

    Creates bins based on pre-defined thresholds. Useful for domain-specific discretization.

    Parameters
    ----------
    bins : list of float
        Pre-defined bin edges.
    labels : list of str, optional
        Custom labels for bins. If None, labels are generated automatically.
    nan_value : any, optional
        Value to assign for missing data. Defaults to "N/A" in transform.
    return_labels : bool, default=True
        If True, `transform` returns category labels; otherwise numeric indices.
    """

    def __init__(self, bins: List[float], labels: Optional[List[str]] = None, nan_value: Optional[any] = None, return_labels: bool = True):
        self.requested_bins = sorted(bins)
        self.labels = labels
        self.nan_value = nan_value
        self.bins: List[float] = []
        self.return_labels = return_labels

    def fit(self, X: pd.Series, y: pd.Series = None) -> List[float]:
        """
        Fit the binner by validating and adjusting predefined bin edges to the data.

        Parameters
        ----------
        X : pd.Series
            Feature values to bin.
        y : pd.Series, optional
            Ignored; included for compatibility with pipeline usage.

        Returns
        -------
        bins : list of float
            Validated and sorted bin edges.
        """
        if X.dropna().empty:
            self.bins = []
            return self.bins

        min_v, max_v = X.dropna().min(), X.dropna().max()
        self.bins = sorted(set(c for c in self.requested_bins if min_v < c < max_v))

        if self.labels is None and self.bins:
            self.labels = [f"<{self.bins[0]:.6g}"]
            for i in range(len(self.bins) - 1):
                self.labels.append(f"[{self.bins[i]:.6g}, {self.bins[i+1]:.6g})")
            self.labels.append(f">={self.bins[-1]:.6g}")

        return self.bins