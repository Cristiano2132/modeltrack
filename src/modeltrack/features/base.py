from abc import ABC, abstractmethod
from modeltrack.utils.dataframe_adapter import DataFrameAdapter

class BaseFeatureTransformer(ABC):
    def __init__(self):
        self.fitted = False

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def transform(self, df):
        pass

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)