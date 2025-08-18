import unittest
import pandas as pd
from pyspark.sql import SparkSession
from modeltrack.features.binning import QuantileBinning
from modeltrack.features.base import BaseFeatureTransformer

class TestQuantileBinningPandas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({"feature": [1,2,3,4,5,6,7,8,9,10]})
    
    def setUp(self):
        self.transformer = QuantileBinning(column="feature", n_bins=4)

    def test_fit_transform_creates_binned_column(self):
        df_transformed = self.transformer.fit_transform(self.df)
        self.assertIn("feature_binned", df_transformed.columns)
        self.assertEqual(df_transformed["feature_binned"].isnull().sum(), 0)
        self.assertTrue(self.transformer.fitted)

    def test_fit_only(self):
        self.transformer.fit(self.df)
        self.assertTrue(self.transformer.fitted)

    def test_transform_only_after_fit(self):
        self.transformer.fit(self.df)
        df_transformed = self.transformer.transform(self.df)
        self.assertIn("feature_binned", df_transformed.columns)

    def test_transform_raises_error_if_not_fitted(self):
        with self.assertRaises(RuntimeError):
            self.transformer.transform(self.df)


class TestQuantileBinningSpark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("ModelTrackTests") \
            .getOrCreate()
        cls.df = cls.spark.createDataFrame(
            [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)], ["feature"]
        )
    
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.transformer = QuantileBinning(column="feature", n_bins=4)

    def test_fit_transform_creates_binned_column(self):
        df_transformed = self.transformer.fit_transform(self.df)
        df_pd = df_transformed.toPandas()
        self.assertIn("feature_binned", df_pd.columns)
        self.assertEqual(df_pd["feature_binned"].isnull().sum(), 0)
        self.assertTrue(self.transformer.fitted)

    def test_fit_only(self):
        self.transformer.fit(self.df)
        self.assertTrue(self.transformer.fitted)

    def test_transform_only_after_fit(self):
        self.transformer.fit(self.df)
        df_transformed = self.transformer.transform(self.df)
        df_pd = df_transformed.toPandas()
        self.assertIn("feature_binned", df_pd.columns)

    def test_transform_raises_error_if_not_fitted(self):
        with self.assertRaises(RuntimeError):
            self.transformer.transform(self.df)

class TestBaseFeatureTransformer(unittest.TestCase):
    def test_abstract_class_instantiation(self):
        with self.assertRaises(TypeError):
            BaseFeatureTransformer()

    def test_fitted_property(self):
        class DummyTransformer(BaseFeatureTransformer):
            def fit(self, X):
                self.fitted = True
                return self
            def transform(self, X):
                if not getattr(self, "fitted", False):
                    raise ValueError("Not fitted")
                return X

        transformer = DummyTransformer()
        transformer.fit(pd.DataFrame({"a": [1]}))
        self.assertTrue(transformer.fitted)
        df_out = transformer.transform(pd.DataFrame({"a": [1]}))
        pd.testing.assert_frame_equal(df_out, pd.DataFrame({"a": [1]}))