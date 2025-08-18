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

class TestBaseFeatureTransformer(unittest.TestCase):
    def test_abstract_class(self):
        with self.assertRaises(TypeError):
            # Não pode instanciar diretamente BaseFeatureTransformer
            BaseFeatureTransformer()