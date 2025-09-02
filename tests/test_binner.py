import unittest
import pandas as pd
import numpy as np
from modeltrack.feature_engineering import TreeBinner, CutBinner
from modeltrack.feature_engineering.base import BaseBinner
from modeltrack.feature_engineering.binning import Binner

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


class TestBinners(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, np.nan],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, np.nan],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1]
        })
    # ===== Binner =====
    def test_binner_instance(self):
        binner = Binner()
        self.assertIsInstance(binner, BaseBinner)
    
    def test_binner_fit_must_rise_error(self):
        binner = Binner()
        binner.bins = [30, 40, 50]
        with self.assertRaises(NotImplementedError):
            binner.fit(self.df["idade"], self.df["target"])

    def test_binner__make_labels(self):
        binner = Binner()
        labels = binner._make_labels([30, 40, 50])
        self.assertEqual(labels, ["<30", "[30, 40)", "[40, 50)", ">=50"])

    def test_binner_transform(self):
        binner = Binner()
        binner.bins = [30, 40]
        out = binner.transform(self.df["idade"])
        expected= ["<30", "<30", ">=40", "[30, 40)", ">=40", ">=40", ">=40", "[30, 40)", "N/A"]
        
        self.assertIsInstance(out, pd.Series)
        self.assertIn("N/A", out.values)
        self.assertTrue(all((v in binner._make_labels(binner.bins)) or v == "N/A" for v in out.values))
        self.assertTrue(all(out.values == expected))

    # ===== TreeBinner =====
    def test_treebinner_instance(self):
        binner = TreeBinner(max_depth=2)
        self.assertIsInstance(binner, BaseBinner)

    def test_treebinner_fit_bins(self):
        binner = TreeBinner(max_depth=2)
        bins = binner.fit(self.df["idade"], self.df["target"])
        self.assertIsInstance(bins, list)
        self.assertTrue(all(isinstance(b, float) for b in bins))

    def test_treebinner_transform_labels(self):
        binner = TreeBinner(max_depth=2)
        binner.fit(self.df["idade"], self.df["target"])
        out = binner.transform(self.df["idade"])
        self.assertIsInstance(out, pd.Series)
        self.assertIn("N/A", out.values)
        self.assertTrue(all((v in binner._make_labels(binner.bins)) or v == "N/A" for v in out.values))

    def test_treebinner_transform_numeric(self):
        binner = TreeBinner(max_depth=2, return_labels=False)
        binner.fit(self.df["idade"], self.df["target"])
        out = binner.transform(self.df["idade"])
        self.assertIsInstance(out, pd.Series)
        # Verifica que todos valores não-N/A são int
        numeric_values = [v for v in out.values if v != "N/A"]
        self.assertTrue(all(isinstance(v, int) for v in numeric_values))
        self.assertIn("N/A", out.values)

    # ===== CutBinner =====
    def test_cutbinner_instance(self):
            binner = CutBinner(bins=[2000, 5000, 8000])
            self.assertIsInstance(binner, BaseBinner)
    
    def test_cutbinner_fit_bins(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        bins = binner.fit(self.df["renda"], self.df["target"])
        self.assertIsInstance(bins, list)
        self.assertEqual(bins, [2000, 5000, 8000])

    def test_cutbinner_transform_labels(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df["renda"])
        out = binner.transform(self.df["renda"])
        self.assertIsInstance(out, pd.Series)
        self.assertIn("N/A", out.values)
        self.assertTrue(all((v in binner.labels) or v == "N/A" for v in out.values))

    def test_cutbinner_transform_numeric(self):
        binner = CutBinner(bins=[2000, 5000, 8000], return_labels=False)
        binner.fit(self.df["renda"])
        out = binner.transform(self.df["renda"])
        self.assertIsInstance(out, pd.Series)
        numeric_values = [v for v in out.values if v != "N/A"]
        self.assertTrue(all(isinstance(v, int) for v in numeric_values))
        self.assertIn("N/A", out.values)


class TestBinnersSpark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("BinnerTest") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        # Dados de exemplo
        self.df_pd = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, None],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, None],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_treebinner_spark_transform_labels(self):
        binner = TreeBinner(max_depth=2)
        binner.fit(self.df_pd["idade"], self.df_pd["target"])
        
        # Aplica o transform PySpark
        col_transformed = binner._transform_spark(F.col("idade"))
        df_out = self.df_spark.withColumn("idade_binned", col_transformed).collect()
        
        # Converte para lista de valores
        out_values = [row["idade_binned"] for row in df_out]
        # Verifica N/A
        self.assertIn("N/A", out_values)
        # Verifica que todos os valores estão nas labels do binner
        self.assertTrue(all(v in binner._make_labels(binner.bins) or v == "N/A" for v in out_values))

    def test_cutbinner_spark_transform_labels(self):        
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])
        
        col_transformed = binner._transform_spark(F.col("renda"))
        df_out = self.df_spark.withColumn("renda_binned", col_transformed).collect()
        
        out_values = [row["renda_binned"] for row in df_out]
        self.assertIn("N/A", out_values)
        self.assertTrue(all(v in binner.labels or v == "N/A" for v in out_values))
        
    def test_cutbinner_pandas_vs_spark(self):
        """
        Verifica se o resultado do transform é equivalente em Pandas e Spark.
        """
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])

        # Pandas transform
        out_pd = binner.transform(self.df_pd["renda"])

        # Spark transform
        col_transformed = binner._transform_spark(F.col("renda"))
        df_out_spark = self.df_spark.withColumn("renda_binned", col_transformed)
        out_spark = df_out_spark.select("renda_binned").toPandas()["renda_binned"]

        # Comparação
        pd_values = out_pd.fillna("N/A").astype(str).tolist()
        spark_values = out_spark.fillna("N/A").astype(str).tolist()

        self.assertEqual(pd_values, spark_values)