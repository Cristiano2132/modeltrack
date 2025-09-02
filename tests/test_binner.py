import unittest
import pandas as pd
import numpy as np
from modeltrack.feature_engineering import TreeBinner, CutBinner
from modeltrack.feature_engineering.base import BaseBinner
from modeltrack.feature_engineering.binning import Binner
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# =========================
# 1. Apenas Pandas
# =========================
class TestBinnersPandas(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, np.nan],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, np.nan],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1]
        })

    def test_instance_Binner(self):
        binner = Binner()
        cut_binner = CutBinner(bins=[2000, 5000, 8000])
        self.assertIsInstance(binner, BaseBinner)
        self.assertIsInstance(cut_binner, BaseBinner)

    def test_binner_transform(self):
        binner = Binner()
        binner.bins = [30, 40]
        out = binner.transform(self.df["idade"])
        expected= ["<30", "<30", ">=40", "[30, 40)", ">=40", ">=40", ">=40", "[30, 40)", "N/A"]
        self.assertTrue(all(out.values == expected))

    def test_treebinner_transform_labels(self):
        binner = TreeBinner(max_depth=2)
        binner.fit(self.df["idade"], self.df["target"])
        out = binner.transform(self.df["idade"])
        self.assertTrue(all((v in binner._make_labels(binner.bins)) or v == "N/A" for v in out.values))

    def test_cutbinner_transform_labels(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df["renda"])
        out = binner.transform(self.df["renda"])
        self.assertTrue(all((v in binner.labels) or v == "N/A" for v in out.values))

    def test_transform_dataframe_pandas(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df["renda"])
        features = ["idade", "renda"]
        out = binner.transform_dataframe(self.df, features)
        # Checa se colunas binned existem
        for f in features:
            self.assertIn(f + "_binned", out.columns)
            # Checa se valores estão dentro das labels ou N/A
            self.assertTrue(all((v in binner.labels) or v == "N/A" for v in out[f + "_binned"]))

# =========================
# 2. Apenas Spark
# =========================
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
        self.df_pd = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, None],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, None],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_treebinner_spark_transform_labels(self):
        binner = TreeBinner(max_depth=2)
        binner.fit(self.df_pd["idade"], self.df_pd["target"])
        col_transformed = binner._transform_spark(F.col("idade"))
        df_out = self.df_spark.withColumn("idade_binned", col_transformed).collect()
        out_values = [row["idade_binned"] for row in df_out]
        self.assertIn("N/A", out_values)

    def test_cutbinner_spark_transform_labels(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])
        col_transformed = binner._transform_spark(F.col("renda"))
        df_out = self.df_spark.withColumn("renda_binned", col_transformed).collect()
        out_values = [row["renda_binned"] for row in df_out]
        self.assertIn("N/A", out_values)

    def test_transform_dataframe_spark(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])
        features = ["idade", "renda"]
        out = binner.transform_dataframe(self.df_spark, features)
        # Checa se colunas binned existem
        for f in features:
            self.assertIn(f + "_binned", out.columns)
            # Checa se valores estão dentro das labels ou N/A
            values = out.select(f + "_binned").toPandas()[f + "_binned"].fillna("N/A").astype(str).tolist()
            self.assertTrue(all((v in binner.labels) or v == "N/A" for v in values))

# =========================
# 3. Spark vs Pandas
# =========================
class TestBinnersSparkVsPandas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("BinnerConsistencyTest") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.df_pd = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, None],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, None],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_cutbinner_pandas_vs_spark(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])
        # Pandas
        out_pd = binner.transform(self.df_pd["renda"])
        # Spark
        col_transformed = binner._transform_spark(F.col("renda"))
        df_out_spark = self.df_spark.withColumn("renda_binned", col_transformed)
        out_spark = df_out_spark.select("renda_binned").toPandas()["renda_binned"]
        # Comparação
        pd_values = out_pd.fillna("N/A").astype(str).tolist()
        spark_values = out_spark.fillna("N/A").astype(str).tolist()
        self.assertEqual(pd_values, spark_values)

    def test_transform_dataframe_multiple_columns(self):
        binner = CutBinner(bins=[2000, 5000, 8000])
        binner.fit(self.df_pd["renda"])
        features = ["idade", "renda"]
        # Pandas
        out_pd = binner.transform_dataframe(self.df_pd, features)
        # Spark
        out_spark = binner.transform_dataframe(self.df_spark, features)
        # Comparação por coluna
        for f in features:
            pd_values = out_pd[f + "_binned"].fillna("N/A").astype(str).tolist()
            spark_values = out_spark.select(f + "_binned").toPandas()[f + "_binned"].fillna("N/A").astype(str).tolist()
            self.assertEqual(pd_values, spark_values)