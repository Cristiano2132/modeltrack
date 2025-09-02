import unittest
import pandas as pd
import numpy as np
from modeltrack.feature_engineering.encoder import WOEEncoder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================
# 1. Apenas Pandas
# =========================

class TestWOEEncoderPandas(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "cat": ["A", "B", "A", "C", "B", "A", None],
            "target": [1, 0, 1, 0, 1, 0, 1]
        })

    def test_encoder_instance(self):
        encoder = WOEEncoder()
        self.assertIsInstance(encoder, WOEEncoder)

    def test_fit_calculates_woe_values(self):
        encoder = WOEEncoder()
        woe_map = encoder.fit(self.df["cat"], self.df["target"], col_name="cat")
        self.assertIsInstance(woe_map, dict)
        # Eventos e não-eventos para a categoria "A"
        events = 2
        non_events = 1

        # Totais de eventos e não-eventos no dataset
        total_events = 4
        total_non_events = 3

        # Regularização usada no cálculo
        regularization = 1.0

        # Taxa de eventos e não-eventos com regularização
        event_rate = (events + regularization) / (total_events + 2 * regularization)
        non_event_rate = (non_events + regularization) / (total_non_events + 2 * regularization)

        # Cálculo do WOE
        a_woe_calc = np.log(event_rate / non_event_rate)
        # Checa valores exatos do WOE
        a_woe = woe_map["A"]
        self.assertIsInstance(a_woe, float)
        self.assertAlmostEqual(a_woe, a_woe_calc, places=5)

    def test_transform_pandas_values(self):
        encoder = WOEEncoder()
        encoder.fit(self.df["cat"], self.df["target"], col_name="cat")
        out = encoder._transform_pandas(self.df["cat"], col_name="cat")
        for idx, val in enumerate(self.df["cat"]):
            expected = encoder.woe_map["cat"].get(val, encoder.unseen_value)
            self.assertAlmostEqual(out.iloc[idx], expected)

    def test_transform_pandas_unseen(self):
        encoder = WOEEncoder()
        encoder.fit(self.df["cat"], self.df["target"], col_name="cat")
        new_series = pd.Series(["A", "D", None])
        out = encoder._transform_pandas(new_series, col_name="cat")
        self.assertEqual(out.iloc[1], encoder.unseen_value)
        self.assertEqual(out.iloc[2], encoder.unseen_value)


# =========================
# 2. Apenas Spark
# =========================

class TestWOEEncoderSpark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("WOEEncoderTest") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.df_pd = pd.DataFrame({
            "cat": ["A", "B", "A", "C", "B", "A", None],
            "target": [1, 0, 1, 0, 1, 0, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_transform_spark_dataframe_values(self):
        encoder = WOEEncoder()
        encoder.fit(self.df_pd["cat"], self.df_pd["target"], col_name="cat")
        df_out = encoder.transform(self.df_spark, columns="cat")
        self.assertIn("cat_woe", df_out.columns)
        # Verifica valores WOE
        out_values = [row["cat_woe"] if row["cat_woe"] is not None else encoder.unseen_value for row in df_out.collect()]
        expected_values = [encoder.woe_map["cat"].get(v, encoder.unseen_value) for v in self.df_pd["cat"]]
        for a, b in zip(out_values, expected_values):
            self.assertAlmostEqual(a, b)


# =========================
# 3. Spark vs Pandas
# =========================
class TestWOEEncoderPandasVsSpark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("WOEEncoderTestCompare") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.df_pd = pd.DataFrame({
            "cat": ["A", "B", "A", "C", "B", "A", None],
            "target": [1, 0, 1, 0, 1, 0, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_pandas_vs_spark_transformation(self):
        encoder = WOEEncoder()
        encoder.fit(self.df_pd["cat"], self.df_pd["target"], col_name="cat")

        # Pandas
        out_pd = encoder._transform_pandas(self.df_pd["cat"], col_name="cat")

        # Spark
        df_out_spark = encoder.transform(self.df_spark, columns="cat")
        out_spark = df_out_spark.select("cat_woe").toPandas()["cat_woe"]
        out_spark = out_spark.fillna(encoder.unseen_value)

        # Compara valores
        for a, b in zip(out_pd.fillna(encoder.unseen_value), out_spark):
            self.assertAlmostEqual(a, b)
            
            
# =========================
# 4. Testes para transform_dataframe
# =========================

class TestWOEEncoderTransformDataFrame(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("WOEEncoderTestTransformDF") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.df_pd = pd.DataFrame({
            "cat": ["A", "B", "A", "C", "B", "A", None],
            "num": ["X", "Y", "X", "Z", "Y", "X", "Z"],
            "target": [1, 0, 1, 0, 1, 0, 1]
        })
        self.df_spark = self.spark.createDataFrame(self.df_pd)

    def test_transform_dataframe_pandas(self):
        encoder = WOEEncoder()
        encoder.fit(self.df_pd["cat"], self.df_pd["target"], col_name="cat")
        encoder.fit(self.df_pd["num"], self.df_pd["target"], col_name="num")

        out_pd = encoder.transform_dataframe(self.df_pd, features=["cat", "num"], suffix="_woe")
        self.assertIn("cat_woe", out_pd.columns)
        self.assertIn("num_woe", out_pd.columns)

        # valida se valores batem com o mapeamento
        for col in ["cat", "num"]:
            expected = self.df_pd[col].map(lambda v: encoder.woe_map[col].get(v, encoder.unseen_value))
            pd.testing.assert_series_equal(
                out_pd[col + "_woe"].fillna(encoder.unseen_value),
                expected.fillna(encoder.unseen_value),
                check_names=False
            )

    def test_transform_dataframe_spark(self):
        encoder = WOEEncoder()
        encoder.fit(self.df_pd["cat"], self.df_pd["target"], col_name="cat")
        encoder.fit(self.df_pd["num"], self.df_pd["target"], col_name="num")

        out_spark = encoder.transform_dataframe(self.df_spark, features=["cat", "num"], suffix="_woe")
        self.assertIn("cat_woe", out_spark.columns)
        self.assertIn("num_woe", out_spark.columns)

        out_pd = out_spark.select("cat", "num", "cat_woe", "num_woe").toPandas()
        for col in ["cat", "num"]:
            expected = self.df_pd[col].map(lambda v: encoder.woe_map[col].get(v, encoder.unseen_value))
            pd.testing.assert_series_equal(
                out_pd[col + "_woe"].fillna(encoder.unseen_value),
                expected.fillna(encoder.unseen_value),
                check_names=False
            )