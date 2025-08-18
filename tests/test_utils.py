import unittest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from modeltrack.utils.dataframe_adapter import DataFrameAdapter

class TestDataFrameAdapterPandas(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40]
        })

    def setUp(self):
        self.adapter = DataFrameAdapter(self.df.copy())

    def test_backend_detection(self):
        self.assertEqual(self.adapter.backend, "pandas")

    def test_head(self):
        head_df = self.adapter.head(2)
        pd.testing.assert_frame_equal(head_df, self.df.head(2))

    def test_select_columns(self):
        selected = self.adapter.select_columns(["a"])
        pd.testing.assert_frame_equal(selected, self.df[["a"]])

    def test_add_column(self):
        self.adapter.add_column("c", self.df["a"] + 1)
        self.assertIn("c", self.adapter.df.columns)
        expected_series = self.df["a"] + 1
        expected_series.name = "c"
        pd.testing.assert_series_equal(self.adapter.df["c"], expected_series)

    def test_to_pandas(self):
        df_out = self.adapter.to_pandas()
        pd.testing.assert_frame_equal(df_out, self.df)

    def test_invalid_dataframe_type(self):
        with self.assertRaises(ValueError):
            DataFrameAdapter("not a df")


class TestDataFrameAdapterSpark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .master("local[1]") \
            .appName("DataFrameAdapterTests") \
            .getOrCreate()
        cls.df = cls.spark.createDataFrame(
            [(1, 10), (2, 20), (3, 30), (4, 40)],
            ["a", "b"]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.adapter = DataFrameAdapter(self.df)

    def test_backend_detection(self):
        self.assertEqual(self.adapter.backend, "spark")

    def test_head(self):
        head_df = self.adapter.head(2)
        expected = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        pd.testing.assert_frame_equal(head_df, expected)

    def test_select_columns(self):
        selected = self.adapter.select_columns(["a"])
        pd_df = selected.toPandas()
        expected = pd.DataFrame({"a": [1, 2, 3, 4]})
        pd.testing.assert_frame_equal(pd_df, expected)

    def test_add_column(self):
        self.adapter.add_column("c", F.col("a") + 1)
        df_out = self.adapter.df.toPandas()
        expected = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "c": [2, 3, 4, 5]})
        pd.testing.assert_frame_equal(df_out, expected)

    def test_to_pandas(self):
        df_out = self.adapter.to_pandas()
        expected = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        pd.testing.assert_frame_equal(df_out, expected)