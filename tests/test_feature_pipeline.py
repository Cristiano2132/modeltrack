import unittest
import shutil
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession

from modeltrack.feature_engineering.feature_pipeline import FeaturePipelineDev, FeaturePipelinePrd


class TestFeaturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Criar SparkSession
        cls.spark = SparkSession.builder.master("local[1]").appName("TestPipeline").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        # Criar diretório temporário para salvar JSONs
        self.artifacts_path = Path("test_artifacts")
        if self.artifacts_path.exists():
            shutil.rmtree(self.artifacts_path)

        # Dados de teste (20 linhas)
        self.df = pd.DataFrame({
            "idade": [22, 25, 45, 33, 40, 50, 60, 35, None, 28,
                    31, 42, 55, 38, None, 47, 29, 52, 61, 36],
            "renda": [1500, 1800, 7000, 2500, 3000, 10000, 8000, 1200, None, 2200,
                    2700, 6500, 7200, 3100, 4000, 5600, 1950, 8800, None, 3300],
            "sexo": ["M", "F", "M", "F", "M", "M", "F", "M", "F", "M",
                    "F", "M", "F", "M", "M", "F", "M", "F", "M", "F"],
            "target": [0, 0, 1, 0, 1, 1, 0, 1, 1, 0,
                    1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        })

    def tearDown(self):
        if self.artifacts_path.exists():
            shutil.rmtree(self.artifacts_path)

    def test_dev_pipeline_fit_transform_and_save(self):
        # Pipeline DEV
        pipeline_dev = FeaturePipelineDev(max_depth=2)
        df_transformed = pipeline_dev.fit_transform(
            df=self.df,
            features_binning=["idade", "renda"],
            features_encoding=["idade", "renda", "sexo"],
            target="target"
        )

        # Checar se colunas *_binned e *_woe foram criadas
        for col in ["idade", "renda"]:
            self.assertIn(col + "_binned", df_transformed.columns)
        for col in ["idade", "renda", "sexo"]:
            self.assertIn(col + "_woe", df_transformed.columns)

        # Salvar artifacts
        pipeline_dev.save(self.artifacts_path)

        self.assertTrue((self.artifacts_path / "binner.json").exists())
        self.assertTrue((self.artifacts_path / "encoder.json").exists())
        

    def test_prd_pipeline_transform(self):
        # Primeiro faz fit DEV e salva
        pipeline_dev = FeaturePipelineDev(max_depth=2)
        pipeline_dev.fit_transform(
            df=self.df,
            features_binning=["idade", "renda"],
            features_encoding=["idade", "renda", "sexo"],
            target="target"
        )
        pipeline_dev.save(self.artifacts_path)

        # PRD
        pipeline_prd = FeaturePipelinePrd(self.artifacts_path)
        df_prd_transformed = pipeline_prd.transform(
            df=self.df,
            features_binning=["idade", "renda"],
            features_encoding=["idade", "renda", "sexo"]
        )

        # Checar se colunas *_binned e *_woe foram criadas
        for col in ["idade", "renda"]:
            self.assertIn(col + "_binned", df_prd_transformed.columns)
        for col in ["idade", "renda", "sexo"]:
            self.assertIn(col + "_woe", df_prd_transformed.columns)

        # Checar se valores não nulos
        self.assertFalse(df_prd_transformed["idade_binned"].isnull().all())
        self.assertFalse(df_prd_transformed["idade_woe"].isnull().all())

    def test_spark_pipeline_transform(self):
        df_spark = self.spark.createDataFrame(self.df)

        # Fit DEV e salvar
        pipeline_dev = FeaturePipelineDev(max_depth=2)
        pipeline_dev.fit_transform(
            df=self.df,
            features_binning=["idade", "renda"],
            features_encoding=["idade", "renda", "sexo"],
            target="target"
        )
        pipeline_dev.save(self.artifacts_path)

        # PRD
        pipeline_prd = FeaturePipelinePrd(self.artifacts_path)
        df_spark_transformed = pipeline_prd.transform(
            df=df_spark,
            features_binning=["idade", "renda"],
            features_encoding=["idade", "renda", "sexo"]
        )

        # Checar se colunas *_binned e *_woe existem
        self.assertTrue(all(col in df_spark_transformed.columns for col in
                            ["idade_binned", "renda_binned", "idade_woe", "renda_woe", "sexo_woe"]))

