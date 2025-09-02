import sys
from pathlib import Path
import pandas as pd

from modeltrack.data.load_data import load_data
from modeltrack.feature_engineering.feature_engineering import (
    TreeCategizer,
    WOEEncoder,
    custom_cut,
)
from modeltrack.evaluation.metrics import get_report_metrics
from modeltrack.utils import get_summary
from modeltrack.models.model_builder import ModelBuilder


# # Configurar paths
# BASE_DIR = Path(__file__).resolve().parent.parent
# sys.path.append(str(BASE_DIR / "src"))
# sys.path.append(str(BASE_DIR / "data"))


if __name__ == "__main__":

    # data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    data_path = "data/raw/diabetes.csv"
    df = load_data(data_path)
    # Pré-processamento
    features = df.columns[:-1]
    label = df.columns[-1]
    train_index = df.sample(frac=0.8, random_state=42).index
    test_index = df.drop(train_index).index
    print("Resumo inicial dos dados:")
    print(get_summary(df))

    # Feature engineering
    bins_dict = {}
    tc = TreeCategizer(df=df.loc[train_index].dropna())
    for feat in features:
        bins_dict[feat] = tc.get_splits(target_column=label, feature_column=feat)

    for feat in features:
        df[feat] = custom_cut(df, feat, bins_dict[feat])

    print("Resumo após TreeCategizer e custom_cut:")
    print(get_summary(df))

    encoder = WOEEncoder(df.loc[train_index], label)
    for feat in features:
        encoder.fit(feat)
        df[feat] = df[feat].map(encoder.woe_dict.get(feat))
    print("Resumo após WOE encoding:")
    print(get_summary(df))

    models = ["logistic", "lgbm", "xgb"]
    for model_name in models:

        model_builder = ModelBuilder(
            X=df.loc[train_index][features],
            y=df.loc[train_index][label],
            random_state=42,
        )
        best_params = model_builder.build_optimized_model(
            model_type=model_name, return_best_params=True
        )

        # Treinamento do modelo com otimização de hiperparâmetros
        model_builder = ModelBuilder(
            X=df.loc[train_index][features],
            y=df.loc[train_index][label],
            random_state=42,
        )
        model = model_builder.build_model(model_type=model_name, params=best_params)
        print("Modelo treinado com sucesso!")

        # Avaliação do modelo no conjunto de treino
        y_pred = model.predict(df[features])
        df_result = pd.DataFrame({"y": df[label], "y_pred": y_pred})
        report_train = get_report_metrics(
            df=df_result, proba_col="y_pred", true_value_col="y", base="train"
        )

        # Avaliação do modelo no conjunto de teste
        y_pred = model.predict(df.loc[test_index][features])
        df_result = pd.DataFrame({"y": df.loc[test_index][label], "y_pred": y_pred})

        report_test = get_report_metrics(
            df=df_result, proba_col="y_pred", true_value_col="y", base="test"
        )

        # Logando as métricas
        report = {}
        report.update(report_train)
        report.update(report_test)

        print(report)
