from pyspark.sql import SparkSession

def main():
    # Criando a sessão Spark
    spark = SparkSession.builder \
        .appName("TestePySpark") \
        .master("local[*]") \
        .getOrCreate()

    # Criando um DataFrame simples
    dados = [("Alice", 29), ("Bob", 31), ("Cris", 40)]
    colunas = ["nome", "idade"]

    df = spark.createDataFrame(dados, colunas)

    print("\n### DataFrame criado ###")
    df.show()

    print("\n### Estatísticas ###")
    df.describe().show()

    # Finalizando
    spark.stop()

if __name__ == "__main__":
    main()