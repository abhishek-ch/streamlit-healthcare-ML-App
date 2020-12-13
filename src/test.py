from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from src.com.abc.webapp.sci import scikit_func as sf

spark = SparkSession.builder.appName('healthcare_pyspark') \
    .config("spark.pyspark.python", "python3") \
    .config("spark.pyspark.driver.python", "python3") \
    .getOrCreate()


def data_prep(dataframe):
    inputCols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
                 'active']
    assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df_temp = assembler.transform(dataframe).select("features", "cardio")
    df_temp.show(10)

    (trainingData, testData) = df_temp.randomSplit([0.7, 0.3])
    training(trainingData, testData)


def training(train_df, test_df):
    dt = DecisionTreeClassifier(labelCol="cardio", featuresCol="features")
    model = dt.fit(train_df)
    predictions = model.transform(test_df)
    predictions.show(2)
    evaluator = BinaryClassificationEvaluator(labelCol="cardio", rawPredictionCol="rawPrediction")
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)


def main():
    sf_df = sf.load_data()
    # print(sf_df.head())
    dataframe = spark.createDataFrame(sf_df).drop("id")
    print(dataframe.show(10))
    data_prep(dataframe)


if __name__ == '__main__':
    main()
