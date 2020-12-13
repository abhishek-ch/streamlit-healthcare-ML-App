from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


def get_spark_session():
    """
    Creating Spark Session from Localhost
    TODO: Technically, this must be changed to support running it oon cluster by using master url
    :return:
    """
    return SparkSession.builder.appName('healthcare_pyspark') \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.driver.python", "python3") \
        .getOrCreate()


def prepare_dataset(spark, pd_data):
    dataframe = spark.createDataFrame(pd_data).drop("id")
    inputCols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco',
                 'active']
    assembler = VectorAssembler(inputCols=inputCols, outputCol='features')
    df_temp = assembler.transform(dataframe).select("features", "cardio")

    (trainingData, testData) = df_temp.randomSplit([0.7, 0.3])
    return trainingData, testData


def get_model(classifier, params):
    """
    TODO: Add support for params in pyspark ML
    :param classifier:
    :param params:
    :return:
    """
    if classifier == 'Decision Tree':
        return DecisionTreeClassifier(labelCol="cardio", featuresCol="features")
    return RandomForestClassifier(labelCol="cardio", featuresCol="features")


def training(spark, classifier, train_df, test_df):
    pyspark_classifier = get_model(classifier, None)
    model = pyspark_classifier.fit(train_df)
    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="cardio", rawPredictionCol="rawPrediction")
    accuracy = evaluator.evaluate(predictions)
    return accuracy


def get_sidebar_classifier():
    return 'Decision Tree', 'Random Forest'
