# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoanPredictions").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC # Load LOAN PREDICTIONS data
# MAGIC ## - Explore
# MAGIC ## - Create and Store features
# MAGIC ## - Model
# MAGIC ## - Compute accuracy
# MAGIC ## - Deploy

# COMMAND ----------

# df = spark.table("training.loans_prediction.train") # from unity catalog, disabled in "no isolation shared" clusters
# df = spark.table("hive_metastore.default.augmented_loans_prediction") # from hivemetastore legacy
df = spark.sql("SELECT * FROM hive_metastore.default.loans_prediction")  # from hive also, but via custom sql
display(df)

# COMMAND ----------

# spark processing
from pyspark.sql.functions import avg

grouped_df = df.groupBy("loan_status").agg(
    avg("person_age").alias("avg_person_age"),
    avg("person_income").alias("avg_person_income"),
    avg("loan_amnt").alias("avg_loan_amount"),
    avg("loan_percent_income").alias("avg_loan_percent_income"),
    avg("cb_person_cred_hist_length").alias("avg_credit_length"),
    avg("loan_int_rate").alias("avg_loan_interest_rate"),
    avg("person_emp_length").alias("avg_person_emp_length"),
)

display(grouped_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split train-test and store split as a column

# COMMAND ----------

from pyspark.sql.functions import lit, when
from pyspark.sql import DataFrame

# Split the dataframe into train and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=22)

# Add a column specifying if the row belongs to train_set or test_set
train_df = train_df.withColumn("set_type", lit("train"))
test_df = test_df.withColumn("set_type", lit("test"))

# Union the train and test dataframes to keep the original dataframe full
full_df_with_set_type = train_df.union(test_df)

display(full_df_with_set_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Features and store into feature store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import log

# create feature as log(income)
full_df_with_set_type = full_df_with_set_type.withColumn("log_person_income", log("person_income"))

# Initialize the Feature Store client
fs = FeatureStoreClient()

# Define the feature table name
feature_table_name = "hive_metastore.default.loans_prediction_data"

# Create or update the feature table
fs.create_table(
    name=feature_table_name,
    primary_keys=["id"],
    df=full_df_with_set_type.select("id", "log_person_income", "person_age", "loan_percent_income", "loan_status", "set_type"),
    description="Feature table for loans prediction"
)

# Write the new feature to the feature store
fs.write_table(
    name=feature_table_name,
    df=full_df_with_set_type.select("id", "log_person_income","person_age", "loan_percent_income", "loan_status", "set_type"),
    # mode="merge",
    mode="overwrite"
)

# COMMAND ----------

from pyspark.sql.functions import col

# load from feature store
# feature_table_name = "hive_metastore.default.loans_prediction_data"
train_df = fs.read_table(name=feature_table_name).filter(col("set_type") == "train")
test_df = fs.read_table(name=feature_table_name).filter(col("set_type") == "test")

# switch to pandas
train_df_pandas = train_df.toPandas()
test_df_pandas = test_df.toPandas()

# create X and y
num_samples = -1
X_train = train_df_pandas[["person_age", "log_person_income", "loan_percent_income"]][:num_samples].values
y_train = train_df_pandas["loan_status"][:num_samples].values

X_test = test_df_pandas[["person_age", "log_person_income", "loan_percent_income"]].values
y_test = test_df_pandas["loan_status"].values

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)


import mlflow
import mlflow.sklearn


n_estimators = 100
seed = 22

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", seed)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=seed)
    rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    f1 = f1_score(y_test, predicted, average='weighted')
    p = precision_score(y_test, predicted)
    r = recall_score(y_test, predicted)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", p)
    mlflow.log_metric("recall", r)

    print(f'Mean accuracy score: {accuracy:.3}')
    print(f'Weighted F1 score: {f1:.3}')
    print(f'Binary Precision score: {p:.3}')
    print(f'Binary Recall score: {r:.3}')
    print(f'F1 binary score: {(2/(1/r + 1/p)):.3}')

# COMMAND ----------

# Second run for showing MLFlow UI

n_estimators =200
seed = 42

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", seed)
    
    # create model and train
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=seed)
    rf.fit(X_train, y_train)

    # evaluate
    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    f1 = f1_score(y_test, predicted, average='weighted')
    p = precision_score(y_test, predicted)
    r = recall_score(y_test, predicted)

    print(f'Mean accuracy score: {accuracy:.3}')
    print(f'Weighted F1 score: {f1:.3}')
    print(f'Binary Precision score: {p:.3}')
    print(f'Binary Recall score: {r:.3}')
    print(f'F1 binary score: {(2/(1/r + 1/p)):.3}')

    # log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", p)
    mlflow.log_metric("recall", r)

    # log model as an sklearn model - provide sample to auto-set the model signature
    mlflow.sklearn.log_model(rf, "model", input_example=X_train[0].reshape(1, -1))

    # register model
    model_uri = f"runs:/{run.info.run_id}/model"
    model_name = "training_loans_prediction_RandomForestClassifierModel"
    model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# BONUS
# FULL SPARK EXAMPLE
# Note VecAssembler does not work in serverless, High-concurrency cluster.

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, MulticlassClassificationEvaluator)
from pyspark.ml.feature import VectorAssembler


# Define feature columns and label column
feature_cols = ["person_age", "log_person_income", "loan_percent_income"]
label_col = "loan_status"
num_trees = 99
max_depth = 25

# load data
# feature_table_name = "hive_metastore.default.loans_prediction_data"
# train_df = fs.read_table(name=feature_table_name).filter(col("set_type") == "train")
# test_df = fs.read_table(name=feature_table_name).filter(col("set_type") == "test")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", seed)
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Define Random Forest model
    rf = RandomForestClassifier(numTrees=num_trees, maxDepth=max_depth, labelCol=label_col, featuresCol="features")

    # Create a pipeline
    pipeline = Pipeline(stages=[assembler, rf])

    # Fit the model
    model = pipeline.fit(train_df)

    # Make predictions
    predictions = model.transform(test_df)

    # Score
    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
    auroc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
    aupr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})

    multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
    accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
    p = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
    r = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})
    f1 = (2/(1/r + 1/p))

    print(f'Mean accuracy score: {accuracy:.3}')
    print(f'AUROC score: {auroc:.3}')
    print(f'AUPR score: {aupr:.3}')
    print(f'Weighted Precision score: {p:.3}')
    print(f'Weighted Recall score: {r:.3}')
    print(f'Weighted F1 score: {f1:.3}')

    # log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auroc", auroc)
    mlflow.log_metric("aupr", aupr)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", p)
    mlflow.log_metric("recall", r)

    # log Spark x MLlib model
    mlflow.spark.log_model(model, "model")

    # register model
    model_uri = f"runs:/{run.info.run_id}/model"
    model_name = "training_loans_prediction_SparkRandomForestClassifierModel"
    model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
p = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
print(p)

# COMMAND ----------

print(auroc)
print(aupr)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Binary classification metrics
binary_evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
auroc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
aupr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})

multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
p = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
r = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "weightedRecall"})

print(f'AUROC score: {auroc:.3}')
print(f'AUPR score: {aupr:.3}')
print(f'Binary Precision score: {p:.3}')
print(f'Binary Recall score: {r:.3}')
print(f'F1 binary score: {(2/(1/r + 1/p)):.3}')

# COMMAND ----------



# COMMAND ----------


