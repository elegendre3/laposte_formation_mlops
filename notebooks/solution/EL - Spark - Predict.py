# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoanPredictions").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC # Load LOAN data data using DBX FeatureStore

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load test set for FS

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
from pyspark.sql.functions import col

# Initialize the Feature Store client
fs = FeatureStoreClient()

# Define the feature table name
feature_table_name = "hive_metastore.default.loans_prediction_data"

# Retrieve features of train_set
test_df = fs.read_table(name=feature_table_name).filter(col("set_type") == "test")

display(test_df)

# COMMAND ----------

# Switch to pandas
# Create X and y

test_df_pandas = test_df.toPandas()
X_test = test_df_pandas[["person_age", "log_person_income", "loan_percent_income"]].values
y_test = test_df_pandas["loan_status"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model as plain sklearn, pass test_set as pandas df to model.predict 

# COMMAND ----------

# Load registered model and Predict
import mlflow
logged_model = 'runs:/37b2f4c4374c4bd29f3b6cc79750d55f/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
preds_test = loaded_model.predict(pd.DataFrame(X_test))
display(preds_test)


from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
accuracy = accuracy_score(y_test, preds_test)
f1 = f1_score(y_test, preds_test, average='weighted')
p = precision_score(y_test, preds_test)
r = recall_score(y_test, preds_test)

print(f'Mean accuracy score: {accuracy:.3}')
print(f'Weighted F1 score: {f1:.3}')
print(f'Binary Precision score: {p:.3}')
print(f'Binary Recall score: {r:.3}')
print(f'F1 binary score: {(2/(1/r + 1/p)):.3}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model as UDF and make predictions in Spark

# COMMAND ----------

# Load as Spark UDF
import mlflow
from pyspark.sql.functions import struct, col

logged_model = "runs:/37b2f4c4374c4bd29f3b6cc79750d55f/model"

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# Predict on a Spark DataFrame.
cols = ["person_age", "log_person_income", "loan_percent_income"]
test_pred_df = test_df.withColumn(
    "predictions", loaded_model(struct(*map(col, test_df[cols].columns)))[0]
)
display(test_pred_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call deployed model

# COMMAND ----------

import json
import os
import requests

url = 'https://adb-34351534749836.16.azuredatabricks.net/serving-endpoints/EL_LoanPrediction_Model_1/invocations'
# headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
headers = {'Authorization': f"Bearer {dbutils.secrets.get(scope='formation-docaposte', key='el_db_api_key')}", 'Content-Type': 'application/json'}

ds_dict = {'dataframe_split': test_df_pandas[["person_age", "log_person_income", "loan_percent_income"]].iloc[10:20].to_dict(orient='split')}
data_json = json.dumps(ds_dict, allow_nan=True)

response = requests.request(method='POST', headers=headers, url=url, data=data_json)
if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
else:
    print('Success - 200')
print(json.dumps(response.json(), indent=4))

# COMMAND ----------



# COMMAND ----------

# BONUS
# 
# # FULL SPARK EXAMPLE - VecAssembler does not work in High-concurrency cluster.
# from pyspark.ml import Pipeline
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import VectorAssembler

# # Define feature columns and label column
# feature_cols = ["person_age", "person_income", "loan_percent_income"]
# label_col = "loan_status"

# # Assemble features
# assembler = VectorAssembler(inputCols=feature_cols, outputCol=label_col)


# # Define Random Forest model
# rf = RandomForestClassifier(labelCol=label_col, featuresCol="features")

# # Create a pipeline
# pipeline = Pipeline(stages=[assembler, rf])

# # Fit the model
# model = pipeline.fit(train_df)

# # Make predictions
# predictions = model.transform(test_df)

# display(predictions)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


