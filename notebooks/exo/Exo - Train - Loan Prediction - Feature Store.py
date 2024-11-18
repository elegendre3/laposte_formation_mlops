# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoanPredictions").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC # RandomForest on LOAN PREDICTIONS data

# COMMAND ----------

# MAGIC %md
# MAGIC Load data from legacy hivemetastore
# MAGIC
# MAGIC Explore data, create visuals
# MAGIC we can use columns: "id", "log_person_income", "person_age", "loan_percent_income", "loan_status"
# MAGIC
# MAGIC Split, store into FeatureStore (under name "hive_metastore.default.loans_prediction_data_exo_< user >")
# MAGIC
# MAGIC Train model using sklearn RandomForestClassifier
# MAGIC move to pandas, create x_train, y_train, x_test, y_test
# MAGIC
# MAGIC Log hyper params, metrics (acc, f1, p, r) and model
# MAGIC
# MAGIC Check out run in Experiment tab
# MAGIC
# MAGIC
# MAGIC BONUS:
# MAGIC Train a similar model but full-spark (using MLlib)
