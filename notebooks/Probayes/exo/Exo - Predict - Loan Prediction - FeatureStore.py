# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoanPredictions").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC # Load LOAN data data using DBX FeatureStore, call logged model

# COMMAND ----------

# MAGIC %md
# MAGIC Read test data from FeatureStore ("hive_metastore.default.loans_prediction_data_exo_< user >")
# MAGIC
# MAGIC Load stored model (basic) (get URI from UI)
# MAGIC Make predictions (w. pandas)
# MAGIC
# MAGIC Load stored model (as spark UDF)
# MAGIC Make prediction with spark directly
# MAGIC
# MAGIC Call deployed model (you can use secret: el_db_api_key)
# MAGIC
# MAGIC

# COMMAND ----------


