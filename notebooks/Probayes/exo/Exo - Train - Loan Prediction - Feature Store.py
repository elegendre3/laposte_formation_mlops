# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoanPredictions").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC # RandomForest on LOAN PREDICTIONS data

# COMMAND ----------

# MAGIC %md
# MAGIC Load data from legacy hivemetastore. (hive_metastore.default.loans_prediction)
# MAGIC
# MAGIC Explore data, create visuals
# MAGIC we can use columns: "id", "log_person_income" (has to be created), "person_age", "loan_percent_income", "loan_status" as features
# MAGIC
# MAGIC Split, store into FeatureStore (under name "hive_metastore.default.loans_prediction_data_exo_< user >") (creating a column for train_set or test_set is a good idea)
# MAGIC
# MAGIC Train model using sklearn RandomForestClassifier (incompatible with spark dfs)
# MAGIC move to pandas, create x_train, y_train, x_test, y_test
# MAGIC
# MAGIC Create your own experiment and log everything into it
# MAGIC
# MAGIC Log hyperparams, metrics (acc, f1, p, r) and model
# MAGIC
# MAGIC Check out run in Experiment tab
# MAGIC
# MAGIC
# MAGIC BONUS:
# MAGIC Train a similar model but full-spark (using MLlib)

# COMMAND ----------


