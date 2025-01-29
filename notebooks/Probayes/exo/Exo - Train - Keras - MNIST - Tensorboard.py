# Databricks notebook source
# MAGIC %md
# MAGIC # Train MNIST model with Keras, monitor w. tensorboard and mlflow
# MAGIC
# MAGIC
# MAGIC Create a new experiment in your workspace, grab experiment_id for mlflow later
# MAGIC
# MAGIC Set up tensorboard (safari blocks iframe? try chrome)
# MAGIC
# MAGIC Load mnist data from keras
# MAGIC Visualize data w. matplotlib
# MAGIC
# MAGIC Create your Keras model (ideally restrain size since there won't be GPUs for everyone)
# MAGIC
# MAGIC Create tensorboard and mlflow callbacks for monitoring epoch per epoch
# MAGIC
# MAGIC (Bonus if possible: Set up for GPU training)
# MAGIC
# MAGIC Use MLFlow to log params, metrics and model
# MAGIC
# MAGIC Check tensorboard UI
# MAGIC
# MAGIC Deploy model in UI or code
# MAGIC
# MAGIC Call deployed model for inference
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------


