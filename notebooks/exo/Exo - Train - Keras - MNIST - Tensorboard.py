# Databricks notebook source
# MAGIC %md
# MAGIC # Train MNIST model with Keras, monitor w. tensorboard and mlflow
# MAGIC
# MAGIC Set up tensorboard (safari blocks iframe?)
# MAGIC
# MAGIC Load mnist data
# MAGIC Visualize data w. matplotlib
# MAGIC
# MAGIC Create Keras model
# MAGIC
# MAGIC Create tensorboard and mlflow callbacks for monitoring epoch per epoch
# MAGIC
# MAGIC Set up for GPU training
# MAGIC
# MAGIC Use MLFLOW to log params, metrics and model
# MAGIC
# MAGIC Check tensorboard UI
# MAGIC
# MAGIC Deploy model in UI or code
# MAGIC
# MAGIC Call deployed model
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### POST and GET secrets via requests

# COMMAND ----------

# # Add a secret to the scope

# import requests

# db_token = "mytoken"

# # Define variables
# workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
# headers = {
#     "Authorization": f"Bearer {db_token}",
#     "Content-Type": "application/json"
# }
# add_secret_url = f"{workspace_url}/api/2.0/secrets/put"

# data = {
#     "scope": "formation-docaposte",
#     "key": "el_db_api_key",
#     "string_value": f"{db_token}"
# }
# response = requests.post(add_secret_url, headers=headers, json=data)
# if response.status_code == 200:
#     print("Secret added successfully.")
# else:
#     print(f"Failed to add secret: {response.text}")





# retrieve secrets
# scopes = dbutils.secrets.listScopes()
# display(scopes)
# secrets = dbutils.secrets.list("formation-docaposte")
# display(secrets)
# api_key = dbutils.secrets.get(scope="formation-docaposte", key="el_db_api_key")
# display(api_key)

# COMMAND ----------


