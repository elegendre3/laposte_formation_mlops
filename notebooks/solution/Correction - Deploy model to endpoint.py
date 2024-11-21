# Databricks notebook source
import mlflow.pyfunc
import mlflow
import tensorflow as tf
import numpy as np


# COMMAND ----------

import requests
import json
import mlflow

run_id = "5fbba7c840ab42ef9393e05ec893997e"

# Define variables
endpoint_name = "pyfunc_wrapper"
model_uri = f"runs:/{run_id}/Loans_prediction_wrapper"
token = dbutils.secrets.get(scope="formation-docaposte", key="cnaamane_access_token")

# Create or update the registered model
client = mlflow.MlflowClient()
try:
    client.create_registered_model(endpoint_name)
except mlflow.exceptions.MlflowException:
    print(f"Model {endpoint_name} already exists.")

# Register the model
model_version = mlflow.register_model(model_uri, endpoint_name)

# Transition the model to production
client.transition_model_version_stage(
    name=endpoint_name,
    version=model_version.version,
    stage="Production"
)

# Get the Databricks instance URL from the workspace configuration
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Create or update a serving endpoint using Databricks REST API
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

endpoint_url = f"{workspace_url}/api/2.0/serving-endpoints"
data = {
    "name": endpoint_name,
    "config": {
            "served_entities": [
                {
                    "entity_name": endpoint_name,
                    "entity_version": model_version.version,
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True
              }
                ]
    }
}

response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))

# Check the response
if response.status_code == 200:
    print("Endpoint created successfully.")
elif response.status_code == 400:  # Endpoint already exists
    update_url = f"{endpoint_url}/{endpoint_name}/config"
    update_data = data['config']

    update_response = requests.put(update_url, headers=headers, data=json.dumps(update_data))
    if update_response.status_code == 200:
        print("Endpoint updated successfully.")
    else:
        raise ValueError(f"Failed to update endpoint: {update_response.text}")
else:
    print(f"Failed to create endpoint: {response.text}")
