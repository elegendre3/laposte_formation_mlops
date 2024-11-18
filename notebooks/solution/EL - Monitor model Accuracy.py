# Databricks notebook source
import json
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "training_mnist_TFKeras"

# Retrieve all versions of the model
model_versions = client.search_model_versions(f"name='{model_name}'")

best_model = {
  "name": model_name,
  "val_accuracy": -999.0,
  "run_id": None,
  "version": None,
}

for version in model_versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics
    val_accuracy = metrics.get("test_accuracy", 0)
    print(f'[{run_id}] - v[{version.version}] has val acc [{val_accuracy}]')
    if val_accuracy > best_model["val_accuracy"]:
      best_model["val_accuracy"] = val_accuracy
      best_model["run_id"] = run_id
      best_model["version"] = version.version
      best_model["current_stage"] = version.current_stage

print()
print(f"Best accuracy [{best_model['val_accuracy']}]")
print(json.dumps(best_model, indent=4))
    

# COMMAND ----------

# from mlflow.tracking import MlflowClient

# client = MlflowClient()
# model_name = "training_mnist_TFKeras"

# # Retrieve all versions of the model
# model_versions = client.search_model_versions(f"name='{model_name}'")

# # Get accuracy and deploy the model if val_accuracy is above 0.8
# for version in model_versions:
#     run_id = version.run_id
#     metrics = client.get_run(run_id).data.metrics
#     val_accuracy = metrics.get("test_accuracy", 0)
#     print(f'[{run_id}] has val acc [{val_accuracy}]')
#     if val_accuracy > 0.8:
#         print('Transitionning to Staging')
#         client.transition_model_version_stage(
#             name=model_name,
#             version=version.version,
#             stage="Staging"
#         )
#         print(f"Model version {version.version} with val_accuracy {val_accuracy} deployed to STG.")
