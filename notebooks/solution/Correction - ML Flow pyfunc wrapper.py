# Databricks notebook source
import mlflow.pyfunc
import mlflow
from databricks.feature_store import FeatureStoreClient
import tensorflow as tf
import numpy as np
import databricks
from tensorflow import keras

# COMMAND ----------

class MyModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = mlflow.pyfunc.load_model(context.artifacts["model"])

    def preprocessing(self, input_):
        return input_
    
    def postprocessing(self, output_):
        return output_

    def predict(self, context, model_input):
        processed_input = self.preprocessing(model_input)
        prediction = self.model.predict(processed_input)
        processed_output = self.postprocessing(prediction)
        
        return processed_output


# COMMAND ----------



# Load the original model
run_id = "b9fe618046594f0fbfba4e20329d5ba6"
model_uri = f"runs:/{run_id}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Define the artifacts
artifacts = {
    "model": model_uri
}

class contextObject():
  def __init__(self):
    self.artifacts = {'model': model_uri}

context = contextObject()


#model_test.predict(context=context, model_input=model_input)
pip_requirements=["pandas", 
                  "mlflow", 
                  "databricks-feature-store", 
                  "numpy",
                  "tensorflow"]

artifact_path = "Loans_prediction_wrapper"



# COMMAND ----------

model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

# COMMAND ----------

# Save the wrapped model
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=MyModelWrapper(),
        artifacts=artifacts,
        pip_requirements = pip_requirements
    )

# COMMAND ----------

run.info.run_id
