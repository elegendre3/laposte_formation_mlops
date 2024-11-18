# Databricks notebook source
import tensorflow as tf
from tensorflow import keras as keras
import tensorboard

print(tf.__version__)
print(tensorboard.__version__)

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = "/Workspace/Users/eliott.legendre@openvalue.fr/runs/"

# COMMAND ----------

#  check existing tensorboard instances
from tensorboard import notebook
notebook.list()

# %sh kill -9 19269

# COMMAND ----------

# MAGIC %tensorboard --logdir $experiment_log_dir --port=6105 --bind_all

# COMMAND ----------

# MAGIC %md
# MAGIC ## get_data and get_model helpers

# COMMAND ----------

def get_dataset(num_classes, rank=0, size=1):
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)


def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers
  
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model


def get_model_dumb(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers
  
  model = models.Sequential()
  model.add(layers.Flatten(input_shape=(28, 28, 1)))
  model.add(layers.Dense(784, activation='relu'))
  model.add(layers.Dense(30, activation='relu'))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define callbacks for tracking

# COMMAND ----------

# tensorboard callback
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=experiment_log_dir, histogram_freq=1)

# mlflow callback
class MlflowLoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric("epoch_loss", logs["loss"], step=epoch)
        mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)
        mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)

mlflow_logging_callback = MlflowLoggingCallback()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, log w. mlflow, register model

# COMMAND ----------

import mlflow
from sklearn.metrics import f1_score
import numpy as np

batch_size = 128
epochs = 50
num_classes = 10
learning_rate = 0.1

# These steps are automatically skipped on a CPU cluster
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

experiment_id = '3040381985975142'
with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("learning_rate", learning_rate)

        (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
        
        # permute y_train to tank learning
        y_train_randomized = np.random.permutation(y_train)
        
        model = get_model(num_classes)
        print(model.summary())

        # Specify the optimizer -- use Adadelta so that Horovod can adjust the learning rate during training
        optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        # model.fit(x_train, y_train_randomized,
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(x_test, y_test),
                callbacks=[tensorboard_callback, mlflow_logging_callback]
        )
        
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
        
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        f1 = f1_score(y_true, y_pred_classes, average='weighted')
        
        print("loss:", loss)
        print("accuracy:", accuracy)
        print("f1 score:", f1)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # log model as an sklearn model - provide sample to auto-set the model signature
        mlflow.keras.log_model(model, "model")

        # register model
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = "training_mnist_TFKeras"
        model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Check existing model versions and transition to STG if accuracy sufficient

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "training_mnist_TFKeras"

# Retrieve all versions of the model
model_versions = client.search_model_versions(f"name='{model_name}'")

# Get accuracy and deploy the model if val_accuracy is above 0.8
for version in model_versions:
    run_id = version.run_id
    metrics = client.get_run(run_id).data.metrics
    val_accuracy = metrics.get("test_accuracy", 0)
    print(f'[{run_id}] has val acc [{val_accuracy}]')
    if val_accuracy > 0.8:
        print('Transitionning to Staging')
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging"
        )
        print(f"Model version {version.version} with val_accuracy {val_accuracy} deployed to STG.")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize MNIST data w. matplotlib

# COMMAND ----------


import matplotlib.pyplot as plt
import numpy as np

y_test_randomized = np.random.permutation(y_test)

# Visualize the first 10 images in the test set
for i in range(2):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(y_test[i])}")
    plt.show()
    print(f"Label [{y_test[i]}]")
    print(f"Label Shuffled [{y_test_randomized[i]}]")



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Call deployed model

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json


def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-34351534749836.16.azuredatabricks.net/serving-endpoints/mnist_2/invocations'
    headers = {'Authorization': f"Bearer {dbutils.secrets.get(scope='formation-docaposte', key='el_db_api_key')}", 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


(_, _), (x_test, y_test) = get_dataset(10)
# score_model(x_test[:2000])
score_model(x_test[:100])

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


