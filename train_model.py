import argparse
import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from tensorflow import keras


def load_data_from_bq(bq_uri: str) -> pd.DataFrame:
    """
    Load data from BigQuery
    Inputs:
        bq_uri (str)
    Returns
        df (pd.DataFrame)
    """
    if not bq_uri.startswith("bq://"):
        raise Exception("uri is not a BQ uri. It should be bq://project_id.dataset.table")

    project_id, dataset, table = bq_uri[5:].split(".")

    query_string = """
    SELECT * from `{}.{}.{}`
    """.format(project_id, dataset, table)

    client = bigquery.Client(project=project_id)
    df = client.query(query_string).to_dataframe()
    return df


def make_model(train_df: pd.DataFrame) -> keras.Sequential:
    """
    Make model
    Inputs:
        train_df (pd.DataFrame)
    Returns
        model (keras.engine.sequential.Sequential)
    """
    model = keras.Sequential([
        keras.layers.BatchNormalization(input_shape=(train_df.shape[-1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model


def run(args: dict):

    # get training data
    train_df = load_data_from_bq(bq_uri=args["training_data_uri"]).astype(float)

    # if Vertex AI Pipelines execution
    if os.environ.get("AIP_MODEL_DIR"):
        # get test and validation data
        test_df = load_data_from_bq(bq_uri=args["test_data_uri"]).astype(float)
        val_df = load_data_from_bq(bq_uri=args["validation_data_uri"]).astype(float)

        # Vertex AI Pipelines splits data but does not account for class imbalance
        # Automatic split described below:
        # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob
        # merge train, test, and val datasets due to class imbalance inherent in dataset
        # class imbalance described below:
        # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        train_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    # determine class imbalance
    neg_class_count, pos_class_count = np.bincount(train_df[args["class_column"]])
    total_class_count = neg_class_count + pos_class_count

    # remove unneeded columns
    for remove_column in args["remove_input_columns"]:
        if remove_column in train_df:
            train_df.pop(remove_column)

    # split dataset
    X = train_df.drop(columns=args["class_column"])
    y = train_df[args["class_column"]].values

    train_df, test_df, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    train_df, val_df, y_train, y_val = train_test_split(train_df, y_train, test_size=0.2)

    print("train_df.shape:", train_df.shape)
    print("y_train.shape:", y_train.shape)
    print("val_df.shape:", val_df.shape)
    print("y_val.shape:", y_val.shape)
    print("test_df.shape:", test_df.shape)
    print("y_test.shape:", y_test.shape)

    # Write test dataset to GCS file path for downstream batch prediction and evaluation
    # test dataset loaded at path functions as gcs_source_uri input into ModelBatchPredictOp
    output_test_df = pd.DataFrame()
    output_test_df["input_features"] = test_df.apply(lambda x: x.values, axis=1)
    output_test_df["Class"] = y_test.astype(int).astype(str)
    output_test_df = output_test_df.reset_index().rename(columns={"index": "key"})

    # if Vertex AI Pipelines execution
    if os.environ.get("AIP_MODEL_DIR"):
        output_test_df.to_json(args["test_dataset_file_path"], orient="records", lines=True)

    # build the model
    model = make_model(train_df)

    # calculate class weights
    weight_for_0 = (1 / neg_class_count) * (total_class_count / 2.0)
    weight_for_1 = (1 / pos_class_count) * (total_class_count / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # train model
    model.fit(
        train_df,
        y_train,
        batch_size=args["batch_size"],
        epochs=args["epochs"],
        validation_data=(val_df, y_val),
        class_weight=class_weight
    )

    # print results
    results = model.evaluate(test_df, y_test, batch_size=args["batch_size"], verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(name, ": ", value)

    # if model_dir, export model
    if args["model_dir"]:
        print("model_dir:", args["model_dir"])

        # wrap the model, so we can add a custom serving function
        class ExportModel(keras.Model):
            def __init__(self, model):
                super().__init__(self)
                self.model = model

            # instance key needs to be added to serving signature
            # input data requirements for online and batch prediction described below:
            # https://cloud.google.com/vertex-ai/docs/predictions/get-predictions#input_data_requirements
            @tf.function(input_signature=[tf.TensorSpec([None, train_df.shape[-1]], dtype=tf.float32),
                                          tf.TensorSpec([None], dtype=tf.int32)])
            def custom_serve(self, input_features, key):
                scores = self.model(input_features)
                return {"scores": scores}

        restored_model = make_model(train_df)
        restored_model.set_weights(model.get_weights())

        # export model in the standard SavedModel format
        serving_model = ExportModel(restored_model)
        serving_model.trainable = False
        tf.saved_model.save(serving_model, args["model_dir"], signatures={"serving_default": serving_model.custom_serve})


# Define all the command line arguments your model can accept for training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        help="number of training epochs",
        type=int,
        default=os.environ["EPOCHS"] if "EPOCHS" in os.environ else 100
    )

    parser.add_argument(
        "--batch_size",
        help="training batch size",
        type=int,
        default=os.environ["BATCH_SIZE"] if "BATCH_SIZE" in os.environ else 2048
    )

    parser.add_argument(
        "--model_dir",
        help="directory to output model and artifacts",
        type=str,
        default=os.environ["AIP_MODEL_DIR"] if "AIP_MODEL_DIR" in os.environ else ""
    )

    parser.add_argument(
        "--data_format",
        choices=["csv", "bigquery"],
        help="format of data uri csv for gs:// paths and bigquery for project.dataset.table formats",
        type=str,
        default=os.environ["AIP_DATA_FORMAT"] if "AIP_DATA_FORMAT" in os.environ else "csv"
    )

    parser.add_argument(
        "--training_data_uri",
        help="location of training data in either gs:// uri or bigquery uri; default location for local runs",
        type=str,
        default=os.environ["AIP_TRAINING_DATA_URI"] if "AIP_TRAINING_DATA_URI" in os.environ else ""
    )

    parser.add_argument(
        "--validation_data_uri",
        help="location of validation data in either gs:// uri or bigquery uri",
        type=str,
        default=os.environ["AIP_VALIDATION_DATA_URI"] if "AIP_VALIDATION_DATA_URI" in os.environ else ""
    )

    parser.add_argument(
        "--test_data_uri",
        help="location of test data in either gs:// uri or bigquery uri",
        type=str,
        default=os.environ["AIP_TEST_DATA_URI"] if "AIP_TEST_DATA_URI" in os.environ else ""
    )

    parser.add_argument(
        "--class_column",
        help="class column",
        type=str,
        default=os.environ["CLASS_COLUMN"] if "CLASS_COLUMN" in os.environ else "Class"
    )

    parser.add_argument(
        "--remove_input_columns",
        help="remove input columns",
        nargs="+",
        default=json.loads(os.environ["REMOVE_INPUT_COLUMNS"]) if "REMOVE_INPUT_COLUMNS" in os.environ else ["Row_Weight", "Time"])

    parser.add_argument(
        "--test_dataset_file_path",
        help="test dataset file path",
        type=str,
        default=os.environ["TEST_DATASET_FILE_PATH"] if "TEST_DATASET_FILE_PATH" in os.environ else ""
    )

    args = parser.parse_args()
    args = args.__dict__

    run(args)
