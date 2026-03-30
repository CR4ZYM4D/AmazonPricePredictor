import numpy as np


""" Common constants to be used throughout the project"""

# split ratio
SPLIT_RATIO = 0.2

# target column
TARGET_COLUMN = 'price'

# artifact directory name
ARTIFACT_DIR = 'artifacts'

# feature store directory name
FEATURE_STORE = 'features'

# base dataset directory
BASE_DATASET_DIR = 'base_dataset'

# base dataset file
BASE_DATASET = 'train.csv'

# collection and DB name
COLLECTION_NAME = 'PricePredictor'
DB_NAME = 'AmazonPricePredictor'

# train split file name
TRAIN_FILE_NAME = 'train.csv'
# test split file name
TEST_FILE_NAME = 'test.csv'

"""Constants related to data ingestion"""

INGESTION_DIR_NAME = 'data_ingestion'
PIPELINE_NAME = 'PricePrediction'
INGESTED_DIR_NAME = 'ingested_data.csv'

"""Constants related to data pre-processing"""

# pre-processing directory name
PREPROCESSING_DIRECTORY = 'preprocessing'
# pre-processed data file name
PREPROCESSED_FILE_NAME = 'preprocessed_data.csv'

"""Constants related to data validation"""

# data validation directory
VALIDATION_DIR_NAME = "data_validation"
# valid data directory
VALID_DATA_DIR_NAME = "valid_data"
# invalid data directory
INVALID_DATA_DIR_NAME = "invalid_data"
# drift report directory 
DRIFT_REPORT_DIR_NAME = "drift_reports"
# drift report file
DRIFT_REPORT_FILE_NAME = "drift_report.yaml"
# schema directory
SCHEMA_DIR_PATH = "schema"
# schema file name
SCHEMA_FILE_PATH = "column_schema.yaml"


"""Constants related to data transformation"""

# transformation directory
TRANSFORMATION_DIR = "data_transformation"
# transformed data directory
TRANSFORMED_DATA_DIR = "transformed_data"
# transformation object directory
TRANSFORMATION_OBJECT_DIR = "transformation_object"
# transformation object file
TRANSFORMATION_OBJECT_FILE = "transformation_object"
# transformed train file
TRANSFORMED_TRAIN_FILE = "train.csv"
# transformed test file
TRANSFORMED_TEST_FILE = "test.csv"
# KNN imputer parameters
KNN_IMPUTER_PARAMS = {
                    
                    "missing_values": np.nan,
                    "n_neighbours": 5,
                    "weights": "distance"

                 }

SIMPLE_IMPUTER_PARAMS = {
    
                        "strategy": "constant",
                        "fill_value": "missing"

                     }

TARGET_COLUMN = 'price'

"""Constants related to model training"""

# model directory name
MODEL_DIR_NAME = "models"
# trained model directory
TRAINED_MODELS = "trained_models"
# failed model directory
FAILED_MODELS = 'poor_fit_models'
# trained model name
MODEL_NAME = "pricePredictorModel"
# expected metrics
ACCURACY_BASE = 0.60
OVERFIT_UNDERFIT_THRESHOLD = 0.05