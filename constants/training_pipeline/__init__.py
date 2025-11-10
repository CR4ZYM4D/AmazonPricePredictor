
import os 
import sys
import numpy as np
import pandas as pd

""" Common constants to be used throughout the project"""

SPLIT_RATIO = 0.2

TARGET_COLUMN = 'price'
ARTIFACT_DIR = 'artifacts'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

"""Constants related to data ingestion"""

COLLECTION_NAME = 'PricePredictor'
DB_NAME = 'AmazonPricePredictor'
INGESTION_DIR_NAME = 'data_ingestion'
PIPELINE_NAME = 'PricePrediction'
INGESTED_DIR_NAME = 'ingested_data'
FEATURE_STORE = 'features'

"""Constants related to data validation"""

VALIDATION_DIR_NAME = "data_validation"
VALID_DATA_DIR_NAME = "valid_data"
INVALID_DATA_DIR_NAME = "invalid_data"
DRIFT_REPORT_DIR_NAME = "drift_reports"
DRIFT_REPORT_FILE_NAME = "drift_report"
SCHEMA_DIR_PATH = "schema"
SCHEMA_FILE_PATH = "schema.yaml"

