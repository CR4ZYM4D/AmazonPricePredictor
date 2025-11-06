
import os 
import sys
import numpy as np
import pandas as pd

"""constants related to data ingestion"""

COLLECTION_NAME = 'PricePredictor'
DB_NAME = 'AmazonPricePredictor'
INGESTION_DIR_NAME = 'data_ingestion'
PIPELINE_NAME = 'PricePrediction'
INGESTED_DIR_NAME = 'ingested_data'
FEATURE_STORE = 'features'


""" Common constants to be used throughout the project"""

SPLIT_RATIO = 0.2

TARGET_COLUMN = 'price'
ARTIFACT_DIR = 'artifacts'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
