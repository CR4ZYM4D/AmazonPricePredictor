
import os
from datetime import datetime
import constants.training_pipeline as train_p

class TrainingPipelineConfig():

    """
        Class for the Training Pipeline object
    """

    def __init__(self, timestamp = datetime.now()):
        
        self.training_pipeline = train_p.PIPELINE_NAME
        self.artifact_name = train_p.ARTIFACT_DIR
        self.feature_store = train_p.FEATURE_STORE
        self.artifact_dir = os.path.join(self.artifact_name, f"_{timestamp}")
        self.timestamp = timestamp
        return

class IngestionConfig():

    """
        Class for the data ingestion config object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # DB and collection name
        self.db_name = train_p.DB_NAME
        self.collection_name = train_p.COLLECTION_NAME

        # data ingestion and ingested data directory paths
        self.ingestion_dir = os.path.join(self.tpo.artifact_dir, train_p.INGESTION_DIR_NAME)
        self.ingested_dir = os.path.join(self.ingestion_dir, train_p.INGESTED_DIR_NAME)

        # feature store directory and train/test file paths
        self.feature_store = os.path.join(self.ingestion_dir, train_p.FEATURE_STORE)
        self.training_file_path = os.path.join(self.feature_store, train_p.TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.feature_store, train_p.TEST_FILE_NAME)

        # train/test split ratio
        self.split_ratio = train_p.SPLIT_RATIO
        return

class ValidationConfig():

    """
        Class for the Data Validation Config object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # validation and valid/invalid data directory paths
        self.validation_dir = os.path.join(self.tpo.artifact_dir, train_p.VALIDATION_DIR_NAME)
        self.valid_data_dir = os.path.join(self.validation_dir, train_p.VALID_DATA_DIR_NAME)
        self.invalid_data_dir = os.path.join(self.validation_dir, train_p.INVALID_DATA_DIR_NAME)

        # drift report directory and file paths         
        self.drift_report_dir = os.path.join(self.validation_dir, train_p.DRIFT_REPORT_DIR_NAME)
        self.drift_report_file = os.path.join(self.drift_report_dir, train_p.DRIFT_REPORT_FILE_NAME)
        
        # schema file path
        self.schema_file_path = os.path.join(os.getcwd(), train_p.SCHEMA_DIR_PATH)
        return

        