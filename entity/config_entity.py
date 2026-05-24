
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

        return
    
class PreprocessingConfig():

    """
        Class for the data pre-processing object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # feature store directory and train/test file paths
        self.feature_store = os.path.join(self.tpo.artifact_dir, train_p.FEATURE_STORE)
        self.preprocessing_directory = os.path.join(self.feature_store, train_p.PREPROCESSING_DIRECTORY)
        self.preprocessed_file_path = os.path.join(self.preprocessing_directory, train_p.PREPROCESSED_FILE_NAME)

        return

class ValidationConfig():

    """
        Class for the Data Validation Config object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # base dataset path
        self.base_dataset_path = os.path.join(train_p.BASE_DATASET_DIR, train_p.BASE_DATASET)

        # validation and valid/invalid data directory paths
        self.validation_dir = os.path.join(self.tpo.artifact_dir, train_p.VALIDATION_DIR_NAME)
        self.valid_data_dir = os.path.join(self.validation_dir, train_p.VALID_DATA_DIR_NAME)
        self.invalid_data_dir = os.path.join(self.validation_dir, train_p.INVALID_DATA_DIR_NAME)

        self.train_file_path = os.path.join(self.valid_data_dir, train_p.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.valid_data_dir, train_p.TEST_FILE_NAME)

        self.split_ratio = train_p.SPLIT_RATIO

        # drift report directory and file paths         
        self.drift_report_dir = os.path.join(self.validation_dir, train_p.REPORT_DIR_NAME)
        self.drift_report_file = os.path.join(self.drift_report_dir, train_p.DRIFT_REPORT_FILE_NAME)
        
        # schema file path
        self.schema_dir_path = os.path.join(os.getcwd(), train_p.SCHEMA_DIR_PATH)
        self.schema_file_path = os.path.join(self.schema_dir_path, train_p.SCHEMA_FILE_PATH)

        return
    
class TransformationConfig():

    """
        Class for the Data Transformation Config object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # data transformation directory path
        self.transformation_dir = os.path.join(self.tpo.artifact_dir, train_p.TRANSFORMATION_DIR)

        # transformed data directory path
        self.transformed_data_dir = os.path.join(self.transformation_dir, train_p.TRANSFORMED_DATA_DIR)

        # transformed train/test directory
        self.transformed_train_directory = os.path.join(self.transformed_data_dir, "train")        
        self.transformed_test_directory = os.path.join(self.transformed_data_dir, "test")

        self.transformed_train_file = os.path.join(self.transformed_train_directory, train_p.TRANSFORMED_TRAIN_FILE)
        self.transformed_test_file = os.path.join(self.transformed_test_directory, train_p.TRANSFORMED_TEST_FILE)

        # transformation object directory path
        self.transformation_object_dir = os.path.join(self.transformation_dir, train_p.TRANSFORMATION_OBJECT_DIR)
        # transformation object file path
        self.transformation_object_path = os.path.join(self.transformation_object_dir, train_p.TRANSFORMATION_OBJECT_FILE)

class ModelTrainerConfig():

    """
        Class for the model trainer config object
    """

    def __init__(self, training_pipeline_object: TrainingPipelineConfig = TrainingPipelineConfig()):

        self.tpo = training_pipeline_object

        # initialize models directory path
        self.models_dir = os.path.join(self.tpo.artifact_dir, train_p.MODEL_DIR_NAME)
            
        # initialize trained/failed models directory
        self.model_trainer_dir = os.path.join(self.models_dir, train_p.TRAINED_MODELS)

        # initialize trained model file path
        self.trained_model_path = os.path.join(self.model_trainer_dir, train_p.MODEL_NAME)

        self.reports_dir = os.path.join(self.tpo.artifact_dir, train_p.REPORT_DIR_NAME)

        # initialize reports path
        self.reports_path = os.path.join(self.reports_dir, train_p.MODEL_METRICS)

        # intialize metrics
        self.base_accuracy = train_p.ACCURACY_BASE
        self.overfit_underfit_threshold = train_p.OVERFIT_UNDERFIT_THRESHOLD            