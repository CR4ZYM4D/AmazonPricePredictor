
import os
from datetime import datetime
import constants.training_pipeline as train_p

"""

    Class for the Training Pipeline object

"""

class TrainingPipelineConfig():

    def __init__(self, timestamp = datetime.now()):
        
        self.training_pipeline = train_p.PIPELINE_NAME
        self.artifact_name = train_p.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp = timestamp
        return


"""

    Class for the data ingestion config object

"""

class IngestionConfig():

    def __init__(self, training_pipeline_object: TrainingPipelineConfig):

        self.tpo = training_pipeline_object

        self.ingestion_dir = os.path.join(self.tpo.artifact_dir, train_p.INGESTION_DIR_NAME)
        
        self.training_file_path = os.path.join(self.ingestion_dir, train_p.TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.ingestion_dir, train_p.TEST_FILE_NAME)

        self.split_ratio = train_p.SPLIT_RATIO
        self.collection_name = train_p.COLLECTION_NAME
        self.db_name = train_p.DB_NAME
        return
        
        