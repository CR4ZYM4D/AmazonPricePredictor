# pipeline/prediction_pipeline.py

from logger.logger import logging
from exception.exception import ProjectError

from entity.config_entity import TrainingPipelineConfig, IngestionConfig, PreprocessingConfig
from entity.artifact_entity import IngestionArtifact, PreprocessingArtifact

from components.data_ingestion import IngestionComponent
from components.data_preprocessing import PreProcessingComponent

import pandas as pd
import sys

class PredictionPipeline():

    """
    Prediction Pipeline class.
    Runs Preprocessing on data ingested from MongoDB
    and returns a processed DataFrame for model inference.
    Same as TrainingPipeline structure but skips Validation, Transformation and Training.
    """

    def __init__(self):
        try:
            self.config = TrainingPipelineConfig()
        except Exception as e:
            raise ProjectError(e, sys)

    def start_ingestion(self) -> IngestionArtifact:
        try:
            ingestion_config    = IngestionConfig(self.config)
            logging.info("----- Prediction Pipeline: Starting Data Ingestion -----")

            ingestion_component = IngestionComponent(ingestion_config)
            ingestion_artifact  = ingestion_component.initiate_ingestion()

            logging.info("Prediction Pipeline: Data Ingestion Complete")
            return ingestion_artifact

        except Exception as e:
            raise ProjectError(e, sys)

    def start_preprocessing(self, ingestion_artifact: IngestionArtifact) -> PreprocessingArtifact:
        try:
            preprocessing_config    = PreprocessingConfig(self.config)
            logging.info("----- Prediction Pipeline: Starting Data Preprocessing -----")

            preprocessing_component = PreProcessingComponent(ingestion_artifact, preprocessing_config)
            preprocessing_artifact  = preprocessing_component.initiate_preprocessing()

            logging.info("Prediction Pipeline: Data Preprocessing Complete")
            return preprocessing_artifact

        except Exception as e:
            raise ProjectError(e, sys)

    def get_processed_dataframe(self) -> pd.DataFrame:
        """
        Runs ingestion + preprocessing and returns the processed DataFrame.
        Called by the /predict route in app.py.
        """
        try:
            ingestion_artifact    = self.start_ingestion()
            preprocessing_artifact = self.start_preprocessing(ingestion_artifact)

            df = pd.read_csv(preprocessing_artifact.preprocessed_file_path)
            logging.info(f"Prediction Pipeline: Loaded processed DataFrame with shape {df.shape}")
            return df

        except Exception as e:
            raise ProjectError(e, sys)