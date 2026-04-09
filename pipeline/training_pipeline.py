
# logger and exception import
from logger.logger import logging
from exception.exception import ProjectError

# artifacts and config imports
from entity.config_entity import (TrainingPipelineConfig, IngestionConfig, ValidationConfig, PreprocessingConfig, TransformationConfig, ModelTrainerConfig)
from entity.artifact_entity import (IngestionArtifact, PreprocessingArtifact, ValidationArtifact, TransformationArtifact, ModelTrainerArtifact)

# class imports
from components.data_ingestion import IngestionComponent
from components.data_preprocessing import PreProcessingComponent
from components.data_validation import ValidationComponent
from components.data_transformation import TransformationComponent
from components.model_trainer import ModelTrainer 

# library import 
import os
import sys

class TrainingPipeline():

    """
    
    """

    def __init__(self):

        try: 
            self.config: TrainingPipelineConfig = TrainingPipelineConfig()

        except Exception as e:
            raise ProjectError(e, sys)
        
    def start_ingestion(self) -> IngestionArtifact:

        try:
            self.ingestion_config: IngestionConfig = IngestionConfig(self.config)
            logging.info("----- Starting Data Ingestion -----")

            ingestion_component = IngestionComponent(self.ingestion_config)
            ingestion_artifact = ingestion_component.initiate_ingestion()

            logging.info("Data Ingestion Complete")
            return ingestion_artifact
        
        except Exception as e:
            raise ProjectError(e, sys)
    
    def start_preprocessing(self, ingestion_artifact: IngestionArtifact) -> PreprocessingArtifact:

        try:
            self.preprocessing_config = PreprocessingConfig(self.config)
            logging.info("----- Starting Data Preprocessing -----")

            preprocessing_component = PreProcessingComponent(ingestion_artifact, self.preprocessing_config)
            preprocessing_artifact = preprocessing_component.initiate_preprocessing()

            logging.info("Data Pre-Processing Complete")
            return preprocessing_artifact
        except Exception as e:
            raise ProjectError(e, sys)
        
    def start_validation(self, preprocessing_artifact: PreprocessingArtifact) -> ValidationArtifact:

        try:
            self.validation_config = ValidationConfig(self.config)
            logging.info("----- Starting Data Validation -----")

            validation_component = ValidationComponent(preprocessing_artifact, self.validation_config)
            validation_artifact = validation_component.initiate_data_validation()

            logging.info("Data Validation Complete")
            return validation_artifact
        except Exception as e:
            raise ProjectError(e, sys)
        
    def start_transformation(self, validation_artifact: ValidationArtifact) -> TransformationArtifact:

        try:
            self.transformation_config = TransformationConfig(self.config)
            logging.info("----- Starting Data Transformation -----")

            transformation_component = TransformationComponent(validation_artifact, self.transformation_config)
            transformation_artifact = transformation_component.initiate_data_transformation()

            logging.info("Data Transformation Complete")
            return transformation_artifact
        except Exception as e:
            raise ProjectError(e, sys)
        
    def start_training(self, transformation_artifact: TransformationArtifact) -> ModelTrainerArtifact:

        try:
            self.training_config = ModelTrainerConfig(self.config)
            logging.info("----- Starting Model Training -----")

            training_component = ModelTrainer(transformation_artifact, self.training_config)
            training_artifact =  training_component.initiate_model_training()

            logging.info("Model Training Complete")
            return training_artifact
        except Exception as e:
            raise ProjectError(e, sys)
        
    def initiate_pipeline(self):

        try:

            ingestion_artifact = self.start_ingestion()
            preprocessing_artifact = self.start_preprocessing(ingestion_artifact)
            validation_artifact = self.start_validation(preprocessing_artifact)
            transformation_artifact = self.start_transformation(validation_artifact)
            training_artifact = self.start_training(transformation_artifact)

            return training_artifact

        except Exception as e:
            raise ProjectError(e, sys)
       