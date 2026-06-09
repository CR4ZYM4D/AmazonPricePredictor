from components.data_preprocessing import PreProcessingComponent
from components.data_transformation import TransformationComponent
from components.data_validation import ValidationComponent
from components.model_trainer import ModelTrainer
from components.data_ingestion import IngestionArtifact

if __name__ == '__main__':

    artifact = IngestionArtifact('./base_dataset/train.csv')
    
    preprocessing = PreProcessingComponent(artifact)
    pre_arti = preprocessing.initiate_preprocessing()

    valid = ValidationComponent(preprocessing)
    valid_arti = valid.initiate_data_validation()

    trans = TransformationComponent(validation_artifact=valid_arti)
    trans_arti = trans.initiate_data_transformation()

    trainer = ModelTrainer(transformation_artifact=trans_arti)
    model = trainer.initiate_model_training()