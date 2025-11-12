from dataclasses import dataclass

@dataclass
class IngestionArtifact:
    """
        Dataclass for the ingestion artifact containing the train and test file paths    
    """
    train_file_path: str
    test_file_path: str

@dataclass
class ValidationArtifact:
    """
        Dataclass for the validation artifact containing the valid/invalid train and test file paths.
        Validation status and the drift report path    
    """
    
    validation_status: bool
    valid_train_file_path: str
    invalid_train_file_path: str
    valid_test_file_path: str
    invalid_test_file_path: str
    drift_report_path:str
