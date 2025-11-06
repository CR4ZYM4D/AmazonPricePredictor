from dataclasses import dataclass

"""
    Dataclass for the ingestion artifact containing the train and test file pathsd    
"""

@dataclass
class IngestionArtifact:
    train_file_path: str
    test_file_path: str