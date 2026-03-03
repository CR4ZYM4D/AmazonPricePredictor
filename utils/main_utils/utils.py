# logging and error import
from logger.logger import logging
from exception.exception import ProjectError

# library import
import os
from pathlib import Path
import sys
import yaml

def read_yaml(file_path: Path | str) -> dict: 

    """
        Reads The YAML file passed in the specified path and returns the stream object\n 
        params -> \n
        ***file_path*** : Path | str containing the path of the yaml file that needs to be read \n
        returns ->\n YAML file stream object
    """

    try:
        
        logging.info(f"Reading {file_path} file")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding = 'utf-8') as f:

                schema = yaml.safe_load(f)

                if schema is None:
                    logging.warning("Schema is empty! no data has been read")
                    raise ProjectError("Schema is empty! no data has been read", sys)

                logging.info(f"Successfully read the yaml file")
                return schema
        else: 
            raise ProjectError("The specified YAML File Path does not exist! Please ensure the path is correct and try again", sys)

    except Exception as e:
        raise ProjectError(e, sys)
    
def write_yaml(data: dict, file_path: Path | str, overwrite: bool = False):

    """
        Writes a data stream into the specified file path as a yaml file. If the path already exists, it throws an error and stops execution by default.\n
        Can be set to overwrite the pre-existing file contents as well.\n
        params ->\n
        ***data***: The data to be written into the yaml file. \n
        ***file_path***: Path | str The path of the yaml file the data is to to be written in. \n
        ***overwrite***: bool *default* = False Whether to overwrite the file if it already exists. False by default. \n
        returns -> \n
        None
    """

    try: 
        
        if os.path.exists(file_path) and overwrite == False:

            logging.info("The specified file path exists and overwrite was set to False. Exitting loop")
            raise(ProjectError("The specified file already exists!"))
        
        logging.info(f"Creating file {file_path}")
        os.makedirs(file_path, exist_ok = True)
        
        with open(file_path, 'w', encoding = 'utf-8') as f:

            yaml.dump(data, f, default_flow_style= False)
            logging.info(f"Successfully written data into file {file_path}")
        return

    except Exception as e:
        raise ProjectError(e, sys)