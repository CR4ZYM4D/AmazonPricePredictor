from logger.logger import logging
from exception.exception import ProjectError

import subprocess
import sys

class S3Sync():

    def __init__():

        try: 
            pass 

        except Exception as e:
            raise ProjectError(e, sys)
        
    def sync_folder_to_s3(self, folder: str, url: str):

        """
        
        """

        try: 
            result = subprocess.run(["aws", "s3", "sync", folder, url], check = True, capture_output=True, text = True)

            logging.info(f"AWS syncing of {folder} to {url} finished with result: {result}")

        except Exception as e:
            raise ProjectError(e, sys)

    def sync_folder_from_bucket(self, folder: str, url: str):

        """
        
        """

        try: 
            result = subprocess.run(["aws", "s3", "sync", url, folder], check = True, capture_output=True, text = True)

            logging.info(f"AWS syncing of {folder} from {url} finished with result: {result}")

        except Exception as e:
            raise ProjectError(e, sys)