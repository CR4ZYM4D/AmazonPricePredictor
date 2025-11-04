import os
from datetime import datetime
import logging

'''
Code to create the logging files named in the format $(month_day_year_hour_month_second.log) 

The log files are stored in the logs/$(month_day_year_hour_month_second) folder

Each log file contains logs in the format:

        YYYY MM DD hh mm ss line_number file_name - level - message

'''

LOG_FILE_NAME = f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}"

LOG_PATH = os.path.join(os.getcwd(), 'logs', LOG_FILE_NAME)

os.makedirs(LOG_PATH, exist_ok= True)

LOG_FILE_PATH = os.path.join(LOG_PATH, f"{LOG_FILE_NAME}.log")

logging.basicConfig(

    filename=LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)