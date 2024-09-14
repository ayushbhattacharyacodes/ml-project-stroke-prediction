import os,sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_cleaning import DataCleaning

from src.logger import logging
from src.exceptions import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path:str =  os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
   
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            data = pd.read_csv('./notebook/data/data.csv')
            logging.info('Read the dataset as a dataframe')
            
            logging.info('Transferring the dataframe into the folder')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Complethe data transfer')

            return self.ingestion_config.raw_data_path
        
        except Exception as e:
            raise CustomException(e,sys)
        


        


        