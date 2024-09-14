import sys
import numpy as np

from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 

class TrainPipeline:
    def __init__(self):
        pass
    def invoke_functions(self):
       try: 
            print('Data Ingestion Starts')
            obj = DataIngestion()
            data = obj.initiate_data_ingestion()
            print('End of Data Ingestion')

            print('Data Cleaning Begins')
            data_cleaner = DataCleaning()
            train_data,test_data = data_cleaner.initiate_data_cleaning(data)
            print('End of Data Cleaning')

            print('Data Transformation Begins')
            data_transformation=DataTransformation()
            train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
            print('End of Data Transformation')

            print('Model Training Begins')
            modeltrainer=ModelTrainer()
            metric_dict=modeltrainer.initiate_model_trainer(train_arr,test_arr)       
            print('End of Model Training')

            return metric_dict
       except Exception as e:
            raise CustomException(e,sys) 
        
        
    def display_result(self):
        try:    
            metric_dict = self.invoke_functions()    
            print(f'''
                    Conclusion on Training and Validating the Data
                    -----------------------------------------------

                    Best Model name is {metric_dict['Best Model']}
                    For Training Data:
                    Accuracy Score is {np.round(metric_dict['Accuracy Score Train'],3)*100}%
                    and.
                    Confusion matrix for testing data is
                    {metric_dict['Confusion Matrix Train']} 
                    and finally the Classification Report is
                    {metric_dict['Classification Report Train']}

                    For Testing Data:
                    Accuracy Score is 
                    {np.round(metric_dict['Accuracy Score Test'],3)*100}%
                    and.
                    Confusion matrix for testing data is
                    {metric_dict['Confusion Matrix Test']} 
                    and finally the Classification Report is
                    {metric_dict['Classification Report Test']}
                ''')
        except Exception as r:
             raise CustomException(e,sys)        
        
if __name__=='__main__':
       train_pipeline = TrainPipeline()
       train_pipeline.display_result()