import os,sys
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        logging.info('Initiating Model training')
        try:
          logging.info('Splitting training and testing dataset')
          
          xtrain,ytrain,xtest,ytest=(
              train_array[:,:-1],
              train_array[:,-1],
              test_array[:,:-1],
              test_array[:,-1]
          )   

          models = {
              'Logistic Regression':LogisticRegression(),
              'Decision Tree Classifier':DecisionTreeClassifier(),
              'Random Forest Classifier':RandomForestClassifier()
          } 
          
          params={
              'Logistic Regression':{},
              'Decision Tree Classifier':{
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 2, 4, 6],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4]
                 },
                 'Random Forest Classifier':{
                         'n_estimators': [100, 200],
                         'max_depth': [None,5, 10],
                         'min_samples_split': [2, 5, 10],
                         'min_samples_leaf': [1, 2],
                        }
          }
      
          model_report:dict = evaluate_models(x_train=xtrain,y_train=ytrain,x_test=xtest,y_test=ytest,
                                             models=models,param=params)
          
          best_model_score = max(sorted(model_report.values()))
          best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
          best_model = models[best_model_name]

          train_model_score,test_model_score=best_model_score
          train_acc_score,cm_train = train_model_score
          test_acc_score,cm_test = test_model_score
                    
          predicted_train=best_model.predict(xtrain)
          class_report_train = classification_report(ytrain, predicted_train)
          
          predicted_test=best_model.predict(xtest)
          class_report_test = classification_report(ytest, predicted_test)

          if train_acc_score<0.7:
                raise CustomException("No best model found")
          logging.info(f"Best found model on both training and testing dataset {best_model_name}")
          
          save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
           
          return (
              {
                 'Best Model': best_model_name,
                 'Accuracy Score Train':train_acc_score,
                 'Confusion Matrix Train':cm_train,
                 'Classification Report Train':class_report_train,
                 'Accuracy Score Test': test_acc_score,
                 'Confusion Matrix Test':cm_test,
                 'Classification Report Test':class_report_test
               }
            )
        
        except Exception as e:
            raise CustomException(e,sys) 