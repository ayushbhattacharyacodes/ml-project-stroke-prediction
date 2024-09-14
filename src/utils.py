import os, sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException

def evaluate_models(x_train,x_test,y_train,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_accuracy =  accuracy_score(y_train,y_train_pred)
            cm_train = confusion_matrix(y_train,y_train_pred)


            test_model_accuracy =  accuracy_score(y_test,y_test_pred)
            cm_test = confusion_matrix(y_test,y_test_pred)

            train_model_score = train_model_accuracy,cm_train
            test_model_score = test_model_accuracy,cm_test
            report[list(models.keys())[i]] = train_model_score,test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
        
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as f:
            pickle.dump(obj, f)
            
    except Exception as e:
        raise CustomException(e,sys)            
    
def load_object(file_path):
    try:        
        with open(file_path,"rb") as f:
            return pickle.load(f)
            
    except Exception as e:
        raise CustomException(e,sys)    