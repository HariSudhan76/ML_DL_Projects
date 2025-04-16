import os
import sys 
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)  

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in models:
            m = models[i]
            params = param[i]
            gs = GridSearchCV(m,params,cv=3)
            gs.fit(X_train,y_train) #Train Model

            m.set_params(**gs.best_params_)
            m.fit(X_train,y_train)

            y_train_pred = m.predict(X_train)
            y_test_pred = m.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[i] = test_model_score 
        
        return report 
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)