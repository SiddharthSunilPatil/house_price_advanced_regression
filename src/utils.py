import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import make_scorer,r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(models,X_train,y_train):
    model_score_r2=[]
    model_score_rmse=[]

    def test_model_r2(model,X_train=X_train,y_train=y_train):
        cv=KFold(n_splits=7,shuffle=True,random_state=45)
        r2=make_scorer(r2_score)
        r2_val_score=cross_val_score(model,X_train,y_train,cv=cv,scoring=r2)
        score=r2_val_score.mean()
        return score

    def test_model_rmse(model,X_train=X_train,y_train=y_train):
        cv=KFold(n_splits=7,shuffle=True,random_state=45)
        mse=make_scorer(mean_squared_error)
        mse_val_score=cross_val_score(model,X_train,y_train,cv=cv,scoring=mse)
        score=np.sqrt(mse_val_score.mean())
        return score

    for i in range(len(models)):
        print("Model Name:",list(models.keys())[i])
        print("-"*50)
        score_r2=test_model_r2(list(models.values())[i],X_train,y_train)
        score_rmse=test_model_rmse(list(models.values())[i],X_train,y_train)
        print("r2_Score of the model:",score_r2)
        print("rmse of the model:",score_rmse)
        print("="*50)
        model_score_r2.append(score_r2)
        model_score_rmse.append(score_rmse)

    model_performance=pd.DataFrame(zip(models,model_score_r2,model_score_rmse),columns=["Model Name","R2_Score","RMSE"]).sort_values(by=["RMSE"])

    return(
        model_performance
    )
        

    


    