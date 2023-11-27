import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    model_performace_file_path=os.path.join('artifacts','model_performance.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,y_train):
        try:
            logging.info("Entered the model trainer method")
            models={
                "LinearRegression":LinearRegression(),
                "SVR":SVR(),
                "SGDRegressor":SGDRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "GaussianProcessRegressor":GaussianProcessRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "MLPRegressor":MLPRegressor(),
            }

            

            model_performance=evaluate_models(models,X_train,y_train)
            logging.info("Evaluation of models completed")

            model_performance.to_csv(self.model_trainer_config.model_performace_file_path)

            best_model_score=model_performance.iloc[0][2]
            best_model_name=model_performance.iloc[0][0]

            best_model=models[best_model_name]
            logging.info("Best model found")

            print("The best model is:",best_model_name)
            print("RMSE for the model is:",best_model_score)

    
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model  
            )

            return(
                best_model_score,
                best_model_name,
                best_model
            )
        
        except Exception as e:
            CustomException(e,sys)

    def hyperparameter_tuning(self,model,model_name,X_train,y_train):
        try:
            logging.info("Entered the hyperparameter tuning method")
            params={
                "LinearRegression":{},
                "SVR":{},
                "SGDRegressor":{},
                "KNeighborsRegressor":{},
                "GaussianProcessRegressor":{},
                "DecisionTreeRegressor":{
                                        'criterion':['squared_error'],
                                        'max_features':['sqrt']
                                        },
                "GradientBoostingRegressor":{
                                            'criterion': ['squared_error'],
                                            'learning_rate': [0.5, 0.1, 0.05, 0.01],
                                            'max_depth': [5, 8, 10], 
                                            'max_features': ['sqrt'],
                                            'n_estimators': [8,16,32,64,128,256,512]
                                            },
                "RandomForestRegressor":{
                                        'criterion':['squared_error'],
                                        'n_estimators':[8,16,32,64,128,256,512]
                                        },
                "XGBRegressor":{'eval_metric':['rmse'],
                                'learning_rate':[.1,.01,.05,.001],
                                'n_estimators': [8,16,32,64,128,256,512]
                                },
                "MLPRegressor":{},

            }

            model_tuned=GridSearchCV(model,params[model_name],cv=5)
            model_tuned.fit(X_train,y_train)

            model=model.set_params(**model_tuned.best_params_)
            model.fit(X_train,y_train)
            logging.info("Hyperparameter tuning & fitting of model completed")

            tuned_model_dict={}
            tuned_model_dict[model_name]=model

            tuned_model_perf=evaluate_models(tuned_model_dict,X_train,y_train)

            print("RMSE of tuned model:",tuned_model_perf.iloc[0][2])
            print("R2 score of tuned model:",tuned_model_perf.iloc[0][1])

        except Exception as e:
            raise CustomException(e,sys)
            
