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
from sklearn.model_selection import RandomizedSearchCV
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

            print(best_model_name)
            print(best_model_score)

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model  
            )

            return(
                best_model_score,
                best_model_name
            )
        
        except Exception as e:
            CustomException(e,sys)
