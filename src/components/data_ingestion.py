import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    concat_data_path: str=os.path.join('artifacts',"concat.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            train_df=pd.read_csv('Notebook\\data\\train.csv')
            test_df=pd.read_csv('Notebook\\data\\test.csv')
            logging.info('Read the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df=pd.concat([train_df,test_df])
            df.to_csv(self.ingestion_config.concat_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.concat_data_path
                
    

        except Exception as e:
            raise CustomException(e,sys)
        
  
        
if __name__=="__main__":
    obj=DataIngestion()
    concat_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    _,X_train,y_train,X_test,y_test=data_transformation.initiate_data_transformation(concat_path)

    
    model_trainer=ModelTrainer()
    _,model_name,model=model_trainer.initiate_model_trainer(X_train,y_train)
    model_trainer.hyperparameter_tuning(model,model_name,X_train,y_train,X_test)


    

    
