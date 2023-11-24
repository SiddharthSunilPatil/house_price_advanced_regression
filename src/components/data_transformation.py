import os
import sys

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.base import BaseEstimator,TransformerMixin

import numpy as np
import pandas as pd

class numeric_cat_converter(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.astype(str)
        return X

#class log_transformer(BaseEstimator, TransformerMixin):
    #def fit(self,X,y=None):
        #return self
    
    #def transform(self,X):
        #X['SalePriceLog']=np.log(X['SalePrice'])
        #X=X.drop(['SalePrice'],axis=1)
        #return X

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    feature_transform_obj_file_path=os.path.join('artifacts',"featuretransform.csv")
    target_variable_obj_file_path=os.path.join('artifacts',"targetvariable.csv")
    ft_train_data_path=os.path.join('artifacts',"train_data.csv")
    ft_train_target_path=os.path.join('artifacts',"train_target.csv")
    ft_test_data_path=os.path.join('artifacts',"test_data.csv")
    ft_test_target_path=os.path.join('artifacts',"test_target.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation

        '''
        try:
            #target_column=['SalePrice']
            pipeline_one_columns=['MSSubClass','YearBuilt','YearRemodAdd','MoSold','YrSold']
            pipeline_two_columns=['MSZoning','Exterior1st','Exterior2nd','MasVnrType','Electrical',
                                  'Functional','SaleType']
            pipeline_three_columns=['LotFrontage']
            pipeline_four_columns=['LotArea','OverallQual','OverallCond','1stFlrSF','2ndFlrSF','LowQualFinSF',
                                   'GrLivArea','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd',
                                   'Fireplaces','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
                                   'ScreenPorch','PoolArea','MiscVal']
            pipeline_five_columns=['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
                                   'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                   'Foundation','Heating','CentralAir','SaleCondition']
            pipeline_six_columns=['Utilities','KitchenQual']
            pipeline_six_categories=[["ELO","NoSeWa","NoSeWr","AllPub"],["Po","Fa","TA","Gd","Ex"]]
            pipeline_seven_columns=['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
                                    'BsmtHalfBath','GarageCars','GarageArea']
            pipeline_eight_columns=['ExterQual','ExterCond','HeatingQC','PavedDrive']
            pipeline_eight_categories=[["Po","Fa","TA","Gd","Ex"],["Po","Fa","TA","Gd","Ex"],
                                       ["Po","Fa","TA","Gd","Ex"],["N","P","Y"]]
            pipeline_nine_columns=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                                   'FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC']
            pipeline_nine_categories=[["NA","Po","Fa","TA","Gd","Ex"],["NA","Po","Fa","TA","Gd","Ex"],
                                      ["NA","No","Mn","Av","Gd"],["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
                                      ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],["NA","Po","Fa","TA","Gd","Ex"],
                                      ["NA","Unf","RFn","Fin"],["NA","Po","Fa","TA","Gd","Ex"],
                                      ["NA","Po","Fa","TA","Gd","Ex"],["NA","Fa","TA","Gd","Ex"]
                                      ]
            pipeline_ten_columns=['Alley','GarageType','Fence','MiscFeature']
            pipeline_eleven_columns=['GarageYrBlt']

            #target_pipe=Pipeline(
            #    steps=[
            #        ("Log transformation",log_transformer())
            #    ]
            #)

            pipeline_one=Pipeline(
                steps=[
                    ("numeric to categorical converter",numeric_cat_converter()),
                    ("Nominal encoding",OneHotEncoder())
                ]
            )

            pipeline_two=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Nominal encoding",OneHotEncoder())
                ]
            )

            pipeline_three=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median"))
                ]
            )

            pipeline_five=Pipeline(
                steps=[
                    ("Nominal encoding",OneHotEncoder())
                ]
            )

            pipeline_six=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Ordinal encoding",OrdinalEncoder(categories=pipeline_six_categories))   
                ]
            )

            pipeline_seven=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="constant",fill_value=0))  
                ]
            )

            pipeline_eight=Pipeline(
                steps=[
                    ("Ordinal encoding",OrdinalEncoder(categories=pipeline_eight_categories))  
                ]
            )

            pipeline_nine=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="constant",fill_value="NA")),
                    ("Ordinal encoding",OrdinalEncoder(categories=pipeline_nine_categories))  
                ]
            )

            pipeline_ten=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="constant",fill_value="NA")),
                    ("Nominal encoding",OneHotEncoder())  
                ]
            )

            pipeline_eleven=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="constant",fill_value=0)),
                    ("numeric to categorical converter",numeric_cat_converter()),
                    ("Nominal encoding",OneHotEncoder())  
                ]
            )

            preprocessor=ColumnTransformer(
                [#("targetcolumn",target_pipe,target_column),
                 ("Pipeline1",pipeline_one,pipeline_one_columns),
                 ("Pipeline2",pipeline_two,pipeline_two_columns),
                 ("Pipeline3",pipeline_three,pipeline_three_columns),
                 ("Pipeline5",pipeline_five,pipeline_five_columns),
                 ("Pipeline6",pipeline_six,pipeline_six_columns),
                 ("Pipeline7",pipeline_seven,pipeline_seven_columns),
                 ("Pipeline8",pipeline_eight,pipeline_eight_columns),
                 ("Pipeline9",pipeline_nine,pipeline_nine_columns),
                 ("Pipeline10",pipeline_ten,pipeline_ten_columns),
                 ("Pipeline11",pipeline_eleven,pipeline_eleven_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,concat_path):

        try:
            logging.info("Entered Data Transformation object")
            preprocessor_obj=self.get_data_transformer_obj()
            
            concat_df=pd.read_csv(concat_path)
            logging.info("Read the concatenated data set")

            target_var=pd.DataFrame(np.log(concat_df["SalePrice"]))
            concat_df=concat_df.drop(["SalePrice"],axis=1)
            logging.info("Separation of target variable completed")


            df_transformed=preprocessor_obj.fit_transform(concat_df)
            df_transformed=df_transformed.toarray()
            df_transformed=pd.DataFrame(df_transformed)

            ##df_transformed.to_csv(self.data_transformation_config.feature_transform_obj_file_path,index=False,header=True)
            ##target_var.to_csv(self.data_transformation_config.target_variable_obj_file_path,index=False,header=True )
            logging.info("Feature transformation completed")
  
            logging.info("Entered the data split method")
            len_train=pd.read_csv('Notebook\\data\\train.csv').shape[0]
            X_train=df_transformed[:len_train]
            y_train=target_var[:len_train]
            X_test=df_transformed[len_train:]
            y_test=target_var[len_train:]
            logging.info("Splitting of dataset into train set and test set completed")
            y_train.to_csv(self.data_transformation_config.ft_train_target_path,index=False,header=True)
            y_test.to_csv(self.data_transformation_config.ft_test_target_path,index=False,header=True)


            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
      
            sc=StandardScaler()
            X_train=pd.DataFrame(sc.fit_transform(X_train))
            X_test=pd.DataFrame(sc.transform(X_test))
            logging.info("Feature scaling completed")

            X_train.to_csv(self.data_transformation_config.ft_train_data_path,index=False,header=True)
            X_test.to_csv(self.data_transformation_config.ft_test_data_path,index=False,header=True)
            
            

            return(
                self.data_transformation_config.preprocessor_obj_file_path,
                X_train,
                y_train,
                X_test,
                y_test
            )
        except Exception as e:
            raise CustomException(e,sys)
        


        

        
        

            
