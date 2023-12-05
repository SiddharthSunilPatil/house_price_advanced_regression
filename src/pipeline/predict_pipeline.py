import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            preds=np.exp(preds)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                MSSubClass:int,  
                MSZoning:str, 
                LotFrontage:float,
                LotArea:int,  
                Street:str,
                Alley:str,
                LotShape:str,
                LandContour:str, 
                Utilities:str,
                LotConfig:str,
                LandSlope:str,
                Neighborhood:str,
                Condition1:str,
                Condition2:str,
                BldgType:str,
                HouseStyle:str,
                OverallQual:int,
                OverallCond:int,  
                YearBuilt:int,  
                YearRemodAdd:int,  
                RoofStyle:str,
                RoofMatl:str,
                Exterior1st:str,
                Exterior2nd:str,
                MasVnrType:str,
                MasVnrArea:float,
                ExterQual:str,
                ExterCond:str,
                Foundation:str,
                BsmtQual:str,
                BsmtCond:str,
                BsmtExposure:str,
                BsmtFinType1:str,
                BsmtFinSF1:float,
                BsmtFinType2:str,
                BsmtFinSF2:float,
                BsmtUnfSF:float,
                TotalBsmtSF:float,
                Heating:str,
                HeatingQC:str,
                CentralAir:str,
                Electrical:str,
                _1stFlrSF:int,  
                _2ndFlrSF:int,  
                LowQualFinSF:int,  
                GrLivArea:int,  
                BsmtFullBath:float,
                BsmtHalfBath:float,
                FullBath:int,  
                HalfBath:int,  
                BedroomAbvGr:int,  
                KitchenAbvGr:int,  
                KitchenQual:str, 
                TotRmsAbvGrd:int,  
                Functional:str, 
                Fireplaces:int,  
                FireplaceQu:str, 
                GarageType:str,
                GarageYrBlt:float,
                GarageFinish:str,
                GarageCars:float,
                GarageArea:float,
                GarageQual:str, 
                GarageCond:str,
                PavedDrive:str,
                WoodDeckSF:int,  
                OpenPorchSF:int,  
                EnclosedPorch:int,  
                _3SsnPorch:int,  
                ScreenPorch:int,  
                PoolArea:int,  
                PoolQC:str, 
                Fence:str,
                MiscFeature:str,
                MiscVal:int,  
                MoSold:int,  
                YrSold:int,  
                SaleType:str, 
                SaleCondition:str
                    ):
        self.MSSubClass=MSSubClass  
        self.MSZoning=MSZoning
        self.LotFrontage=LotFrontage
        self.LotArea=LotArea  
        self.Street=Street
        self.Alley=Alley
        self.LotShape=LotShape
        self.LandContour=LandContour 
        self.Utilities=Utilities
        self.LotConfig=LotConfig
        self.LandSlope=LandSlope
        self.Neighborhood=Neighborhood
        self.Condition1=Condition1
        self.Condition2=Condition2
        self.BldgType=BldgType
        self.HouseStyle=HouseStyle
        self.OverallQual=OverallQual
        self.OverallCond=OverallCond
        self.YearBuilt=YearBuilt
        self.YearRemodAdd=YearRemodAdd
        self.RoofStyle=RoofStyle
        self.RoofMatl=RoofMatl
        self.Exterior1st=Exterior1st
        self.Exterior2nd=Exterior2nd
        self.MasVnrType=MasVnrType
        self.MasVnrArea=MasVnrArea
        self.ExterQual=ExterQual
        self.ExterCond=ExterCond
        self.Foundation=Foundation
        self.BsmtQual=BsmtQual
        self.BsmtCond=BsmtCond
        self.BsmtExposure=BsmtExposure
        self.BsmtFinType1=BsmtFinType1
        self.BsmtFinSF1=BsmtFinSF1
        self.BsmtFinType2=BsmtFinType2
        self.BsmtFinSF2=BsmtFinSF2
        self.BsmtUnfSF=BsmtUnfSF
        self.TotalBsmtSF=TotalBsmtSF
        self.Heating=Heating
        self.HeatingQC=HeatingQC
        self.CentralAir=CentralAir
        self.Electrical=Electrical
        self._1stFlrSF=_1stFlrSF
        self._2ndFlrSF=_2ndFlrSF  
        self.LowQualFinSF=LowQualFinSF
        self.GrLivArea=GrLivArea
        self.BsmtFullBath=BsmtFullBath
        self.BsmtHalfBath=BsmtHalfBath
        self.FullBath=FullBath
        self.HalfBath=HalfBath
        self.BedroomAbvGr=BedroomAbvGr  
        self.KitchenAbvGr=KitchenAbvGr
        self.KitchenQual=KitchenQual
        self.TotRmsAbvGrd=TotRmsAbvGrd  
        self.Functional=Functional 
        self.Fireplaces=Fireplaces  
        self.FireplaceQu=FireplaceQu 
        self.GarageType=GarageType
        self.GarageYrBlt=GarageYrBlt
        self.GarageFinish=GarageFinish
        self.GarageCars=GarageCars
        self.GarageArea=GarageArea
        self.GarageQual=GarageQual
        self.GarageCond=GarageCond
        self.PavedDrive=PavedDrive
        self.WoodDeckSF=WoodDeckSF  
        self.OpenPorchSF=OpenPorchSF
        self.EnclosedPorch=EnclosedPorch  
        self._3SsnPorch=_3SsnPorch
        self.ScreenPorch=ScreenPorch  
        self.PoolArea=PoolArea  
        self.PoolQC=PoolQC 
        self.Fence=Fence
        self.MiscFeature=MiscFeature
        self.MiscVal=MiscVal  
        self.MoSold=MoSold  
        self.YrSold=YrSold  
        self.SaleType=SaleType 
        self.SaleCondition=SaleCondition
              

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "MSSubClass":[self.MSSubClass], 
            "MSZoning":[self.MSZoning],
            "LotFrontage":[self.LotFrontage],
            "LotArea":[self.LotArea], 
            "Street":[self.Street],
            "Alley":[self.Alley],
            "LotShape":[self.LotShape],
            "LandContour":[self.LandContour], 
            "Utilities":[self.Utilities],
            "LotConfig":[self.LotConfig],
            "LandSlope":[self.LandSlope],
            "Neighborhood":[self.Neighborhood],
            "Condition1":[self.Condition1],
            "Condition2":[self.Condition2],
            "BldgType":[self.BldgType],
            "HouseStyle":[self.HouseStyle],
            "OverallQual":[self.OverallQual],
            "OverallCond":[self.OverallCond],
            "YearBuilt":[self.YearBuilt],
            "YearRemodAdd":[self.YearRemodAdd],
            "RoofStyle":[self.RoofStyle],
            "RoofMatl":[self.RoofMatl],
            "Exterior1st":[self.Exterior1st],
            "Exterior2nd":[self.Exterior2nd],
            "MasVnrType":[self.MasVnrType],
            "MasVnrArea":[self.MasVnrArea],
            "ExterQual":[self.ExterQual],
            "ExterCond":[self.ExterCond],
            "Foundation":[self.Foundation],
            "BsmtQual":[self.BsmtQual],
            "BsmtCond":[self.BsmtCond],
            "BsmtExposure":[self.BsmtExposure],
            "BsmtFinType1":[self.BsmtFinType1],
            "BsmtFinSF1":[self.BsmtFinSF1],
            "BsmtFinType2":[self.BsmtFinType2],
            "BsmtFinSF2":[self.BsmtFinSF2],
            "BsmtUnfSF":[self.BsmtUnfSF],
            "TotalBsmtSF":[self.TotalBsmtSF],
            "Heating":[self.Heating],
            "HeatingQC":[self.HeatingQC],
            "CentralAir":[self.CentralAir],
            "Electrical":[self.Electrical],
            "1stFlrSF":[self._1stFlrSF],
            "2ndFlrSF":[self._2ndFlrSF], 
            "LowQualFinSF":[self.LowQualFinSF],
            "GrLivArea":[self.GrLivArea],
            "BsmtFullBath":[self.BsmtFullBath],
            "BsmtHalfBath":[self.BsmtHalfBath],
            "FullBath":[self.FullBath],
            "HalfBath":[self.HalfBath],
            "BedroomAbvGr":[self.BedroomAbvGr], 
            "KitchenAbvGr":[self.KitchenAbvGr],
            "KitchenQual":[self.KitchenQual],
            "TotRmsAbvGrd":[self.TotRmsAbvGrd], 
            "Functional":[self.Functional],
            "Fireplaces":[self.Fireplaces],
            "FireplaceQu":[self.FireplaceQu], 
            "GarageType":[self.GarageType],
            "GarageYrBlt":[self.GarageYrBlt],
            "GarageFinish":[self.GarageFinish],
            "GarageCars":[self.GarageCars],
            "GarageArea":[self.GarageArea],
            "GarageQual":[self.GarageQual],
            "GarageCond":[self.GarageCond],
            "PavedDrive":[self.PavedDrive],
            "WoodDeckSF":[self.WoodDeckSF],  
            "OpenPorchSF":[self.OpenPorchSF],
            "EnclosedPorch":[self.EnclosedPorch], 
            "3SsnPorch":[self._3SsnPorch],
            "ScreenPorch":[self.ScreenPorch],  
            "PoolArea":[self.PoolArea],  
            "PoolQC":[self.PoolQC], 
            "Fence":[self.Fence],
            "MiscFeature":[self.MiscFeature],
            "MiscVal":[self.MiscVal],  
            "MoSold":[self.MoSold],  
            "YrSold":[self.YrSold],  
            "SaleType":[self.SaleType], 
            "SaleCondition":[self.SaleCondition]
                }

            return pd.DataFrame(custom_data_input_dict).reset_index(drop=True)

        except Exception as e:
            raise CustomException(e, sys)