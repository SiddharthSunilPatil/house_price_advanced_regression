from flask import Flask,request,render_template 
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


    


application=Flask(__name__)

app=application

#route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    #path1=os.path.join('artifacts','trial_data.csv')
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            MSSubClass=request.form.get('MSSubClass'),  
            MSZoning=request.form.get('MSZoning'),
            LotFrontage=request.form.get('LotFrontage'),
            LotArea=request.form.get('LotArea'),  
            Street=request.form.get('Street'),
            Alley=request.form.get('Alley'),
            LotShape=request.form.get('LotShape'),
            LandContour=request.form.get('LandContour'),
            Utilities=request.form.get('Utilities'),
            LotConfig=request.form.get('LotConfig'),
            LandSlope=request.form.get('LandSlope'),
            Neighborhood=request.form.get('Neighborhood'),
            Condition1=request.form.get('Condition1'),
            Condition2=request.form.get('Condition2'),
            BldgType=request.form.get('BldgType'),
            HouseStyle=request.form.get('HouseStyle'),
            OverallQual=request.form.get('OverallQual'),
            OverallCond=request.form.get('OverallCond'),
            YearBuilt=request.form.get('YearBuilt'),
            YearRemodAdd=request.form.get('YearRemodAdd'),
            RoofStyle=request.form.get('RoofStyle'),
            RoofMatl=request.form.get('RoofMatl'),
            Exterior1st=request.form.get('Exterior1st'),
            Exterior2nd=request.form.get('Exterior2nd'),
            MasVnrType=request.form.get('MasVnrType'),
            MasVnrArea=request.form.get('MasVnrArea'),
            ExterQual=request.form.get('ExterQual'),
            ExterCond=request.form.get('ExterCond'),
            Foundation=request.form.get('Foundation'),
            BsmtQual=request.form.get('BsmtQual'),
            BsmtCond=request.form.get('BsmtCond'),
            BsmtExposure=request.form.get('BsmtExposure'),
            BsmtFinType1=request.form.get('BsmtFinType1'),
            BsmtFinSF1=request.form.get('BsmtFinSF1'),
            BsmtFinType2=request.form.get('BsmtFinType2'),
            BsmtFinSF2=request.form.get('BsmtFinSF2'),
            BsmtUnfSF=request.form.get('BsmtUnfSF'),
            TotalBsmtSF=request.form.get('TotalBsmtSF'),
            Heating=request.form.get('Heating'),
            HeatingQC=request.form.get('HeatingQC'),
            CentralAir=request.form.get('CentralAir'),
            Electrical=request.form.get('Electrical'),
            _1stFlrSF=request.form.get('1stFlrSF'),
            _2ndFlrSF=request.form.get('2ndFlrSF'),
            LowQualFinSF=request.form.get('LowQualFinSF'),
            GrLivArea=request.form.get('GrLivArea'),
            BsmtFullBath=request.form.get('BsmtFullBath'),
            BsmtHalfBath=request.form.get('BsmtHalfBath'),
            FullBath=request.form.get('FullBath'),
            HalfBath=request.form.get('HalfBath'),
            BedroomAbvGr=request.form.get('BedroomAbvGr'), 
            KitchenAbvGr=request.form.get('KitchenAbvGr'),
            KitchenQual=request.form.get('KitchenQual'),
            TotRmsAbvGrd=request.form.get('TotRmsAbvGrd'),
            Functional=request.form.get('Functional'),
            Fireplaces=request.form.get('Fireplaces'),  
            FireplaceQu=request.form.get('FireplaceQu'), 
            GarageType=request.form.get('GarageType'),
            GarageYrBlt=request.form.get('GarageYrBlt'),
            GarageFinish=request.form.get('GarageFinish'),
            GarageCars=request.form.get('GarageCars'),
            GarageArea=request.form.get('GarageArea'),
            GarageQual=request.form.get('GarageQual'),
            GarageCond=request.form.get('GarageCond'),
            PavedDrive=request.form.get('PavedDrive'),
            WoodDeckSF=request.form.get('WoodDeckSF'),
            OpenPorchSF=request.form.get('OpenPorchSF'),
            EnclosedPorch=request.form.get('EnclosedPorch'),  
            _3SsnPorch=request.form.get('3SsnPorch'),
            ScreenPorch=request.form.get('ScreenPorch'),  
            PoolArea=request.form.get('PoolArea'),
            PoolQC=request.form.get('PoolQC'), 
            Fence=request.form.get('Fence'),
            MiscFeature=request.form.get('MiscFeature'),
            MiscVal=request.form.get('MiscVal'), 
            MoSold=request.form.get('MoSold'), 
            YrSold=request.form.get('YrSold'),  
            SaleType=request.form.get('SaleType'), 
            SaleCondition=request.form.get('SaleCondition')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
        
    

if __name__=="__main__":
    app.run(host="0.0.0.0")



