# Housing Price Advanced Regression
## This Project aims to predict house prices based on 80+ features using advanced regression techniques 

### The dependent or target variables is:
1. **SalePrice** - The property's sale price in dollars. This is the target variable that you're trying to predict.

### The independent variables or features are as below
2. **MSSubClass:** The building class
3. **MSZoning:** The general zoning classification
4. **LotFrontage:** Linear feet of street connected to property
5. **LotArea:** Lot size in square feet
6. **Street:** Type of road access
7. **Alley:** Type of alley access
8. **LotShape:** General shape of property
9. **LandContour:** Flatness of the property
10. **Utilities:** Type of utilities available
11. **LotConfig:** Lot configuration
12. **LandSlope:** Slope of property
13. **Neighborhood:** Physical locations within Ames city limits
14. **Condition1:** Proximity to main road or railroad
14. **Condition2:** Proximity to main road or railroad (if a second is present)
15. **BldgType:** Type of dwelling
16. **HouseStyle:** Style of dwelling
17. **OverallQual:** Overall material and finish quality
18. **OverallCond:** Overall condition rating
20. **YearBuilt:** Original construction date
21. **YearRemodAdd:** Remodel date
22. **RoofStyle:** Type of roof
23. **RoofMatl:** Roof material
24. **Exterior1st:** Exterior covering on house
25. **Exterior2nd:** Exterior covering on house (if more than one material)
26. **MasVnrType:** Masonry veneer type
27. **MasVnrArea:** Masonry veneer area in square feet
28. **ExterQual:** Exterior material quality
29. **ExterCond:** Present condition of the material on the exterior
30. **Foundation:** Type of foundation
31. **BsmtQual:** Height of the basement
32. **BsmtCond:** General condition of the basement
33. **BsmtExposure:** Walkout or garden level basement walls
34. **BsmtFinType1:** Quality of basement finished area
35. **BsmtFinSF1:** Type 1 finished square feet
36. **BsmtFinType2:** Quality of second finished area (if present)
37. **BsmtFinSF2:** Type 2 finished square feet
38. **BsmtUnfSF:** Unfinished square feet of basement area
39. **TotalBsmtSF:** Total square feet of basement area
40.  **Heating:** Type of heating
41. **HeatingQC:** Heating quality and condition
42. **CentralAir:** Central air conditioning
43. **Electrical:** Electrical system
44. **1stFlrSF:** First Floor square feet
45. **2ndFlrSF:** Second floor square feet
46. **LowQualFinSF:** Low quality finished square feet (all floors)
47. **GrLivArea:** Above grade (ground) living area square feet
48. **BsmtFullBath:** Basement full bathrooms
49. **BsmtHalfBath:** Basement half bathrooms
50. **FullBath:** Full bathrooms above grade
51. **HalfBath:** Half baths above grade
52. **Bedroom:** Number of bedrooms above basement level
53. **Kitchen:** Number of kitchens
54. **KitchenQual:** Kitchen quality
55. **TotRmsAbvGrd:** Total rooms above grade (does not include bathrooms)
56. **Functional:** Home functionality rating
57. **Fireplaces:** Number of fireplaces
58. **FireplaceQu:** Fireplace quality
59. **GarageType:** Garage location
60. **GarageYrBlt:** Year garage was built
61. **GarageFinish:** Interior finish of the garage
62. **GarageCars:** Size of garage in car capacity
63. **GarageArea:** Size of garage in square feet
64. **GarageQual:** Garage quality
65. **GarageCond:** Garage condition
66. **PavedDrive:** Paved driveway
67. **WoodDeckSF:** Wood deck area in square feet
68. **OpenPorchSF:** Open porch area in square feet
69. **EnclosedPorch:** Enclosed porch area in square feet
70. **3SsnPorch:** Three season porch area in square feet
71. **ScreenPorch:** Screen porch area in square feet
72. **PoolArea:** Pool area in square feet
73. **PoolQC:** Pool quality
74. **Fence:** Fence quality
75. **MiscFeature:** Miscellaneous feature not covered in other categories
76. **MiscVal:** Value of miscellaneous feature
77. **MoSold:** Month Sold
78. **YrSold:** Year Sold
79. **SaleType:** Type of sale
80. **SaleCondition:** Condition of sale

## Datasource link: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## AWS deployment link: http://ames-house-price-prediction-env.eba-pxy9mxps.us-east-2.elasticbeanstalk.com

## Exploratory data analysis [EDA] link: https://github.com/SiddharthSunilPatil/house_price_advanced_regression/blob/main/Notebook/housingprice.ipynb

## Project Approach

**1. Data Ingestion:** In this phase, both test and train dataset are read from csv file. Since both train set and test set contain missing values, data is first concatenated and prepared for analysis.

**2. Data Transformation:** The concatenated data is then passed through a column transformer pipeline. Based on the variable the transformer performs various preprocessing treatments such as simple imputing, nominal encoding, ordinal encoding and stadard scaling. The transformed data is then passed to the model trainer.

**3. Model Trainer:** Various models are tested to find out the best perfroming model based on Rott Mean Squred Error[RMSE] score. The best model found was Gradient Boosting Regressor. The model is then improved by hyperparameter tuning.

**4. Prediciton Pipeline:** This pipleline converts input data into a dataframe and loads pickle files for data transformation and model training and predicts final results.

**5. Deployment:** The project is deployed on amazon elastic beanstalk as a flask application to predict house prices.

## Installation

**1. Cloning the repository

1.1. Create a dirctory on your drive.  
1.1. Open anaconda prompt and navigate to the directory with the command "cd <directory path>"  
1.2. 
**1. Setting up the environment using VS Code and anaconda prompt**

1.1. Create a directory on your drive.  
1.2. Navigate to the directory with "cd <directory path>" in annaconda prompt.  
1.3. Launch VS code with "code .".  
1.4  Create virtual environment with "conda create -p venv python -y".  
1.5  Activate environment with "conda activate venv/".  

**2. Cloning the repository**

2.1. Use command "git clone https://github.com/SiddharthSunilPatil/house_price_advanced_regression.git
" to clone repository to existing directory

**3. Installing dependencies






