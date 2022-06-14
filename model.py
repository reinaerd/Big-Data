import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

training_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")

test_id = test_dataset['Id']

data = pd.concat([training_dataset.drop('SalePrice', axis=1), test_dataset], keys=['train', 'test'])
data.drop(['Id'], axis=1, inplace=True)

columns_with_years = ['YearBuilt','GarageYrBlt','YrSold','YearRemodAdd']
columns_with_measurements = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF', 
                             'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
wrong_years = (data[columns_with_years] > 2022).any(axis=1)
data.loc[wrong_years,'GarageYrBlt'] = data[wrong_years]['YearBuilt'] 

numerical_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
             'GrLivArea', 'PoolArea', 'PoolQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
             'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'MiscVal','YrSold']

grading_features = ['OverallQual','OverallCond','GarageCond','GarageQual','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','PoolQC','FireplaceQu']  
text_grades = ['Ex','Gd','TA','Fa','Po']
numerical_grading = [9,7,5,3,1]
gradingdictionary = dict(zip(['Ex','Gd','TA','Fa','Po'],[9,7,5,3,1]))
data[grading_features] = data[grading_features].replace(gradingdictionary)

categorical_features = data.drop(numerical_features,axis=1).columns

features = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional',
         'SaleType']
def fun(x):
    return x.mode().iloc[0]
model = data.groupby('Neighborhood')[features].apply(fun)

for f in features:
    data[f].fillna(data['Neighborhood'].map(model[f]), inplace=True)

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['KitchenQual'].fillna(data['OverallQual'], inplace=True)


category = data.columns[data.dtypes == 'object']
numerical = list(set(data.columns)-set(category))

data['MasVnrType'] = data['MasVnrType'].replace({'None':np.nan})

data[category] = data[category].fillna('0')
data[numerical] = data[numerical].fillna(0)

data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)

categorical_data = data[categorical_features]

for feature in categorical_features:
    unique_values = pd.unique(data[feature])
    if(len(unique_values) < 2):
        data.drop(feature,axis=1,inplace=True)
    elif(len(unique_values) < 3):
        data[feature].replace({unique_values[0]:0,unique_values[1]:1},inplace=True)
        data[feature] = data[feature].astype('int8')
    else:
        data[feature] = data[feature].astype('category')

finaldataformodel = pd.get_dummies(data)

feat_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath',
        'BsmtHalfBath', 'TotalBsmtSF', 'GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageCars', 
          'GarageArea', 'GarageYrBlt', 'Fireplaces', 'FireplaceQu','MasVnrType', 'MasVnrArea','Alley', 'Fence', 'PoolQC', 'MiscFeature']
for feature in finaldataformodel.columns:
    if('_0' in feature) and (feature.split('_')[0] in feat_list):
        finaldataformodel.drop(feature,axis=1,inplace=True)


test = finaldataformodel.loc['test']
train = finaldataformodel.loc['train']

X = train
y = training_dataset['SalePrice']

elastic_net = ElasticNet(l1_ratio=0.9655172413793103)

elastic_net.fit(X,y)

submission = elastic_net.predict(test)
result = pd.DataFrame()
result['Id'] = test_dataset['Id']
result['SalePrice'] = submission

result.to_csv('submission.csv',index=False)






