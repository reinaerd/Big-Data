import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

##some configuration so that we can view everything in the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

##importing the datasets

training_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")
data = training_dataset
data.drop('Id',axis=1, inplace=True)

top15cols = data.corr().nlargest(10,'SalePrice')['SalePrice'].index
top15corr = np.corrcoef(data[top15cols].values.T)
sns.heatmap(top15corr,yticklabels=top15cols.values,xticklabels=top15cols.values, annot=True, cbar=False)
plt.show()

"""
first we will check the data and make sure it is correct:
we should check the years, make sure that there aren't any sales that exceed the current year because that would be impossible
we should check the prices and measurements to see if they are not negative
"""

columns_with_years = ['YearBuilt','GarageYrBlt','YrSold','YearRemodAdd']
columns_with_measurements = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF', 
                             'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

##here we print the maximum years for a specific row and check if it exceeds 2022 => in this case we saw that the test dataset had a max of 2207 for GarageYrBlt
print(data[columns_with_years].max()) 

##here we check with an if statement if any columns with years in them exceed 2022, if so we grab the row
wrong_years = (data[columns_with_years] > 2022).any(axis=1)
print("rows with a wrong year:") 
print(data[wrong_years])

##in this situation we can see that the yearbuilt of the wrong garageyrblt entry is right so we replace garageyrblt with the yearblt value of the house
data.loc[wrong_years,'GarageYrBlt'] = data[wrong_years]['YearBuilt'] 

##now the years are all good 
print(data[columns_with_years].max())

##next up we will check the measurements
wrong_measurements = (data[columns_with_measurements] < 0).any(axis=1)

##this is empty, thus there are no wrong measurements
print("rows with impossible measurements:")
print(data[wrong_measurements])

##lets see what our data looks like again
print(data.shape)

##next up we will split the features and modify the data
# we want to split the features containing sqfeet, prices, etc
# and the features with data containing "Excelllent, good, etc."
# we will then convert the 2nd list of features to a numerical scale since this will be easier to use
numerical_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
             'GrLivArea', 'PoolArea', 'PoolQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
             'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'MiscVal','YrSold', 'SalePrice']

grading_features = ['OverallQual','OverallCond','GarageCond','GarageQual','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','PoolQC','FireplaceQu']  
text_grades = ['Ex','Gd','TA','Fa','Po']
numerical_grading = [9,7,5,3,1]
gradingdictionary = dict(zip(['Ex','Gd','TA','Fa','Po'],[9,7,5,3,1]))
data[grading_features] = data[grading_features].replace(gradingdictionary)

##these are the features that contain categories of data
categorical_features = data.drop(numerical_features,axis=1).columns

print(categorical_features)

#only show the columns that have missing values, not showing columns with 0
print(data.isnull().sum()[data.isnull().sum()>0])

##get all numerical and categorical columns and set their missing values to 0

##not a lot of missing features for these columns so we just took the mode of the column of the neighborhood for each house
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

print(data.isna().sum())

##we want to change the datatypes of some columns since they are a weird choice for the values containing them
#mosold should be an object since its categorical
#half and fullbath should be int64 since you can have halves
#mssubclass is also categorical
#and garagecars same as half and fullbath, you cant have half a car
data['MoSold'] = data['MoSold'].astype('object', copy=False)
data['BsmtHalfBath'] = data['BsmtHalfBath'].astype('int64', copy=False)
data['BsmtFullBath'] = data['BsmtFullBath'].astype('int64', copy=False)
data['MSSubClass'] = data['MSSubClass'].astype('object', copy=False)
data['GarageCars'] = data['GarageCars'].astype('int64', copy=False)

##now we want to modify the categorical features that only have 2 values into binary (0 or 1) values
##we do this because get_dummies would split them into 2 columns

categorical_data = data[categorical_features]

print(data.shape)

for feature in categorical_features:
    unique_values = pd.unique(data[feature])
    if(len(unique_values) < 2):
        data.drop(feature,axis=1,inplace=True)
    elif(len(unique_values) < 3):
        data[feature].replace({unique_values[0]:0,unique_values[1]:1},inplace=True)
        data[feature] = data[feature].astype('int8')
    else:
        data[feature] = data[feature].astype('category')
        

##next up, analysing the data and checking which values would be influencing the saleprice a lot
sns.set(rc={'figure.figsize':(12,8)})
sns.heatmap(data.corr(),vmin=0.5,square=True, linewidths=0.05, linecolor='white')
plt.show()


##so with this we looked up how to make a more detailed heatmap and created this one:
top15cols = data.corr().nlargest(15,'SalePrice')['SalePrice'].index
top15corr = np.corrcoef(data[top15cols].values.T)
sns.heatmap(top15corr,yticklabels=top15cols.values,xticklabels=top15cols.values, annot=True, cbar=False)
plt.show()

##first up we will look at the OverallQual
sns.boxplot(x=data['OverallQual'],y=data['SalePrice'])
plt.show()

plt.scatter(data['GrLivArea'],data['SalePrice'])
plt.xlabel("Above ground living area sqfeet")
plt.ylabel("Pricing")
plt.show()

plt.scatter(data['TotalBsmtSF'],data['SalePrice'])
plt.xlabel("Basement area sqfeet")
plt.ylabel("Pricing")
plt.show()

##add garagecars/saleprice and garagearea/saleprice
sns.boxplot(x=data['GarageCars'], y=data['SalePrice'])
plt.show()


plt.scatter(data['GarageArea'],data['SalePrice'])
plt.xlabel("Garage sqfeet")
plt.ylabel('Pricing')
plt.show()

finaldataformodel = pd.get_dummies(data)

#this is a list containing variables, if they were NA or 0, the house does not have them
feat_list = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath',
        'BsmtHalfBath', 'TotalBsmtSF', 'GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageCars', 
          'GarageArea', 'GarageYrBlt', 'Fireplaces', 'FireplaceQu','MasVnrType', 'MasVnrArea','Alley', 'Fence', 'PoolQC', 'MiscFeature']
for feature in finaldataformodel.columns:
    if('_0' in feature) and (feature.split('_')[0] in feat_list):
        finaldataformodel.drop(feature,axis=1,inplace=True)


#prediction time
reg = LinearRegression()
y = finaldataformodel['SalePrice']
X = finaldataformodel.drop('SalePrice',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(reg.score(X_test,y_test))

##next up, Ridge
ridge = Pipeline([("scaler",StandardScaler(with_mean=False)),("ridge",Ridge(alpha=0.1))])
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge score:----------")
print(ridge.score(X_test, y_test))

##next up, Lasso
lasso = Pipeline([("scaler",StandardScaler(with_mean=False)),("lasso",Lasso(alpha=0.1))])
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso score:----------")
print(lasso.score(X_test, y_test))

l1_space = np.linspace(0,1,30)
param_grid = {'l1_ratio':l1_space}

elastic_net = ElasticNet()

elastic_net.fit(X,y)
gm_cv = GridSearchCV(elastic_net,param_grid)
gm_cv.fit(X, y)

gm_cv_y_pred = gm_cv.predict(X_test)
print(gm_cv.score(X_test,y_test))



