####Regression Case Study#######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant, OLS

df = pd.read_csv("C:\\Users\\shiva\\Desktop\\Edurekha_data\\DAY_37\\DIY\\insurance.csv")
print(df.head(10))

print(df.describe())

print(df.isna().sum()) # no missing value found.

print(df.info())

new_df = df.copy()

sns.boxplot(new_df['charges']) # dependent vaiable.
plt.show()


#Removing outlier using IQR method.
hp = sorted(new_df['charges'])
q1, q3= np.percentile(hp,[25,75])
lower_bound = q1 -(1.5 * (q3-q1)) 
upper_bound = q3 + (1.5 * (q3-q1))
below = new_df['charges'] > lower_bound
above = new_df['charges'] < upper_bound
new_df = new_df[below & above]

print(new_df.shape)

print(new_df.describe() )

sns.distplot(new_df['charges'])
plt.show()

new_df.describe().transpose()

fullRaw2 = pd.get_dummies(new_df).copy() 
print(fullRaw2.shape)
print(fullRaw2.head())
print(fullRaw2.shape)


x = fullRaw2.drop(["charges"], axis = 1).copy()
y = fullRaw2["charges"].copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=100) 


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

###Model building..

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train,y_train)
pred = model.predict(x_test)

## Co-effecient of determination (R-Square)
score1 = model.score(x_test,y_test)
print(score1)

# Adjusted R square.
1 - (1-model.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
fig.suptitle('[Residual Plots]')
fig.set_size_inches(12,5)
axes.plot(model.predict(x_test), y_test-model.predict(x_test), 'bo')
axes.axhline(y=0, color='k')
axes.grid()
axes.set_title('Linear')
axes.set_xlabel('predicted values')
axes.set_ylabel('residuals')


import seaborn as sns

residuals_linear = y_test - model.predict(x_test)
sns.distplot(residuals_linear)
plt.title('Linear')


#visulizing model coefficients

predictors = x_train.columns
coef = pd.Series(model.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Model Coefficients')

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns


### any value of VIF higher than 10 creates a problem in my model.
vif["VIF"]

x = add_constant(x)
tempMaxVIF = 5
maxVIF = 5
trainXCopy = x.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIF):
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"])[-1:]["Column_Name"].values[0]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]
    
    if (tempMaxVIF >= maxVIF):
        print(counter)
        print(tempColumnName)
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName) # here we are making list of deleting variables 
    
    counter = counter + 1

# Remove all those variables which have high VIF

print(highVIFColumnNames)

highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is the intercept.
print(highVIFColumnNames)
print(len(highVIFColumnNames))

x_new = x.drop(highVIFColumnNames, axis = 1)
print(x.shape)

print(x_new)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_new,y,test_size = 0.20,random_state=10) 
m1ModelDef = OLS(y_train2,x_train2) # (Dep_Var, Indep_Vars) # This is model definition
m1ModelBuild = m1ModelDef.fit() # This is model building. fit() creates the linear regression equation.
m1ModelBuild.summary()


score3 =  m1ModelBuild.rsquared
print(score3)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

ridgeReg = Ridge(alpha=0.00001, normalize=True)
x3 = fullRaw2.drop(["charges"], axis = 1).copy()
y3 = fullRaw2["charges"].copy()
x_train3,x_test3,y_train3,y_test3 = train_test_split(x3,y3,test_size = 0.20,random_state=150) 

ridgeReg.fit(x_train3,y_train3)
pred = ridgeReg.predict(x_test3)
score4 = ridgeReg.score(x_test3,y_test3)

print(score4)

ridgeReg.coef_

predictors = x_train.columns
coef = pd.Series(ridgeReg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

lassoReg = Lasso(alpha=0.0001)
lassoReg.fit(x_train3,y_train3)
pred = lassoReg.predict(x_test3)
score5 = lassoReg.score(x_test3,y_test3)
print(score5)

lassoReg.coef_
predictors = x_train.columns
coef = pd.Series(lassoReg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')

print("all model score is:")
print("simple linear regression:          ",score1)
print("After VIF simple linear regression:",score3)
print("ridge regression:                  ",score4)
print("lasso regression:                  ",score5)
