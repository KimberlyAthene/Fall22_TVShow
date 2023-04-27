# ISYE 6740 Project
# This is the analysis portion for the project
# Using the dataset formed in 'data sorting code.py' file
# For the Project: Predicting Longevity of TV Shows

from cmath import log, nan
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing, utils


# To start, read in the dataset formed from data pulled from datasets.imdbws.com
#     and then sorted specifically for TV Shows
a = os.path.abspath(os.path.dirname('project analysis.py'))
b = os.path.join(a, 'data.csv')
data = pd.read_csv(b)
#print(data.head)


# I need to clean the dataset. 
#     Need to create dummies for the genres and bring them together for easier analysis
genre = pd.get_dummies(data['genres'])
#print(dat.head)
secgenre = pd.get_dummies(data['secGenres'])
#print(da.head)

datt = pd.concat([genre, secgenre]).groupby(level=0).any().astype(int)
#print(datt)

# I now have 27 columns of genres (in dummy variable form) for the qualitative variables
# Now I need to merge the genre df with the rest of the data and check that correlation

# Create a new df and drop the 'genre' and 'secGenre' columns (b/c changed to dummies)
quant = data.copy()
quant.drop(['tconst','titleType','genres','secGenres', 'maxSeason','archive_footage','startYear','endYear'], axis=1, inplace=True)
#print(quant)

df = pd.concat([quant, datt], axis=1)
df = df.fillna(0)
#print(df.head)



# Confusion Matrix for Correlation
# Create a correlation heatmap matrix <----------------------- Commented out for now for quicker processing
cormat = df.corr()
#plt.figure()
#htmap = sns.heatmap(cormat, cmap='viridis')
#plt.title("Correlation Heatmap")
#plt.show()
# From the correlation matrix, it can be shown there is little to no correlation in regards to the variables
#     'archive_sound' and 'production_designer' so these variables won't be helpful in the regression analysis
# There is also low correlation between runYears and everything else with correlation ranging between 0.0 and 0.2
#     except for endYear, which was used to create runYears so that isn't much of a surprise. There is a high rate 
#     of correlation between these variables: 'cinematographer','composer','director','editor',and 'producer'
# There doesn't appear to be much strong correlation between the variables and runYears.The strong correlation
#     between the above variables is likely due to them being entire columns between 0 and 1 or 2. There is some
#     hope for a good result with variables 'maxSeason', 'avgMaxEps', and maybe 'averageRating' among others.


# Split data into train/test sets
y = df[['runYears']]
x = df.loc[:, df.columns != 'runYears']
#print(y.head)
#print(x.head) #<- tconst not included b/c it is used for imdb identification only
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=425)
#print(ytrain)



# Scatterplots for Nonlinear Variables
# Using the training set, need to check to see if any transformations should be performed with the highly correlated
#     data we found in the heatmap: 'cinematographer','composer','director','editor',and 'producer'
#print(xtrain)
#plt.figure()
#plt.scatter(xtrain['self'], xtrain['actor'])
#plt.title("Start Year vs. Years Run")
#plt.scatter(xtrain['maxSeason'], ytrain['runYears'])
#plt.scatter(log(xtrain['avgMaxEps']), log(ytrain['runtimeMinutes']))
#plt.hist(xtrain['genres'].astype(str))
#plt.show()


# Using sns pairplot and using the kde diagonal (Kim look up KDE from class and sprinkle in deliciousness)
#   it was clear that maxSeason and archive_footage had little to no correlation with the other variables and 
#   more specifically with RunYears so we are excluding that from the study
#sns.pairplot(df[['runYears','runtimeMinutes','avgMaxEps','actor','actress',
#                'self','writer','averageRating']], diag_kind='kde')
#plt.show()

# Finally a good graph. Okay. This shows there is a negative correlation between the Start Year and how many years
#     a show is likely to run. Which. A show that starts in 1943 has a longer amount of time to run than one that 
#     only started a year ago. It would be more prudent to see if there was an increase in tv shows being created
#     and then not continuing a run than to put too much stock into the year a show started.
# maxSeason vs runYears shows a positive trend which makes sense due to the way seasons work.



# Regression Types
# Ridge and LASSO are not useful for this because there are no significant high correlations among the different
#     variables except the few boolean variables.

# First we need to normalize the variables due to large range and deviations
xtrain_stats = xtrain.describe()
xtrain_stats = xtrain_stats.transpose()
ytrain_stats = ytrain.describe()
ytrain_stats = ytrain_stats.transpose()
#print(train_stats)

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']
normed_xtrain_data = norm(xtrain,xtrain_stats)
normed_xtest_data = norm(xtest,xtrain_stats)
normed_ytrain_data = norm(ytrain, ytrain_stats)
normed_ytest_data = norm(ytest, ytrain_stats)


#sns.pairplot(xtrain[['runtimeMinutes','avgMaxEps','actor','actress',
#                'self','writer','averageRating']], diag_kind='kde')
#plt.show()




#Start with multiple linear regression
mod1 = LinearRegression()
mod1.fit(normed_xtrain_data, normed_ytrain_data)
#Prediction using test set 
y_pred = mod1.predict(normed_xtest_data)
mae=metrics.mean_absolute_error(normed_ytest_data, y_pred)
mse=metrics.mean_squared_error(normed_ytest_data, y_pred)
# Printing the metrics
print('R2 square:',metrics.r2_score(normed_ytest_data, y_pred))
print('MAE: ', mae)
print('MSE: ', mse)


# See if this will work or nah
import statsmodels.api as sm
normed_xtrain_data = sm.add_constant(normed_xtrain_data)
model = sm.OLS(normed_ytrain_data, normed_xtrain_data)
results = model.fit()
#print(results.summary())







# Split data into train/test sets
y2 = df[['runYears']]
x2 = df[['avgMaxEps','self','writer','Adult','Crime', 'Family','Game-Show','News']]
#print(y.head)
#print(x.head) #<- tconst not included b/c it is used for imdb identification only
x2train, x2test, y2train, y2test = train_test_split(x2, y2, test_size=0.2, random_state=425)
#print(ytrain)

# First we need to normalize the variables due to large range and deviations
x2train_stats = x2train.describe()
x2train_stats = x2train_stats.transpose()
y2train_stats = y2train.describe()
y2train_stats = y2train_stats.transpose()
#print(train_stats)

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']
normed_x2train_data = norm(x2train,x2train_stats)
normed_x2test_data = norm(x2test,x2train_stats)
normed_y2train_data = norm(y2train, y2train_stats)
normed_y2test_data = norm(y2test, y2train_stats)


#sns.pairplot(xtrain[['runtimeMinutes','avgMaxEps','actor','actress',
#                'self','writer','averageRating']], diag_kind='kde')
#plt.show()




#Start with multiple linear regression
mod1 = LinearRegression()
mod1.fit(normed_x2train_data, normed_y2train_data)
#Prediction using test set 
y_pred = mod1.predict(normed_x2test_data)
mae=metrics.mean_absolute_error(normed_y2test_data, y_pred)
mse=metrics.mean_squared_error(normed_y2test_data, y_pred)
# Printing the metrics
print('R2 square:',metrics.r2_score(normed_y2test_data, y_pred))
print('MAE: ', mae)
print('MSE: ', mse)


# See if this will work or nah
import statsmodels.api as sm
normed_x2train_data = sm.add_constant(normed_x2train_data)
model = sm.OLS(normed_y2train_data, normed_x2train_data)
results = model.fit()
print(results.summary())



#Then logistic regression
# y has to be transformed to 0 or 1
#lab = preprocessing.LabelEncoder()
#y_trans = lab.fit_transform(normed_ytrain_data)
#mod2 = LogisticRegression()
#mod2.fit(normed_xtrain_data, y_trans.ravel())
#Prediction using test set 
#y_pred = mod2.predict(normed_xtest_data)
#mae=metrics.mean_absolute_error(normed_ytest_data, y_pred)
#mse=metrics.mean_squared_error(normed_ytest_data, y_pred)
# Printing the metrics
#print('R2 square:',metrics.r2_score(normed_ytest_data, y_pred))
#print('MAE: ', mae)
#print('MSE: ', mse)


# What about Decision Tree Regression 
#mod3 = DecisionTreeRegressor()
#mod3.fit(normed_xtrain_data, normed_ytrain_data)
#Prediction using test set 
#y_pred = mod3.predict(normed_xtest_data)
#mae=metrics.mean_absolute_error(normed_ytest_data, y_pred)
#mse=metrics.mean_squared_error(normed_ytest_data, y_pred)
# Printing the metrics
#print('R2 square:',metrics.r2_score(normed_ytest_data, y_pred))
#print('MAE: ', mae)
#print('MSE: ', mse)


# What about LASSO Regression 
#mod4 = Lasso(alpha=1.0)
#mod4.fit(normed_xtrain_data, normed_ytrain_data)
#Prediction using test set 
#y_pred = mod4.predict(normed_xtest_data)
#mae=metrics.mean_absolute_error(normed_ytest_data, y_pred)
#mse=metrics.mean_squared_error(normed_ytest_data, y_pred)
# Printing the metrics
#print('R2 square:',metrics.r2_score(normed_ytest_data, y_pred))
#print('MAE: ', mae)
#print('MSE: ', mse)


# What about Ridge Regression 
#mod5 = Ridge(alpha=500)
#mod5.fit(normed_xtrain_data, normed_ytrain_data)
#Prediction using test set 
#y_pred = mod5.predict(normed_xtest_data)
#mae=metrics.mean_absolute_error(normed_ytest_data, y_pred)
#mse=metrics.mean_squared_error(normed_ytest_data, y_pred)
# Printing the metrics
#print('R2 square:',metrics.r2_score(normed_ytest_data, y_pred))
#print('MAE: ', mae)
#print('MSE: ', mse)