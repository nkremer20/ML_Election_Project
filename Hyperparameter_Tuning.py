#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#%% Importing and splitting data

election2000 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2000\Iowa_2000_Data.xlsx')
election2004 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2004\Iowa_2004_Data.xlsx')
election2008 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2008\Iowa_2008_Data.xlsx')
election2012 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2012\Iowa_2012_Data.xlsx')
election2016 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2016\Iowa_2016_Data.xlsx')

#remove unneeded columns
del election2000['Unnamed: 0']
del election2004['Unnamed: 0']
del election2008['Unnamed: 0']
del election2012['Unnamed: 0']
del election2016['Unnamed: 0']

del election2000['winner']
del election2004['winner']
del election2008['winner']
del election2012['winner']
del election2016['winner']

del election2000['total_other']
del election2004['total_other']
del election2008['total_other']
del election2012['total_other']
del election2016['total_other']

del election2000['total_votes']
del election2004['total_votes']
del election2008['total_votes']
del election2012['total_votes']
del election2016['total_votes']

del election2000['total_absentee']
del election2004['total_absentee']
del election2008['total_absentee']
del election2012['total_absentee']
del election2016['total_absentee']

#removing the republican dependent variable
del election2000['total_rep']
del election2004['total_rep']
del election2008['total_rep']
del election2012['total_rep']
del election2016['total_rep']

#remove the democrat independent variable
del election2000['total_dem']
del election2004['total_dem']
del election2008['total_dem']
del election2012['total_dem']
del election2016['total_dem']

del election2000['county']
del election2004['county']
del election2008['county']
del election2012['county']
del election2016['county']

#import spreadsheets so dependent variable can be used
dep_2000 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2000\Iowa_2000_Data.xlsx')
dep_2004 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2004\Iowa_2004_Data.xlsx')
dep_2008 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2008\Iowa_2008_Data.xlsx')
dep_2012 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2012\Iowa_2012_Data.xlsx')
dep_2016 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2016\Iowa_2016_Data.xlsx')

#create array for republican dependent variable
y_2000_r = dep_2000['total_rep'].values
y_2004_r = dep_2004['total_rep'].values
y_2008_r = dep_2008['total_rep'].values
y_2012_r = dep_2012['total_rep'].values
y_2016_r = dep_2016['total_rep'].values

#creating array for democrat dependent variable
y_2000_d = dep_2000['total_dem'].values
y_2004_d = dep_2004['total_dem'].values
y_2008_d = dep_2008['total_dem'].values
y_2012_d = dep_2012['total_dem'].values
y_2016_d = dep_2016['total_dem'].values

#create arrays for independent variables
x_2000 = election2000.values
x_2004 = election2004.values
x_2008 = election2008.values
x_2012 = election2012.values
x_2016 = election2016.values

#splitting into training and test data for republican dependent variable
x_train_2000_r, x_test_2000_r, y_train_2000_r, y_test_2000_r = train_test_split(x_2000, y_2000_r, test_size = 0.4, shuffle = True)

x_train_2004_r, x_test_2004_r, y_train_2004_r, y_test_2004_r = train_test_split(x_2004, y_2004_r, test_size = 0.4, shuffle = True)

x_train_2008_r, x_test_2008_r, y_train_2008_r, y_test_2008_r = train_test_split(x_2008, y_2008_r, test_size = 0.4, shuffle = True)

x_train_2012_r, x_test_2012_r, y_train_2012_r, y_test_2012_r = train_test_split(x_2012, y_2012_r, test_size = 0.4, shuffle = True)

x_train_2016_r, x_test_2016_r, y_train_2016_r, y_test_2016_r = train_test_split(x_2016, y_2016_r, test_size = 0.4, shuffle = True)

#combine test into one dataframe, and combine training into one dataframe for republican dependent variable
x_train_all_r = np.concatenate((x_train_2000_r, x_train_2004_r, x_train_2008_r, x_train_2012_r, x_train_2016_r),axis=0)

y_train_all_r = np.concatenate((y_train_2000_r, y_train_2004_r, y_train_2008_r, y_train_2012_r, y_train_2016_r),axis=0)

x_test_all_r = np.concatenate((x_test_2000_r, x_test_2004_r, x_test_2008_r, x_test_2012_r, x_test_2016_r),axis=0)

y_test_all_r = np.concatenate((y_test_2000_r, y_test_2004_r, y_test_2008_r, y_test_2012_r, y_test_2016_r),axis=0)

#splitting into training and test data for democrat dependent variable
x_train_2000_d, x_test_2000_d, y_train_2000_d, y_test_2000_d = train_test_split(x_2000, y_2000_d, test_size = 0.4, shuffle = True)

x_train_2004_d, x_test_2004_d, y_train_2004_d, y_test_2004_d = train_test_split(x_2004, y_2004_d, test_size = 0.4, shuffle = True)

x_train_2008_d, x_test_2008_d, y_train_2008_d, y_test_2008_d = train_test_split(x_2008, y_2008_d, test_size = 0.4, shuffle = True)

x_train_2012_d, x_test_2012_d, y_train_2012_d, y_test_2012_d = train_test_split(x_2012, y_2012_d, test_size = 0.4, shuffle = True)

x_train_2016_d, x_test_2016_d, y_train_2016_d, y_test_2016_d = train_test_split(x_2016, y_2016_d, test_size = 0.4, shuffle = True)

#combine test into one dataframe, and combine training into one dataframe for democrat dependent variable
x_train_all_d = np.concatenate((x_train_2000_d, x_train_2004_d, x_train_2008_d, x_train_2012_d, x_train_2016_d),axis=0)

y_train_all_d = np.concatenate((y_train_2000_d, y_train_2004_d, y_train_2008_d, y_train_2012_d, y_train_2016_d),axis=0)

x_test_all_d = np.concatenate((x_test_2000_d, x_test_2004_d, x_test_2008_d, x_test_2012_d, x_test_2016_d),axis=0)

y_test_all_d = np.concatenate((y_test_2000_d, y_test_2004_d, y_test_2008_d, y_test_2012_d, y_test_2016_d),axis=0)

#%% Tuning hyperparameters
parameters = {'n_estimators':[50, 100, 150, 200, 250, 300, 350], 
              'learning_rate':[0.05, 0.1, 0.15, 0.2], 
              'max_depth':[9, 12, 15, 17, 19]}

tuning = GridSearchCV(estimator = GradientBoostingRegressor(),
                      param_grid = parameters, scoring = 'r2')

#republican
print('Start Tuning')
tuning.fit(x_train_all_r, y_train_all_r)
print('End Tuning')
tuning.best_params_, tuning.best_score_

#democrat
print('Start Tuning')
tuning.fit(x_train_all_d, y_train_all_d)
print('End Tuning')
tuning.best_params_, tuning.best_score_
tuning.best_estimator_