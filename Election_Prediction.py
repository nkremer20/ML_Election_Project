#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import statistics
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

#import excel spreadsheets
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

#%% Prediction Model Data/Splitting

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


#setting hyperparamaters for ML models
model_r = ensemble.GradientBoostingRegressor(
        alpha=0.9, 
        criterion='friedman_mse', 
        init=None,
        learning_rate=0.1, 
        loss='ls', 
        max_depth=15,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0, 
        min_impurity_split=None,
        min_samples_leaf=1, 
        min_samples_split=2,
        min_weight_fraction_leaf=0.0, 
        n_estimators=200,
        n_iter_no_change=None, 
        presort='auto',
        random_state=None, 
        subsample=1.0, 
        tol=0.0001,
        validation_fraction=0.1, 
        verbose=0, 
        warm_start=False
)

model_d = ensemble.GradientBoostingRegressor(
        alpha=0.9,
        criterion='friedman_mse', 
        init=None,
        learning_rate=0.05, 
        loss='ls', 
        max_depth=9,
        max_features=None, 
        max_leaf_nodes=None,
        min_impurity_decrease=0.0, 
        min_impurity_split=None,
        min_samples_leaf=1, 
        min_samples_split=2,
        min_weight_fraction_leaf=0.0, 
        n_estimators=300,
        n_iter_no_change=None, 
        presort='auto',
        random_state=None, 
        subsample=1.0, 
        tol=0.0001,
        validation_fraction=0.1, 
        verbose=0, 
        warm_start=False
)

#training the model (republican)
model_r.fit(x_train_all_r, y_train_all_r)

#training the model (democrat)
model_d.fit(x_train_all_d, y_train_all_d)

#predicted training values republican
predict_train_r = model_r.predict(x_train_all_r)

#predicted training values democrat
predict_train_d = model_d.predict(x_train_all_d)

#%% Statistical Evaluation for Republican (Training)

#evaluating training republican
r2_train_r = r2_score(predict_train_r, y_train_all_r)
print('The R-Squared value is: %.2f' %r2_train_r)

mae_train_r = mean_absolute_error(y_train_all_r, model_r.predict(x_train_all_r))
print('The mean absolute error is: %i' %mae_train_r)

std_dev_train_r = statistics.stdev(predict_train_r)
print('The standard deviation is: %.2f' %std_dev_train_r)

mean_train_r = sum(predict_train_r)/len(predict_train_r)
print('The mean predicted votes are: %i' %mean_train_r)

con_int_95_train_r = 1.96*std_dev_train_r
print('The 95th percentile confidence interval is: %.2f' %con_int_95_train_r)

#%% Statistical Evaluation for Democrat (Training)

#evaluating training democrat
r2_train_d = r2_score(predict_train_d, y_train_all_d)
print('The R-Squared value is: %.2f' %r2_train_d)

mae_train_d = mean_absolute_error(y_train_all_d, model_d.predict(x_train_all_d))
print('The mean absolute error is: %i' %mae_train_d)

std_dev_train_d = statistics.stdev(predict_train_d)
print('The standard deviation is: %.2f' %std_dev_train_d)

mean_train_d = sum(predict_train_d)/len(predict_train_d)
print('The mean predicted votes are: %i' %mean_train_d)

con_int_95_train_d = 1.96*std_dev_train_d
print('The 95th percentile confidence interval is: %.2f' %con_int_95_train_d)

#%% Plotting training results

#republican
slp, inters, r, p, stderr = stats.linregress(y_train_all_r, predict_train_r)
plt.title('Republican Training Data Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_train_all_r, predict_train_r)
plt.plot(y_train_all_r, (slp*y_train_all_r+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\Rep_Training_Fit')
plt.close()

#democrat
slp, inters, r, p, stderr = stats.linregress(y_train_all_d, predict_train_d)
plt.title('Democrat Training Data Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_train_all_d, predict_train_d)
plt.plot(y_train_all_d, (slp*y_train_all_d+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\Dem_Training_Fit')
plt.close()

#%% Predicting Test Values

#predicted test values republican
predict_test_r = model_r.predict(x_test_all_r)

#predicted test values democrat
predict_test_d = model_d.predict(x_test_all_d)

#%% Statistical Evaluation for Republican (Test)

#evaluating test republican
r2_test_r = r2_score(predict_test_r, y_test_all_r)
print('The R-Squared value is: %.2f' %r2_test_r)

mae_test_r = mean_absolute_error(y_test_all_r, model_r.predict(x_test_all_r))
print('The mean absolute error is: %i' %mae_test_r)

std_dev_test_r = statistics.stdev(predict_test_r)
print('The standard deviation is: %.2f' %std_dev_test_r)

mean_test_r = sum(predict_test_r)/len(predict_test_r)
print('The mean predicted votes are: %i' %mean_test_r)

con_int_95_test_r = 1.96*std_dev_test_r
print('The 95th percentile confidence interval is: %.2f' %con_int_95_test_r)

#%% Statistical Evaluation for Democrat (Test)

#evaluating test democrat
r2_test_d = r2_score(predict_test_d, y_test_all_d)
print('The R-Squared value is: %.2f' %r2_test_d)

mae_test_d = mean_absolute_error(y_test_all_d, model_r.predict(x_test_all_d))
print('The mean absolute error is: %i' %mae_test_d)

std_dev_test_d = statistics.stdev(predict_test_d)
print('The standard deviation is: %.2f' %std_dev_test_d)

mean_test_d = sum(predict_test_d)/len(predict_test_d)
print('The mean predicted votes are: %i' %mean_test_d)

con_int_95_test_d = 1.96*std_dev_test_d
print('The 95th percentile confidence interval is: %.2f' %con_int_95_test_d)

#%% Plotting test results

#republican
slp, inters, r, p, stderr = stats.linregress(y_test_all_r, predict_test_r)
plt.title('Republican Test Data Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_test_all_r, predict_test_r)
plt.plot(y_test_all_r, (slp*y_test_all_r+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\Rep_Test_Fit')
plt.close()

#democrat
slp, inters, r, p, stderr = stats.linregress(y_test_all_d, predict_test_d)
plt.title('Democrat Test Data Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_test_all_d, predict_test_d)
plt.plot(y_test_all_d, (slp*y_test_all_d+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\Dem_Test_Fit')
plt.close()

#%% 2020 Election Prediction

#import 2020 independent variables
election2020 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2020\Iowa_2020_Data.xlsx')

#removing unneeded columns
del election2020['county']

del election2020['winner']

del election2020['total_other']

del election2020['total_votes']

del election2020['total_absentee']

#remove dependent variables
del election2020['total_rep']

del election2020['total_dem']
 
#2020 dependent variable
dep_2020 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2020\Iowa_2020_Data.xlsx')

#creating arrays for independent and dependent variables
x_2020_r = election2020.values
y_2020_r = dep_2020['total_rep'].values

x_2020_d = election2020.values
y_2020_d = dep_2020['total_dem'].values

#predict the outcome of 2020 election
predict_2020_r = model_r.predict(x_2020_r)

predict_2020_d = model_d.predict(x_2020_d)

#statistical analysis of 2020 election prediction (republican)
r2_r = r2_score(predict_2020_r, y_2020_r)
print('The R-Squared value is: %.2f' %r2_r)

mae_r = mean_absolute_error(y_2020_r, predict_2020_r)
print('The mean absolute error is: %i' %mae_r)

std_dev_r = statistics.stdev(predict_2020_r)
print('The standard deviation is: %.2f' %std_dev_r)

mean_r = sum(predict_2020_r)/len(predict_2020_r)
print('The mean predicted votes are: %i' %mean_r)

con_int_95_r = 1.96*std_dev_r
print('The 95th percentile confidence interval is: %.2f' %con_int_95_r)

#statistical analysis of 2020 election prediction (democrat)
r2_d = r2_score(predict_2020_d, y_2020_d)
print('The R-Squared value is: %.2f' %r2_d)

mae_d = mean_absolute_error(y_2020_d, predict_2020_d)
print('The mean absolute error is: %i' %mae_d)

std_dev_d = statistics.stdev(predict_2020_d)
print('The standard deviation is: %.2f' %std_dev_d)

mean_d = sum(predict_2020_d)/len(predict_2020_d)
print('The mean predicted votes are: %i' %mean_d)

con_int_95_d = 1.96*std_dev_d
print('The 95th percentile confidence interval is: %.2f' %con_int_95_d)

#%% Plotting predicted 2020 election results

#republican
slp, inters, r, p, stderr = stats.linregress(y_2020_r, predict_2020_r)
plt.title('2020 Republican Vote Prediction vs. Actual Results')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_2020_r, predict_2020_r)
plt.plot(y_2020_r, (slp*y_2020_r+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\2020_Rep')
plt.close()

#democrat
slp, inters, r, p, stderr = stats.linregress(y_2020_d, predict_2020_d)
plt.title('2020 Republican Vote Prediction vs. Actual Results')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(y_2020_d, predict_2020_d)
plt.plot(y_2020_d, (slp*y_2020_d+inters))
plt.savefig(r'G:\Python\Project\Maps&Figures\2020_Dem')
plt.close()

#%% Calculate the winners for each county,and exporting the results

#concatenating dataframes
r_2020_df = pd.DataFrame({'Rep_Predict': pd.Series(predict_2020_r)})

d_2020_df = pd.DataFrame({'Dem_Predict': pd.Series(predict_2020_d)})

predict_2020 = pd.concat([r_2020_df, d_2020_df], axis=1)

#create winner column
predict_2020['winner'] = np.nan

#Calculate winner column
predict_2020.winner = 0
predict_2020.winner[predict_2020.Rep_Predict > predict_2020.Dem_Predict] = 1

#exporting to excel spreadsheet
predict_2020.to_excel(r'G:\Python\Project\Data\Iowa\2020\Iowa_2020_Prediction.xlsx')

#actual and predicted vote totals
sum_r = dep_2020['total_rep'].sum()
print('The Republican votes = %i' %sum_r)
predict_sum_r = predict_2020['Rep_Predict'].sum()
print('The predicted Republican votes = %i' %predict_sum_r)

sum_d = dep_2020['total_dem'].sum()
print('The Democrat votes = %i' %sum_d)
predict_sum_d = predict_2020['Dem_Predict'].sum()
print('The predicted Democrat votes = %i' %predict_sum_d)