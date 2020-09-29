import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# Start by importing the data into python to create a data frame (df) #
df = pd.read_stata('C:/Users/mauri/github/Econometrics-741/HW2.dta')

# Let's do exploratory data analysis before we start working with the df.
print(df.shape)  # ( 6718, 8) Number of rows, number of columns'
print(df.info())  # 'Complete summary of data types and entries'
print(df.head(10))  # 'Visualize the first 5 entries of dt'

# Age column data type is in a categorical form thus need to be
# transform to interget format for filter to work.
df = df.astype({'age': 'int32'})
print(df.dtypes)

# Filter data to women who are employed and age between 30 to 39
df2 = df.loc[(df.empstat == 'employed') & (df.age >= 35) & (df.sex == 'female')]

print(df2.shape)  # '(4700, 8) Number of rows and columns'

# Data Visualization
x = df2[['age']]
y = df2[['incwage']]

# Fit x and y into Linear Regression Model
model = LinearRegression()
model.fit(x, y)

print(model.intercept_)
print(model.coef_)

# y_hat = 171886.32905437 + 1349.45725237age

# Reproduce scikit-learn results with linear algebra
N = len(x)
p = len(x.columns) + 1  # plus one because LinearRegression adds an intercept term


X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = x.values

beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y.values
print(beta_hat)

# y_hat = 171886.32905437 + 1349.45725237age

# Compute standard errors of the parameter estimates
y_hat = model.predict(x)
residuals = y.values - y_hat
residual_sum_of_squares = residuals.T @ residuals
sigma_squared_hat = residual_sum_of_squares[0, 0] / (N - p)
var_beta_hat = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * sigma_squared_hat
for p_ in range(p):
    standard_error = var_beta_hat[p_, p_] ** 0.5
    print(f"SE(beta_hat[{p_}]): {standard_error}")

# SE(beta_hat[0]): 25328.832831136446
# SE(beta_hat[1]): 570.3044786642951

# Confirm with statsmodels
ols = sm.OLS(y.values, X_with_intercept)
ols_result = ols.fit()
print(ols_result.summary())

# Chapter 6 Questions
# 1. Continuing with the regressions from above, report results using White’s
# Heteroskedasticity Robust Standard Errors.
ols = sm.RLM(y.values, X_with_intercept)
ols_robust = ols.fit()
print(ols_robust.summary())

# 2 Let’s see how earnings varies with age. Take the log of earnings before
# running these regressions. Now please use both men and women
# who are employed and 35 years older or more.

df3 = df.loc[(df.empstat == 'employed') & (df.age >= 35)]
df3['lnwage'] = np.log(df3.incwage)
df3['age2'] = np.power((df3.age), 2)
df3['age3'] = np.power((df3.age), 3)
print(df3.head(3))

# a) Run a linear regression of earnings on age.
x = df3[['age']]
y = df3[['lnwage']]

# Reproduce scikit-learn results with linear algebra
N = len(x)
p = len(x.columns) + 1  # plus one because LinearRegression adds an intercept term


X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = x.values

ols = sm.OLS(y.values, X_with_intercept)
ols_result_ln = ols.fit()
print(ols_result_ln.summary())

#  (b) Run a quadratic regression of earnings on age. (4 points)
x = df3[['age', 'age2']]
y = df3[['lnwage']]

# Reproduce scikit-learn results with linear algebra
N = len(x)
p = len(x.columns) + 1  # plus one because LinearRegression adds an intercept term


X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = x.values

ols = sm.OLS(y.values, X_with_intercept)
ols_result_ln = ols.fit()
print(ols_result_ln.summary())

#  (c) Run a cubic regression of earnings on age.
x = df3[['age', 'age2', 'age3']]
y = df3[['lnwage']]

# Reproduce scikit-learn results with linear algebra
N = len(x)
p = len(x.columns) + 1  # plus one because LinearRegression adds an intercept term


X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:p] = x.values

ols = sm.OLS(y.values, X_with_intercept)
ols_result_ln = ols.fit()
print(ols_result_ln.summary())