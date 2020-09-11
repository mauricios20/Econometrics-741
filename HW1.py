import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Start by importing the data into python to create a data frame (df) #
df = pd.read_stata('C:/Users/mauri/github/Econometrics-741/741_data_1.dta')

# Let's do exploratory data analysis before we start working with the df.
print(df.shape)  # '(463090, 5) Number of rows, number of columns'.
print(df.info())  # 'Complete summary of data types and entries'
print(df.head(10))  # 'Visualize the first 5 entries of dt'

# Age column data type is in a categorical form thus need to be
# transform to interget format for filter to work.
df = df.astype({'age': 'int32'})
print(df.dtypes)

# Filter data to women who are employed and age between 30 to 39
df2 = df.loc[(df.empstat == 'employed') & (df.age >= 30) & (df.age <= 39)]

print(df2.shape)  # '(14517, 5) Number of rows and columns'

# (a) Find the avergae income of these population set
muinc = df2['incwage'].mean()  # '$48,348.34 average income'
print(muinc.round(2))

# (b) Take a random sample of 1000 women, random state is used to ensure the
# code chooses the same random samples everytime it runs.
df3 = df2.sample(n=1000, random_state=23)
# (i) Sample mean
smu = df3.incwage.mean()  # '$47,355.63 average income sample'
print(smu.round(2))

# (ii) Sample Standard Deviation
sigma = df3.incwage.std()  # '$39,169.62'
print(sigma.round(2))

# (iii) Standard Error
SE = sigma/np.sqrt(1000)  # '1238.65'
print(SE.round(2))

# (iv) Give the 95% confidence interval on your estimate of μ.
dgf = 1000-1
alpha = (1-.95)/2  # '0.025'

t = 1.962  # 'From t distribution table'
t = stats.t.ppf(1-0.025, 999)  # 'From stats package which is same as table'

CI95 = SE*t  # '+/- 2427.06661315927'
# Lower end of the range
print((smu-CI95).round(2))  # '44924.97'

# Upper end of the range
print((smu+CI95).round(2))  # '49786.29'

# (v) Ho m0 = 50,000 ; H1 m1 =! 50,000
ttest = (smu - 50000)/SE  # '2.13 which is greater than t, thus reject null'
# alpha between 0.025 & 0.01
print(ttest.round(2))
# (vi) Ho m = 55,000 ; H1 m =! 50000
ttest = (smu - 55000)/SE  # '6.17 thus reject Ho and alpha is very small'
print(ttest.round(2))


# (c) Now  take  10000  random  samples  of  1000  women.
# Record  thesample mean for each sample and save it to a matrix.
xbars = []
for i in range(0, 10001):
    xbar = df2.incwage.sample(n=1000).mean()
    xbars.append(xbar)

arr = np.asarray(xbars)

# (i) Produce  a  graph  of  the  distribution  of  these  10,000  samplemeans
plt.plot(arr)
plt.title('Random Samples')
plt.axhline(y=muinc, color='r', linestyle='-')  # 'population set mean'
plt.ylabel('Sample Mean for each sample')
plt.xlabel('Number of samples')
plt.show()

# (ii) Report the average of these 10,000 sample means.
ave = arr.mean()  # '$48329.9'
print(ave.round(2))
# (iii) Report the standard deviation of these 10,000 sample means.
sig = arr.std()  # '1470.4'
print(sig.round(2))
