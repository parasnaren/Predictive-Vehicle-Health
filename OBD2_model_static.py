#897 rows
import pandas as pd
import numpy as np
import json
import re
from matplotlib import pyplot as plt

##### Cleaning data and merging into dataset
url = 'https://raw.githubusercontent.com/parasnaren/sensapp/master/net.modelbased.sensapp.data.samples/OBD_II/Car1/json/split/factorized/'
urls = ['altitude.json','CO2.json','coolant.json','fuel.json', 'GPS_speed.json', 'lat.json','lon.json','OBD_speed.json','RPM.json']
names = ['Altitude','CO2','Coolant','Litre per 100km(Instant)','GPS Speed','Latitude','Longitude','OBD Speed','RPM']
df_final = pd.DataFrame()
for i, name in zip(urls, names):
    val = []
    ext = url + i
    df = pd.read_json(ext, orient='columns')
    df = pd.DataFrame(df)
    for line in df['e']:
        line = str(line)
        line = re.sub(r"^\s'.+\s", "", line.split(',')[1])
        line = float(re.sub(r"}","", line))
        val.append(line)
    df_final[name] = val
#df_final.to_csv('OBD_2_dataset.csv')
    
##### Importing the dataset
df = pd.read_csv('OBD_2_dataset.csv')
df = df.drop('Unnamed: 0', axis=1)
df.corr()
df['Speed'] = (df['GPS Speed'] + df['OBD Speed']) / 2
df = df.drop(['GPS Speed','OBD Speed'], axis=1)

y = df.iloc[:, [3]].values
X = df.drop('Litre per 100km(Instant)', axis=1).values

##### OLS Summary
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((len(X), 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.model.data.xnames = ['const','Altitude','CO2','Coolant','Latitude', 'Longitude','RPM','Speed']
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.model.data.xnames = ['const','Altitude','CO2','Coolant','Latitude','RPM','Speed']
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 6, 7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.model.data.xnames = ['const','CO2','Coolant','RPM','Speed']
regressor_OLS.summary()

# Columns chosen are C02, Coolant, RPM, Speed
df = df.drop(['Latitude','Longitude','Altitude'], axis=1)
X = df.drop('Litre per 100km(Instant)',axis=1).values

##### Splitting data into test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 6)

"""from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)"""
"""from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)"""


##### Plotting relationships
import seaborn as sns
sns.pairplot(df)

plt.scatter(X.iloc[:, [3]], y)
plt.show()

##### Scaling
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

from sklearn.ensemble import GradientBoostingRegressor as GBR
reg = GBR()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

reg.score(X_test, y_test)

plt.scatter(y_pred, y_test)
plt.xlabel('Predicted')
plt.ylabel('True values')
plt.show()

y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(y_test)

total = y_pred - y_test
print('Mean = ', np.mean(total), '\n', 'STD = ', np.std(total))



#####
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()

################ Using ridge fitting 
"""from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
X_F1_poly = poly.fit_transform(X_new.as_matrix())
linreg = Ridge().fit(X_train, y_train)

print('(poly deg 3 + ridge) linear model coeff (w):\n{}'.format(linreg.coef_))
print('(poly deg 3 + ridge) linear model intercept (b): {:.3f}'.format(linreg.intercept_))
print('(poly deg 3 + ridge) R-squared score (training): {:.3f}'.format(linreg.score(X_train, y_train)))
print('(poly deg 3 + ridge) R-squared score (test): {:.3f}'.format(linreg.score(X_test, y_test)))

predicted=linreg.predict(X_test)

linreg.score(X_test, y_test)"""