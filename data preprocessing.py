import pandas as pd
import seaborn as sns


data = pd.read_csv(r"data.txt",sep='|').dropna(axis=0)

# gdp: Real gross domestic product
# consumption: Real consumption expenditures
# invest: Real investment by private sector
# government: Real government expenditures
# dpi: Real disposable personal income
# cpi: Consumer price index
# m1: Nominal money stock
# tbill: Quarterly average of month end 90 day treasury bill rate
# unemp: Unemployment rate
# population: Population
# inflation: Inflation rate
# interest: Ex-post real interest rate

y = data['consumption'].diff().dropna().reset_index(drop=True)
X=data[['gdp','dpi','cpi','unemp','invest','inflation','interest']].diff().dropna().reset_index(drop=True)
# vedere se tra i regressori c'è correlazione
sns.pairplot(X[['gdp','dpi','cpi','unemp','invest','inflation','interest']], diag_kind='kde')

X.describe().transpose()[['mean', 'std']]

#togli regressori multicollineari con vif < 4. Se tra 5-10 o >10 c'è multcoll
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
# dpi,cpi,unemp,invest
X_true = X[['dpi','cpi','unemp','invest']]

# tolgo outliers
X_true =X_true[(X_true > X_true.quantile(q=.15)) &
               (X_true < X_true.quantile(q=.85))]

#fillna con la media della colonna
X_true.fillna(X_true.mean(axis=0),inplace=True)


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_true =pd.DataFrame(scaler.fit_transform(X=X_true),columns=X_true.columns)
y = pd.DataFrame(scaler.fit_transform(X=y.values.reshape(-1,1)),columns=[y.name])
