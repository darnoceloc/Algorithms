import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

data = pd.read_excel("prices.xlsx")
print(data)
print(data.describe())
data['Average Price'] = data['Average Price'].astype(np.float64)

print(data.info())
nunique = data.nunique()
print(nunique)
print(data.isnull().sum())

sns.histplot(data['Average Price'], kde=True)
plt.show()
sns.histplot(data['Median Household Income'], binwidth=5000, kde=True)
plt.show()

sample = data.sample(n=206, random_state=7)
print(sample)
sample.to_csv('prices_sample.csv')
sns.histplot(sample['Average Price'], kde=True)
plt.legend(title='Average Price Distribution', loc='upper center', title_fontsize=20)
plt.show()

print(sample.describe())

slope, intercept, r_value, p_value, std_err = stats.linregress(data['Median Household Income'], data['Average Price'])
print(slope, intercept, r_value, p_value, std_err)
line = (slope*data['Median Household Income']) + intercept
plt.scatter(data['Median Household Income'], data['Average Price'])
plt.plot(np.unique(data['Median Household Income']), np.poly1d(np.polyfit(data['Median Household Income'],
                        data['Average Price'], 1))(np.unique(data['Median Household Income'])), c='r')
plt.plot(data['Median Household Income'], line, 'r', label='y={:.2f}x+{:.2f}'.format(slope, intercept))
plt.title("Average Price vs. Median Income", y=1.02, fontsize=22)
plt.xlabel("Median Income (USD)", fontsize=15)
plt.ylabel("Average Price (Millions of USD)", fontsize=15)
plt.legend(fontsize=9)
plt.show()


model = sm.OLS(data['Average Price'], sm.add_constant(data['Median Household Income'])).fit()
print(model.summary())
print(model.params)

gls_model = sm.GLS(data['Average Price'], data['Median Household Income'])
gls_results = gls_model.fit()
print(gls_results.summary())
print(gls_results.params)

model = sm.OLS(sample['Average Price'], sm.add_constant(sample['Median Household Income'])).fit()
print(model.summary())
print(model.params)

gls_model = sm.GLS(sample['Average Price'], sample['Median Household Income'])
gls_results = gls_model.fit()
print(gls_results.summary())
print(gls_results.params)

cf = stats.t.interval(alpha=0.95, df=len(sample['Average Price'])-1, loc=np.mean(sample['Average Price']),
                 scale=stats.sem(sample['Average Price']))

print(cf)

cf_norm = stats.norm.interval(alpha=0.95, loc=np.mean(sample['Average Price']), scale=stats.sem(sample['Average Price']))
print(cf_norm)
