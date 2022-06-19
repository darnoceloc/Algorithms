"""
File:   Door_Dash_DS_Project.py
Author: Conrad Cole
Date:   11/06/21
Desc:   Predicting duration of Door Dash order deliveries
"""

""" =======================  Import dependencies ========================== """

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.inspection import permutation_importance
import joblib
import optuna
from functools import partial


""" ======================  Function definitions ========================== """


class DataFrameImputer(TransformerMixin):
    def __init__(self, num_colz):
        self.num_colz = num_colz
        self.scalerx = StandardScaler()
        self.scalery = StandardScaler()
        self.fill_cat = pd.Series(dtype=object)
        self.fill_num = pd.Series(dtype=float)

    def fit(self, X, y=None):
        self.fill_num = pd.Series([X[c].median() for c in self.num_colz])
        self.scalerx = StandardScaler().fit(X[self.num_colz].values)
        if y is None:
            return self
        if y.any():
            self.scalery = StandardScaler().fit(np.array(y).reshape(-1, 1))
            return self

    def transform(self, X, y=None, cols=None):
        if cols is not None:
            self.num_colz = [col for col in X.columns if col in self.num_colz]
        X = X.fillna(self.fill_num)[self.num_colz]
        feats = self.scalerx.transform(X[self.num_colz].values)
        X[self.num_colz] = feats
        if y is None:
            return X
        if y.any():
            y = self.scalery.transform(np.array(y).reshape(-1, 1))
            return X, y


def run(trial, X_train, y_train, X_valid, y_valid):
    n_estimators = trial.suggest_int("n_estimators", 7000, 20000, 500)
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 5, 15)

    model = xgb.XGBRegressor(
        random_state=42,
        tree_method="auto",
        n_estimators=n_estimators,
        predictor="auto",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        n_jobs=-1
    )
    model.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[(X_valid, y_valid)], verbose=1000)
    preds_valid = model.predict(X_valid)
    rmse = sm.mean_squared_error(y_valid, preds_valid, squared=False)
    return rmse


"""======================================================================================="""

# import raw training data from csv file and view summaries
data = pd.read_csv('historical_data.csv',
                   na_values='NA')
print(data.dtypes)
print(data.describe())
print(data.info())

# get number of unique elements in each column/feature
nunique = data.nunique()
print(nunique)

# Add new features that may intuitively increase predictive performance
data['total_idle_dashers'] = data['total_onshift_dashers'] - data['total_busy_dashers']
data['onshift_dashers_per_outstanding_orders'] = data['total_onshift_dashers']/data['total_outstanding_orders']
data['busy_dashers_per_outstanding_orders'] = data['total_busy_dashers']/data['total_outstanding_orders']
data['idle_dashers_per_outstanding_orders'] = data['total_idle_dashers']/data['total_outstanding_orders']

# Preprocess date time features and add new features that may be relevant
data['created_at'] = pd.to_datetime(data['created_at'])
data['created_at_day'] = data['created_at'].dt.day
data['created_at_hour'] = data['created_at'].dt.hour
data['created_at_minute'] = data['created_at'].dt.minute

data['actual_delivery_time'] = pd.to_datetime(data['actual_delivery_time'])
data['actual_delivery_time_day'] = data['actual_delivery_time'].dt.day
data['actual_delivery_time_hour'] = data['actual_delivery_time'].dt.hour
data['actual_delivery_time_minute'] = data['actual_delivery_time'].dt.minute

data['actual_delivery_time'] = (data['actual_delivery_time'] - data['created_at'].min())
data['created_at'] = (data['created_at'] - data['created_at'].min())

data['duration'] = (data['actual_delivery_time'] - data['created_at'])

data['created_at'] = data['created_at'] / np.timedelta64(1, 's')
data['actual_delivery_time'] = data['actual_delivery_time'] / np.timedelta64(1, 's')
data['duration'] = data['duration'] / np.timedelta64(1, 's')

cols = [col for col in data.columns]

# Drop infinite and other troublesome elements
data = data.replace([np.inf, -np.inf, 'None'], np.nan)
data.dropna(axis=0, inplace=True)
print(data.info())
print(data.isnull().sum())

# Visualize data and obtain correlations between features and target
print(data.corrwith(data['duration']))
data.hist()
plt.show()

# Partition and scale data via standardization
X = data.drop(columns='duration')
y = data['duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
imputer = DataFrameImputer(X.columns)
X_train = imputer.fit_transform(X=X_train)
X_test = imputer.transform(X=X_test)

# Utilize optuna to tune hyperparameters
print("train_start_xgb_reg", dt.datetime.now())
opt_fun = partial(
    run,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_test,
    y_valid=y_test
)

study = optuna.create_study(direction="minimize")
study.optimize(opt_fun, n_trials=10)
print(study.best_value, study.best_params)
print("train_end_xgb_reg", dt.datetime.now())

# Train optimized model and generate performance stats
xgb_rg = xgb.XGBRegressor(random_state=42, tree_method="auto", n_estimators=study.best_params['n_estimators'],
                          predictor="auto", learning_rate=study.best_params['learning_rate'],
                          reg_lambda=study.best_params['reg_lambda'], reg_alpha=study.best_params['reg_alpha'],
                          subsample=study.best_params['subsample'], colsample_bytree=study.best_params['colsample_bytree'],
                          max_depth=study.best_params['max_depth'], n_jobs=-1)
xgb_rg.fit(X_train, y_train, verbose=True)
joblib.dump(xgb_rg, "model.pkl")
# xgb_rg = joblib.load("model.pkl")
cv_score_xgb_train = cross_val_score(xgb_rg, X_train, y_train, cv=10)
rsqared_xg_tr = xgb_rg.score(X_train, y_train)
pred_xgb_rg = xgb_rg.predict(X_test)
pd.DataFrame(pred_xgb_rg, columns=['predicted_duration']).to_csv('valid_pred.csv')
rsqared_xg_te = xgb_rg.score(X_test, y_test)
mse = sm.mean_squared_error(y_test, pred_xgb_rg)
mae = sm.mean_absolute_error(y_test, pred_xgb_rg)
evs = sm.explained_variance_score(y_test, pred_xgb_rg)
cv_score_xgb_test = cross_val_score(xgb_rg, X_test, y_test, cv=10)

print("\n#### XGBoost Reg. performance ####")
print("Train R squared =", rsqared_xg_tr)
print("cv_score_train", cv_score_xgb_train)
print("RMSE", np.sqrt(mse))
print("Test R squared =", rsqared_xg_te)
print("explained variance", evs)
print("cv_score_test", cv_score_xgb_test)
print("Mean absolute error =", mae)

# Visualize Order of Feature Importances
feat_importance_rg = xgb_rg.feature_importances_
sorted_idx_1 = feat_importance_rg.argsort()

perm_importance_rg = permutation_importance(xgb_rg, X_train, y_train)
sorted_idx_2 = perm_importance_rg.importances_mean.argsort()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.barh(X_train.columns[sorted_idx_1][10:], feat_importance_rg[sorted_idx_1][10:])
ax1.title.set_text("XGB Reg Feature Importance")
ax2.barh(X_train.columns[sorted_idx_2][10:], perm_importance_rg.importances_mean[sorted_idx_2][10:])
ax2.title.set_text("XGB Reg Permutation Importance")
fig.tight_layout()
plt.legend()
plt.show()

df_pred = pd.read_csv('/home/darnoc/Documents/Machine Learning/Corporate_Challenges/predict_data.csv',
                   na_values='NA')
print(df_pred.dtypes)
print(df_pred.describe())
print(df_pred.info())
print(df_pred.isnull().sum())

cols = [col for col in df_pred.columns]
df_pred['created_at'] = pd.to_datetime(df_pred['created_at']) - pd.to_datetime(df_pred['created_at'].min())
df_pred['created_at'] = df_pred['created_at'] / np.timedelta64(1, 's')

missing_cols = set(X_train.columns) - set(df_pred.columns)
print(missing_cols)
for c in missing_cols:
    df_pred[c] = 0
df_pred = df_pred[X_train.columns]
X_pred = imputer.transform(X=df_pred, cols=df_pred.columns)
pred_xgb_rg = xgb_rg.predict(X_pred)
pd.DataFrame(pred_xgb_rg, columns=['predicted_duration']).to_csv('data_to_predict.csv')
