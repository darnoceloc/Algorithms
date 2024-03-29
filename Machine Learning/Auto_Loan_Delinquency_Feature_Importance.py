# -*- coding: utf-8 -*-#
"""
File:   Auto_Loan_ML.py
Author: Conrad Cole
Date:   3/15/20
Desc:   Analysis of GM Financial Consumer Automobile Receivables Trust Data Tape
	Prediction of Delinquency via Tree-Based Feature Importance Methods
"""


""" =======================  Import dependencies ========================== """


import numpy as np
import pandas as pd
import os
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


""" ======================  Function definitions ========================== """


def getIndexes(dfObj, value):
    # Empty list
    listOfPos = []
    # isin() method will return a dataframe with boolean values, True at the positions where element exists
    result = dfObj.isin([value])
    # any() method will return a boolean series
    seriesObj = result.any()
    # Get list of columns where element exists
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over the list of columns and extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
            # This list contains a list tuples with
    # the index of element in the dataframe
    return listOfPos


def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # Sort the index values and flip them so that they are arranged in decreasing order of importance
    index_sorted = np.flipud(np.argsort(feature_importances))
    # Center the location of the labels on the X-axis (for display purposes only)
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # Plot the bar graph
    feature_names_ord = [x for _, x in sorted(zip(index_sorted, feature_names))]
    print(feature_names_ord[:10])
    plt.figure()
    plt.bar(pos[:10], feature_importances[index_sorted][:10], align='center')
    plt.xticks(pos[:10], feature_names_ord[:10], fontsize=12, rotation=45)
    plt.xlabel("Feature Names", fontdict={'size': 18}, labelpad=27)
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


def z_score(df_std, norm):
    assert isinstance(df_std, pd.DataFrame)
    for column in norm:
        df_std[column] = pd.to_numeric(df_std[column], errors='coerce', downcast='float')
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std


def clean_dataset(df, num_cols):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    result = df.copy()
    indices_to_keep = ~df[df.columns[~df.columns.isin(num_cols)]].isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[df.columns[~df.columns.isin(num_cols)]].astype(np.float64)
    for numz in num_cols:
        df[numz] = result[numz]
    return df[indices_to_keep]


def cleaner(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df = df.replace(to_replace=['None', '-'], value=np.nan).dropna()
    df = df.replace(to_replace='false', value=0)
    df = df.replace(to_replace=['true'], value=1)
    df = df.replace(to_replace=['1; 2'], value=1)
    df = df.replace(to_replace=['2; 1'], value=2)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def clean_time_series(df, timez):
    assert isinstance(df, pd.DataFrame)
    for col in timez:
        df[col] = (df[col]/np.timedelta64(1, 'D')).astype(np.float64)
        df[col] = df[col][~((df[col] - df[col].mean()).abs() > 3.5 * df[col].std())]
    df = df.dropna()
    return df
    
    
""" ======================  Class definitions ========================== """


class RandomForestExample:
    def __init__(self, path):
        # reading the data from the csv file
        self.path = path

    def getDFS(self, path):
        all_files = glob.iglob(os.path.join(path + "*.csv"))
        return pd.concat((pd.read_csv(f, dtype='unicode') for f in all_files), ignore_index=True)

    def getDF(self, filename):
        return pd.read_csv(filename, dtype='unicode')

    def train(self, model, data_train, labelz):
        model = model.fit(data_train, labelz)
        return model

    def predictLabels(self, model, featurez):
        model = model.predict(featurez)
        return model
	
	
"""======================================================================================="""


path_dfs = ''

rfe = RandomForestExample(path=path_dfs)

df_all = rfe.getDFS(path_dfs).drop(columns=['assetTypeNumber', 'assetNumber', 'reportingPeriodBeginningDate',
            'reportingPeriodEndingDate', 'originatorName', 'underwritingIndicator',
            'obligorCreditScoreType', 'assetAddedIndicator', 'reportingPeriodModificationIndicator',
             'zeroBalanceEffectiveDate', 'primaryLoanServicerName', 'repossessedIndicator', 'modificationTypeCode',
            'assetSubjectDemandStatusCode', 'repurchaseAmount', 'DemandResolutionDate', 'repurchaserName',
            'repurchaseReplacementReasonCode', 'mostRecentServicingTransferReceivedDate',
            'zeroBalanceCode',  'assetSubjectDemandIndicator', 'coObligorIndicator',
            'actualOtherCollectedAmount', 'vehicleValueSourceCode', 'paymentTypeCode', 'vehicleTypeCode',
            'interestCalculationTypeCode', 'originalInterestRateTypeCode', 'servicerAdvancedAmount',
            'servicingFlatFeeAmount', 'servicingAdvanceMethodCode', 'otherServicerFeeRetainedByServicer',
            'paymentExtendedNumber', 'repossessedProceedsAmount', 'chargedoffPrincipalAmount',
            'obligorIncomeVerificationLevelCode', 'obligorEmploymentVerificationCode', 'originalInterestOnlyTermNumber'])

print(df_all[df_all.index.duplicated()])
df_train_dates = df_all
df_train_dates['originationDate'] = pd.to_datetime(df_all['originationDate'], format='%m/%Y', errors='raise')
df_train_dates['loanMaturityDate'] = pd.to_datetime(df_all['loanMaturityDate'], format='%m/%Y', errors='raise')
df_train_dates['originalFirstPaymentDate'] = pd.to_datetime(df_all['originalFirstPaymentDate'], format='%m/%Y', errors='raise')
df_train_dates['interestPaidThroughDate'] = pd.to_datetime(df_all['interestPaidThroughDate'], format='%m-%d-%Y', errors='coerce')

df_train_date_deltas = df_train_dates
df_train_date_deltas['originationDate'] = df_train_dates.originationDate - df_train_dates.originationDate.min()
df_train_date_deltas['loanMaturityDate'] = df_train_dates.loanMaturityDate - df_train_dates.loanMaturityDate.min()
df_train_date_deltas['originalFirstPaymentDate'] = df_train_dates.originalFirstPaymentDate - df_train_dates.originalFirstPaymentDate.min()
df_train_date_deltas['interestPaidThroughDate'] = df_train_dates.interestPaidThroughDate - df_train_dates.interestPaidThroughDate.min()

norm_cols = ['originalLoanAmount', 'originalLoanTerm', 'originalInterestRatePercentage', 'vehicleValueAmount',
        'obligorCreditScore', 'paymentToIncomePercentage', 'remainingTermToMaturityNumber',
        'reportingPeriodBeginningLoanBalanceAmount', 'nextReportingPeriodPaymentAmountDue',
        'otherAssessedUncollectedServicerFeeAmount', 'scheduledInterestAmount', 'scheduledPrincipalAmount',
        'reportingPeriodActualEndBalanceAmount', 'reportingPeriodScheduledPaymentAmount', 'totalActualAmountPaid',
        'actualInterestCollectedAmount', 'actualPrincipalCollectedAmount']

num_colz = ['vehicleManufacturerName', 'vehicleModelName', 'obligorGeographicLocation',
            'originationDate', 'loanMaturityDate', 'originalFirstPaymentDate', 'interestPaidThroughDate']

timez = ['originationDate', 'loanMaturityDate', 'originalFirstPaymentDate', 'interestPaidThroughDate']
cat_cols = ['vehicleManufacturerName', 'vehicleModelName', 'obligorGeographicLocation']

df_train_clean = cleaner(df_train_date_deltas)
df_train_clean = clean_dataset(df_train_clean, num_colz)
df_train_clean = clean_time_series(df_train_clean, timez)

df_train_wocats = df_train_clean.drop(columns=cat_cols)

df_train_cats = pd.get_dummies(df_train_clean, columns=cat_cols, drop_first=False, dtype=float).dropna()
df_train_cats = df_train_cats.replace(to_replace=['None', '-'], value=np.nan).dropna()

df_train_norm = cleaner(df_train_cats)
df_train_norm_wocats = cleaner(df_train_wocats)
for colz in norm_cols:
    df_train_norm[colz] = pd.to_numeric(df_train_norm[colz], errors='coerce', downcast='float')
    df_train_norm_wocats[colz] = pd.to_numeric(df_train_norm_wocats[colz], errors='coerce', downcast='float')
    df_train_norm[colz] = df_train_norm[colz][~((df_train_norm[colz] -
            df_train_norm[colz].mean()).abs() > 3.5*df_train_norm[colz].std())]
    df_train_norm_wocats[colz] = df_train_norm_wocats[colz][~((df_train_norm_wocats[colz] -
            df_train_norm_wocats[colz].mean()).abs() > 3.5*df_train_norm_wocats[colz].std())]
# df_train_norm = z_score(df_train_norm, norm_cols)
# df_train_norm_wocats = z_score(df_train_norm_wocats, norm_cols)
df_train_norm = cleaner(df_train_norm)
df_train_norm_wocats = cleaner(df_train_norm_wocats)

sns.set(font_scale=1.4)

sns.boxplot(x="subvented", y="obligorCreditScore", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="subvented", y="currentDelinquencyStatus", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="subvented", y="paymentToIncomePercentage", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="subvented", y="originalLoanAmount", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="subvented", y="vehicleValueAmount", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="subvented", y="originalInterestRatePercentage", data=df_train_norm_wocats)
plt.show()

sns.histplot(data=df_train_norm_wocats, x='obligorCreditScore', stat="probability")
plt.show()

listOfElems = ['1; 2', '-', 'false']
dictOfPos = {elem: getIndexes(df_train_norm, elem) for elem in listOfElems}
print('Position of given elements in Dataframe are : ')
# Looping through key, value pairs in the dictionary
for key, value in dictOfPos.items():
    print(key, ' : ', value)

labels = df_train_norm['currentDelinquencyStatus']
labels_wocats = df_train_norm_wocats['currentDelinquencyStatus']
labels = labels.replace(to_replace=['None', '-'], value=np.nan).dropna()

arr = np.array([0, 30, 60, 90, 120])
labels_wocats = np.searchsorted(arr, labels_wocats.values, side='right')
df_train_norm_wocats['currentDelinquencyStatus'] = labels_wocats

sns.boxplot(x="currentDelinquencyStatus", y="obligorCreditScore", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="currentDelinquencyStatus", y="paymentToIncomePercentage", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="currentDelinquencyStatus", y="originalLoanAmount", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="currentDelinquencyStatus", y="vehicleValueAmount", data=df_train_norm_wocats)
plt.show()
sns.boxplot(x="currentDelinquencyStatus", y="originalInterestRatePercentage", data=df_train_norm_wocats)
plt.show()

df_train_norm = df_train_norm.drop(columns='currentDelinquencyStatus')
df_train_norm_wocats = df_train_norm_wocats.drop(columns='currentDelinquencyStatus')

features = df_train_norm.columns
print(len(features))
features_wocats = df_train_norm_wocats.columns
print(len(features_wocats))

#Apply Classifier and Make Prediction
X_train, X_test, y_train, y_test = train_test_split(df_train_norm_wocats, labels_wocats, test_size=0.25, random_state=100)
dt_classifier = DecisionTreeClassifier(max_depth=20)
dt_classifier.fit(X_train, y_train)
cv_score_dt_train = cross_val_score(dt_classifier, X_train, y_train, cv=10)
ab_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), n_estimators=40, random_state=7)
ab_classifier.fit(X_train, y_train)
cv_score_ab_train = cross_val_score(ab_classifier, X_train, y_train, cv=10)

# dt_regressor = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10000, max_features=15)
# dt_trained = dt_regressor.fit(X_train, y_train)
# cv_score_dt_train = cross_val_score(dt_trained, X_train, y_train, cv=10)
# rsqared_dt_train = dt_trained.score(X_train, y_train)
#
# ab_regressor = AdaBoostRegressor(dt_regressor, n_estimators=40, random_state=7)
# ab_trained = ab_regressor.fit(X_train, y_train)
# cv_score_ab_train = cross_val_score(ab_trained, X_train, y_train, cv=10)
# rsquared_ab_train = ab_trained.score(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)
cv_score_dt_test = cross_val_score(dt_classifier, X_test, y_test, cv=10)
mse = sm.mean_squared_error(y_test, y_pred_dt)
evs = sm.explained_variance_score(y_test, y_pred_dt)
print("\n#### Decision Tree performance ####")
print("Train Cross val score =", cv_score_dt_train)
print("Test Cross val score =", cv_score_dt_test)
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("Accuracy:", sm.accuracy_score(y_test, y_pred_dt))
print("Classification Report:", classification_report(y_test, y_pred_dt))

y_pred_ab = ab_classifier.predict(X_test)
cv_score_ab_test = cross_val_score(ab_classifier, X_test, y_test, cv=10)
mse = sm.mean_squared_error(y_test, y_pred_ab)
evs = sm.explained_variance_score(y_test, y_pred_ab)
print("\n#### AdaBoost performance ####")
print("Train Cross val score =", cv_score_ab_train)
print("Test Cross val score =", cv_score_ab_test)
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("Accuracy:", sm.accuracy_score(y_test, y_pred_ab))
print("Classification Report:", classification_report(y_test, y_pred_ab))

# y_pred_dt = dt_trained.predict(X_test)
# rsqared_dt_test = dt_trained.score(X_test, y_test)
# cv_score_dt_test = cross_val_score(dt_trained, X_test, y_test, cv=10)
# mse = sm.mean_squared_error(y_test, y_pred_dt)
# mae = sm.mean_absolute_error(y_test, y_pred_dt)
# evs = sm.explained_variance_score(y_test, y_pred_dt)
# print("\n#### Decision Tree performance ####")
# print("Train R squared =", rsqared_dt_train)
# print("Train Cross val score =", cv_score_dt_train)
# print("Test R squared =", rsqared_dt_test)
# print("Test Cross val score =", cv_score_dt_test)
# print("Mean squared error =", round(mse, 2))
# print("Mean absolute error =", round(mae, 2))
# print("Explained variance score =", round(evs, 2))
#
# y_pred_ab = ab_trained.predict(X_test)
# rsqared_ab_test = ab_trained.score(X_test, y_test)
# cv_score_ab_test = cross_val_score(ab_trained, X_test, y_test, cv=10)
# mse = sm.mean_squared_error(y_test, y_pred_ab)
# mae = sm.mean_absolute_error(y_test, y_pred_ab)
# evs = sm.explained_variance_score(y_test, y_pred_ab)
# print("\n#### AdaBoost performance ####")
# print("Train R squared =", rsquared_ab_train)
# print("Train Cross val score =", cv_score_ab_train)
# print("Test R squared =", rsqared_ab_test)
# print("Test Cross val score =", cv_score_ab_test)
# print("Mean squared error =", round(mse, 2))
# print("Mean absolute error =", round(mae, 2))
# print("Explained variance score =", round(evs, 2))

random_forest_classifier = RandomForestClassifier(n_estimators=100, max_depth=20,
                                random_state=111, max_features='auto')
modelRF = rfe.train(random_forest_classifier, X_train, y_train)
cv_score_rf_train = cross_val_score(modelRF, X_train, y_train, cv=10)
pred_rf = rfe.predictLabels(modelRF, X_test)
predicted_scores = accuracy_score(y_test, pred_rf)
mse = sm.mean_squared_error(y_test, pred_rf)
evs = sm.explained_variance_score(y_test, pred_rf)
cv_score_rf_test = cross_val_score(modelRF, X_test, y_test, cv=10)

# random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=111, max_features=15)
# modelRF = rfe.train(random_forest_regressor, X_train, y_train)
# rsqared_rf_train = modelRF.score(X_train, y_train)
# cv_score_rf_train = cross_val_score(modelRF, X_train, y_train, cv=10)
# pred_rf = rfe.predictLabels(modelRF, X_test)
# rsqared_rf_test = modelRF.score(X_test, y_test)
# cv_score_rf_test = cross_val_score(modelRF, X_test, y_test, cv=10)
# mse = sm.mean_squared_error(y_test, pred_rf)
# mae = sm.mean_absolute_error(y_test, pred_rf)
# evs = sm.explained_variance_score(y_test, pred_rf)

print("\n#### Random Forest performance ####")
print("Train Cross val score =", cv_score_rf_train)
print("Test Cross val score =", cv_score_rf_test)
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("Accuracy:", sm.accuracy_score(y_test, pred_rf))
print("Classification Report:", classification_report(y_test, pred_rf))
# print("Train R squared =", rsqared_rf_train)
# print("Train Cross val score =", cv_score_rf_train)
# print("Test R squared =", rsqared_rf_test)
# print("Test Cross val score =", cv_score_rf_test)
# print("Mean squared error =", round(mse, 2))
# print("Mean absolute error =", round(mae, 2))
# print("Explained variance score =", round(evs, 2))

plot_feature_importances(dt_classifier.feature_importances_, 'Decision Tree Regressor', features_wocats)
plot_feature_importances(ab_classifier.feature_importances_, 'AdaBoost Regressor', features_wocats)
plot_feature_importances(modelRF.feature_importances_, 'Random Forest Regressor', features_wocats)

result = permutation_importance(modelRF, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
perm_sorted_idx = np.flipud(result.importances_mean.argsort())

tree_importance_sorted_idx = np.flipud(np.argsort(modelRF.feature_importances_))
tree_indices = np.arange(0, len(modelRF.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices[:20],  modelRF.feature_importances_[tree_importance_sorted_idx][:20], height=0.7)
ax1.set_yticklabels(features_wocats[tree_importance_sorted_idx][:20])
ax1.set_yticks(tree_indices[:20])
ax1.set_ylim((0, len(random_forest_classifier.feature_importances_[:20])))
ax2.boxplot(result.importances[perm_sorted_idx[:20]].T, vert=False, labels=features_wocats[perm_sorted_idx][:20])
ax1.title.set_text('Random Forest Feature Importance')
ax2.title.set_text('Permutation Feature Importance')
fig.tight_layout()
plt.legend()
plt.show()

print(features_wocats[tree_importance_sorted_idx][:10])
print(features_wocats[perm_sorted_idx][:10])
