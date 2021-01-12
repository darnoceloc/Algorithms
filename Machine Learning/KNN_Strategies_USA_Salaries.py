# Reference for some code: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

def preprocess(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'Class']
    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Class']
    values = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    items = []
    for i in range(len(lines)):
        line = lines[i].split(', ')
        itemFeatures = {}
        for j in range(len(line)):
            f = features[j]
            # Add feature to dict
            itemFeatures[f] = line[j]
            # Append temp dict to items
        items.append(itemFeatures)
    np.random.shuffle(items)
    df = pd.DataFrame(data=items)
    df = df[~(df.astype(str) == '?').any(1)]
    ax1 = sns.countplot(df['Class'])
    plt.show()
    ax2 = sns.countplot(x='Class', hue=df['sex'], data=df)
    plt.show()
    ax2 = sns.countplot(x='Class', hue=df['race'], data=df)
    plt.show()
    f = plt.figure(figsize=(20, 4))
    f.add_subplot(1, 2, 1)
    sns.distplot(df['age'])
    plt.show()
    f.add_subplot(1, 2, 2)
    sns.boxplot(df['age'].astype(int))
    plt.show()
    df.drop(columns=['fnlwgt', 'relationship', 'capital-gain', 'capital-loss'])
    # def normalize(df):
    #     result = df.copy()
    #     for feature_name in values:
    #         df[feature_name] = df[feature_name].apply(pd.to_numeric, errors='ignore')
    #         max_value = df[feature_name].max()
    #         min_value = df[feature_name].min()
    #         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    #     return result
    def z_score(df):
        df_std = df.copy()
        for column in values:
            df_std[column] = df_std[column].apply(pd.to_numeric, errors='ignore')
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        return df_std
    # df = normalize(df)
    df = z_score(df)
    df = pd.get_dummies(df, columns=columns, drop_first=False)
    return df

df_train = preprocess('adult.data')
df_test = preprocess('adult.test')
df_test['native-country_Holand-Netherlands'] = 0
print(df_train.columns.difference(df_test.columns))
print(df_test.columns.difference(df_train.columns))

knn = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto', leaf_size=5, p=1, metric='minkowski', n_jobs=4)#Create x and y variables.
x_train = df_train.drop(columns=['Class_<=50K'])
y_train = df_train['Class_<=50K']#Split data into training and testing.
x_test = df_test.drop(columns=['Class_<=50K.'])
y_test = df_test['Class_<=50K.']#Split data into training and testing.
knn.fit(x_train, y_train)#Predict test data set.
y_pred = knn.predict(x_test)#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))#Checking performance our model with ROC Score.
print(roc_auc_score(y_test, y_pred))


# evaluate a model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall', 'f1'), cv=cv, n_jobs=8, return_train_score=True, error_score='raise')
    return scores
print(evaluate_model(knn, x_train, y_train))

# List Hyperparameters that we want to tune.
hyperparameters = dict(leaf_size=[20, 200, 1000], n_neighbors=[5, 10, 50, 100, 500, 1000], p=[1, 2])#Create new KNN object
knn_2 = KNeighborsClassifier()#Use GridSearch
clf = GridSearchCV(estimator=knn_2, param_grid=hyperparameters, n_jobs=8, cv=10, return_train_score=True)#Fit the model
best_model = clf.fit(x_train, y_train)#Print The value of best Hyperparameters
print('CV results:', best_model.cv_results_)
print('Best estimator:', best_model.best_estimator_)
print('Best model:', best_model.best_score_)
print('Best parameters:', best_model.best_params_)
y_pred = clf.predict(x_test)#Checking performance our model with classification report.
print(classification_report(y_test, y_pred))#Checking performance our model with ROC Score.
print(roc_auc_score(y_test, y_pred))