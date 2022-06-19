import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import numpy as np
from sklearn import decomposition
import datetime as dt
import os

os.chdir("")
cwd = os.getcwd()
# import calendar and obtain a vector "days" that counts the day to each cashflow.
Calendar = pd.read_excel(cwd+'/SwissGovYields.xls', header=2, usecols="A:I", nrows=120, keep_default_na=False, parse_dates=True)
Calendar[Calendar.select_dtypes(include=['number']).columns] *= 0.01
Calendar.rename(columns={'Unnamed: 0': 'realizations'}, inplace=True)
Calendar.dropna()
months = Calendar['realizations'][1:].dropna()
print(Calendar.iloc[0:1, 1:].values)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.plot(Calendar["realizations"], 100*Calendar['2 years'], c='b', marker='o', label='2 years')
ax.plot(Calendar["realizations"], 100*Calendar['3 years'], c='r', lw=1,  marker='*',  markersize=5, label='3 years')
ax.plot(Calendar["realizations"], 100*Calendar['4 years'], c='g', lw=1,  marker='^', markersize=5, label='4 years')
ax.plot(Calendar["realizations"], 100*Calendar['5 years'], c='m', lw=1,  marker='D', markersize=7, label='5 years')
ax.plot(Calendar["realizations"], 100*Calendar['7 years'], c='k', lw=1,  marker='o', markersize=5, label='7 years')
ax.plot(Calendar["realizations"], 100*Calendar['10 years'], c='C1', lw=1,  marker='s', markersize=5, label='10 years')
ax.plot(Calendar["realizations"], 100*Calendar['20 years'], c='c', lw=1,  marker='p', markersize=5, label='20 years')
ax.plot(Calendar["realizations"], 100*Calendar['30 years'], c='y', lw=1,  marker='+', markersize=5, label='30 years')
plt.title("Swiss Gov. Yield Curves", y=1.02, fontsize=22)
plt.xlabel("Real Time")
plt.ylabel("Yield (%)")
ax.legend(loc=0)
plt.show()

Calendar_diff = Calendar.diff().dropna()
X = Calendar_diff.T
print(X)
print(np.exp(X.iloc[1:2, 0:1].values.astype(float)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.plot(months, 100*X.iloc[1:2].T, c='b', marker='o', label='2 years')
ax.plot(months, 100*X.iloc[2:3].T, c='r', lw=1,  marker='*',  markersize=5, label='3 years')
ax.plot(months, 100*X.iloc[3:4].T, c='g', lw=1,  marker='^', markersize=5, label='4 years')
ax.plot(months, 100*X.iloc[4:5].T, c='m', lw=1,  marker='D', markersize=7, label='5 years')
ax.plot(months, 100*X.iloc[5:6].T, c='k', lw=1,  marker='o', markersize=5, label='7 years')
ax.plot(months, 100*X.iloc[6:7].T, c='C1', lw=1,  marker='s', markersize=5, label='10 years')
ax.plot(months, 100*X.iloc[7:8].T, c='c', lw=1,  marker='p', markersize=5, label='20 years')
ax.plot(months, 100*X.iloc[8:9].T, c='y', lw=1,  marker='+', markersize=5, label='30 years')
plt.title("Monthly Changes in Yield Curve (Stationary)", y=1.02, fontsize=22)
plt.xlabel("Real Time")
plt.ylabel("Loading")
ax.legend(loc=0)
plt.show()

print(np.mean(X.iloc[1:2, :].values))
print(np.mean(X.iloc[1:, 0:1].values))

mus = np.zeros([X.shape[1], 1])
stdev = np.zeros([X.shape[1], 1])

i = 1
while i < X.shape[1]:
    mus[i-1] = np.mean(X.iloc[1:, i-1:i].values)
    stdev[i-1] = np.std(X.iloc[1:, i-1:i].values)
    i = i+1

X_cent = np.zeros([X.shape[0]-1, X.shape[1]])
k = 1
while k < X.shape[1]:
    print(mus[k-1])
    print(X.iloc[1:9, k-1:k].values.astype(float) - mus[k-1])
    X_cent[:, k-1:k] = X.iloc[1:9, k-1:k].values.astype(float) - mus[k-1]
    k = k + 1
print(X_cent)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.plot(months, 100*X_cent[0].T, c='b', marker='o', label='2 years')
ax.plot(months, 100*X_cent[1].T, c='r', lw=1,  marker='*',  markersize=5, label='3 years')
ax.plot(months, 100*X_cent[2].T, c='g', lw=1,  marker='^', markersize=5, label='4 years')
ax.plot(months, 100*X_cent[3].T, c='m', lw=1,  marker='D', markersize=7, label='5 years')
ax.plot(months, 100*X_cent[4].T, c='k', lw=1,  marker='o', markersize=5, label='7 years')
ax.plot(months, 100*X_cent[5].T, c='C1', lw=1,  marker='s', markersize=5, label='10 years')
ax.plot(months, 100*X_cent[6].T, c='c', lw=1,  marker='p', markersize=5, label='20 years')
ax.plot(months, 100*X_cent[7].T, c='y', lw=1,  marker='+', markersize=5, label='30 years')
plt.title("Monthly Changes in Yield Curve (Mean Centered)", y=1.02, fontsize=22)
plt.xlabel("Real Time")
plt.ylabel("Loading")
ax.legend(loc=0)
plt.show()

Q = np.cov(X_cent.T, bias=True)
L, A = np.linalg.eig(Q)
print("numpy eigenvectors", A)
print("numpy eigenvalues", L)

empir_cov = np.zeros([X.shape[1], X.shape[1]])
for m in range(X.shape[1]):
    for n in range(X.shape[1]):
        empir_cov[m][n] = (X_cent[:, m].T@X_cent[:, n])/8
print("empircov1", empir_cov)
Lemp, Aemp = np.linalg.eig(empir_cov)
print("eigenvectors", Aemp)
print("eigenvalues", Lemp)
fract = Lemp/np.sum(Lemp)

tot = sum(Lemp)
var_exp = [(i/tot) for i in sorted(Lemp, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1, 9), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1, 9), cum_var_exp, alpha=0.5, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.plot(Calendar.columns[1:], Aemp[0].T, c='b', marker='o', label='PC1')
ax.plot(Calendar.columns[1:], Aemp[1].T, c='r', lw=1,  marker='*',  markersize=7, label='PC2')
ax.plot(Calendar.columns[1:], Aemp[2].T, c='g', lw=1,  marker='^', markersize=7, label='PC3')
ax.plot(Calendar.columns[1:], Aemp[3].T, c='m', marker='p', markersize=7, label='PC4')
ax.plot(Calendar.columns[1:], Aemp[4].T, c='c', lw=1,  marker='D',  markersize=7, label='PC5')
ax.plot(Calendar.columns[1:], Aemp[5].T, c='y', lw=1,  marker='>', markersize=7, label='PC6')
ax.plot(Calendar.columns[1:], Aemp[6].T, c='C1', lw=1,  marker='o',  markersize=7, label='PC7')
ax.plot(Calendar.columns[1:], Aemp[7].T, c='k', lw=1,  marker='+', markersize=7, label='PC8')
plt.title("Variance between Samples along Principal Components", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity")
plt.ylabel("Loading")
ax.legend(loc=0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.plot(Calendar.columns[1:], Aemp[:, 0].T, c='b', marker='o', label='Level PC1')
ax.plot(Calendar.columns[1:], Aemp[:, 1].T, c='r', lw=1,  marker='*',  markersize=10, label='Slope PC2')
ax.plot(Calendar.columns[1:], Aemp[:, 2].T, c='g', lw=1,  marker='^', markersize=10, label='Curvature PC3')
plt.title("Yield Curve Loadings", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity")
plt.ylabel("Loading")
ax.legend(loc=0)
plt.show()

Yemp = Aemp@X_cent
for h in range(Yemp.size):
    for x in range(8):
        if x == h:
            print("coveigsemp", np.dot(Yemp[h], Yemp[x])/119)
print("empirpca", Yemp[:, 0:3])
print("empirpca", Yemp[0:3, :])

Xsmp = Aemp[:, 0:2]@Yemp[0:2, :]
print(Xsmp.shape)
print(Xsmp[:, 0:2])
x_pca = np.zeros([8, 119])
for k in range(119):
    x_pca[:, k] = Xsmp[:, k] + mus[k]
print("sample pca shape", x_pca.shape)
print("sample pca shape", x_pca[:, 0].shape)
print(x_pca)


# Q = np.cov(X.iloc[1:, :].astype(float), bias=False)
# print("covariance", Q)
# L, A = np.linalg.eig(Q)
# print("eigenvectors", A)
# print("eigenvalues", L)
# fract = L/np.sum(L)
# print(fract)
# Y = np.dot(A.T, X_cent)
# print("empirpca", Y[:, 0:])
# for h in range(8):
#     for x in range(8):
#         if x == h:
#             print("coveigs", np.dot(Y[h], Y[x])/119)

# pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(pcs, Yemp[:, 0].T, c='b', marker='o', label='PC1')
# ax.plot(pcs, Yemp[:, 1].T, c='r', lw=1,  marker='*',  markersize=10, label='PC2')
# ax.plot(pcs, Yemp[:, 2].T, c='g', lw=1,  marker='^', markersize=10, label='4 Years')
# ax.plot(pcs, Yemp[:, 3].T, c='m', lw=1,  marker='+', markersize=10, label='5 Years')
# ax.plot(pcs, Yemp[:, 4].T, c='c', lw=1,  marker='o', markersize=10, label='7 Years')
# ax.plot(pcs, Yemp[:, 5].T, c='C1', lw=1,  marker='s', markersize=10, label='10 Years')
# ax.plot(pcs, Yemp[:, 6].T, c='k', lw=1,  marker='p', markersize=10, label='20 Years')
# ax.plot(pcs, Yemp[:, 7].T, c='y', lw=1,  marker='+', markersize=10, label='30 Years')
# plt.title("Sample and PC Correlation", y=1.02, fontsize=22)
# plt.xlabel("Principal Components")
# plt.ylabel("Yield (%)")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(months, Yemp[0], c='b', marker='o',  markersize=10, label='PC1')
# ax.plot(months, Yemp[1], c='r', lw=1,  marker='*',  markersize=5, label='PC2')
# ax.plot(months, Yemp[2], c='g', lw=1,  marker='^', markersize=5, label='PC3')
# # ax.plot(months, Yemp[3], c='m', lw=1,  marker='D', markersize=5, label='5 Years')
# # ax.plot(months, Yemp[4], c='k', lw=1,  marker='>', markersize=5, label='7 Years')
# # ax.plot(months, Yemp[5], c='C1', lw=1,  marker='s', markersize=5, label='10 Years')
# # ax.plot(months, Yemp[6], c='c', lw=1,  marker='p', markersize=5, label='20 Years')
# # ax.plot(months, Yemp[7], c='y', lw=1,  marker='d', markersize=5, label='30 Years')
# plt.title("Empirical PCA Projections", y=1.02, fontsize=22)
# plt.xlabel("Real Time")
# plt.ylabel("Yield Changes")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(months, Yemp2[0], c='b', marker='o',  markersize=10, label='PC1')
# ax.plot(months, Yemp2[1], c='r', lw=1,  marker='*',  markersize=5, label='PC2')
# plt.title("Empirical PC1 and 2 Projections", y=1.02, fontsize=22)
# plt.xlabel("Real Time")
# plt.ylabel("Yield Changes")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(months, x_pca[:, 0], c='b', marker='o',  markersize=10, label='PC1')
# ax.plot(months, x_pca[:, 1], c='r', lw=1,  marker='*',  markersize=5, label='PC2')
# plt.title("PC1 and 2 Sample Projections", y=1.02, fontsize=22)
# plt.xlabel("Real Time")
# plt.ylabel("Yield Changes")
# ax.legend(loc=0)
# plt.show()

# x_pca = np.zeros([X.shape[0]-1, X.shape[1]])
# for i in range(X.shape[0]-1):
#     # for b in range(X.shape[0] - 1):
#         x_pca[i:i+1, :] = Aemp[i, 0:2]@Yemp + mus[i]
# print("sample pcs", x_pca)

pca = decomposition.PCA(n_components=2)
x_pca_sk = pca.fit_transform(X.tail(X.shape[0]-1).T)
x_pca_df = pd.DataFrame(x_pca_sk, columns=['PC1', 'PC2'])
"Data expressed as a linear combination of the Principal Components"
"The principal components replace original variables as the basis/domain"
"Dimensionality reduction is observed as first few PC's capture majority of variance in data"
print("pca", x_pca_df)
pca_weights = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'])
"Projections of original data onto principal components"
"Empirical Principal components expressed as linear combinations of original variables"
"Columns contain eigenvectors associated with each principal component"
"Each element represents a loading, how much (weight) each original variable contributes to the corresponding PC"
print("weights", pca_weights)
# for h in range(8):
#     for x in range(8):
#         print("coveigs", np.dot(pca.components_[h], pca.components_[x]))
loadings_sklearn = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings_sklearn, columns=['PC1', 'PC2'])
"Loadings matrix/correlations between the original variables and the principal components."
"Dimensionality reduction also observed as majority of correlation is captured by first few PC's"
print("loading_matrix", loading_matrix)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(months, x_pca_sk[:, 0:1], c='b', marker='o', label='PC1')
# ax.plot(months, x_pca_sk[:, 1:2], c='r', lw=1,  marker='*',  markersize=10, label='PC2')
# # ax.plot(months, x_pca_sk[:, 2:3], c='g', lw=1,  marker='^', markersize=10, label='PC3')
# plt.title("Uncorrelated Principal Components Sklearn", y=1.02, fontsize=22)
# plt.xlabel("Months")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(pcs[0:2], x_pca_sk[0].T, c='b', marker='o', label='PC1')
# ax.plot(pcs[0:2], x_pca_sk[1].T, c='r', lw=1,  marker='*',  markersize=10, label='PC2')
# # ax.plot(Calendar.columns[1:], x_pca_sk[2:3, ].T, c='g', lw=1,  marker='^', markersize=10, label='PC3')
# # ax.plot(Calendar.columns[1:], x_pca_sk[3:4, ].T, c='c', marker='D', label='PC4')
# # ax.plot(Calendar.columns[1:], x_pca_sk[4:5, ].T, c='k', lw=1,  marker='d',  markersize=10, label='PC5')
# # ax.plot(Calendar.columns[1:], x_pca_sk[5:6, ].T, c='y', lw=1,  marker='p', markersize=10, label='PC6')
# plt.title("Covariance of Principal Components Sklearn", y=1.02, fontsize=22)
# plt.xlabel("Principal Components")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(Calendar.columns[1:], pca.components_[0:1, ].T, c='b', marker='o', label='PC1 Level')
# ax.plot(Calendar.columns[1:], pca.components_[1:2, ].T, c='r', lw=1,  marker='*',  markersize=10, label='PC2 Slope')
# # ax.plot(Calendar.columns[1:], pca.components_[2:3, ].T, c='g', lw=1,  marker='^', markersize=10, label='PC3 Curvature')
# # ax.plot(Calendar.columns[1:], pca.components_[3:4, ].T, c='m', lw=1,  marker='D', markersize=5, label='PC4')
# # ax.plot(pcs, pca.components_[4:5, ].T, c='k', lw=1,  marker='>', markersize=5, label='PC5')
# # ax.plot(pcs, pca.components_[5:6, ].T, c='C1', lw=1,  marker='s', markersize=5, label='PC6')
# # ax.plot(pcs, pca.components_[6:7, ].T, c='c', lw=1,  marker='p', markersize=5, label='PC7')
# # ax.plot(pcs, pca.components_[7:8, ].T, c='y', lw=1,  marker='d', markersize=5, label='PC8')
# plt.title("Yield Curve Loadings (Sklearn)", y=1.02, fontsize=22)
# plt.xlabel("Time to Maturity")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(pcs[0:2], pca.components_[:, 0:1], c='b', marker='o', label='PC1')
# ax.plot(pcs[0:2], pca.components_[:, 1:2], c='r', lw=1,  marker='*',  markersize=10, label='PC2')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 2:3], c='g', lw=1,  marker='^', markersize=10, label='PC3')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 3:4], c='m', lw=1,  marker='D', markersize=5, label='PC4')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 4:5], c='k', lw=1,  marker='>', markersize=5, label='PC5')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 5:6], c='C1', lw=1,  marker='s', markersize=5, label='PC6')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 6:7], c='c', lw=1,  marker='p', markersize=5, label='PC7')
# # ax.plot(Calendar.columns[1:], pca.components_[:, 7:8], c='y', lw=1,  marker='d', markersize=5, label='PC8')
# plt.title("Yield Curve Movements (Sklearn)", y=1.02, fontsize=22)
# plt.xlabel("Time to Maturity")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(pcs[0:2], loadings_sklearn[0:1, ].T, c='b', marker='o', label='PC1')
# ax.plot(pcs[0:2], loadings_sklearn[1:2, ].T, c='r', lw=1,  marker='*',  markersize=10, label='PC2')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[2:3, ].T, c='g', lw=1,  marker='^', markersize=10, label='PC3')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[3:4, ].T, c='m', lw=1,  marker='D', markersize=5, label='PC4')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[4:5, ].T, c='k', lw=1,  marker='>', markersize=5, label='PC5')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[5:6, ].T, c='C1', lw=1,  marker='s', markersize=5, label='PC6')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[6:7, ].T, c='c', lw=1,  marker='p', markersize=5, label='PC7')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[7:8, ].T, c='y', lw=1,  marker='d', markersize=5, label='PC8')
# plt.title("Degree of Correlation in PC's (Sklearn)", y=1.02, fontsize=22)
# plt.xlabel("Time to Maturity")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_axisbelow(True)
# ax.yaxis.grid(color='gray', linestyle='dashed')
# ax.plot(pcs, loadings_sklearn[:, 0:1], c='b', marker='o', label='2 Years')
# ax.plot(pcs, loadings_sklearn[:, 1:2], c='r', lw=1,  marker='*',  markersize=10, label='3 Years')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 2:3], c='g', lw=1,  marker='^', markersize=10, label='PC3')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 3:4], c='m', lw=1,  marker='p', markersize=10, label='PC4')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 4:5], c='k', lw=1,  marker='>', markersize=5, label='PC5')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 5:6], c='C1', lw=1,  marker='s', markersize=5, label='PC6')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 6:7], c='c', lw=1,  marker='p', markersize=5, label='PC7')
# # ax.plot(Calendar.columns[1:], loadings_sklearn[:, 7:8], c='y', lw=1,  marker='d', markersize=5, label='PC8')
# plt.title("Correlations between PC's and Original Variables (Sklearn)", y=1.02, fontsize=22)
# plt.xlabel("Time to Maturity")
# plt.ylabel("Loading")
# ax.legend(loc=0)
# plt.show()

T_i = np.array([2, 3, 4, 5])
c_i = np.array([80, 70, 150, 40])
y_i = np.array([0.407, 0.522, 0.685, 0.863])
dyi = np.array(X.iloc[1:5, -1:])
dV = np.zeros([8, 1])

print('1st_term', 0.01*y_i[0]*c_i[0]*np.exp(-0.01*y_i[0]*T_i)/12)
print('test', 0.01*y_i[0]*c_i[0]*np.exp(-0.01*y_i[0]*T_i[0])/12 - T_i[0]*c_i[0]*np.exp(-0.01*y_i[0]*T_i[0])*dyi[0])

sum = np.zeros([1, 119])

for s in range(4):
    y = np.zeros([1, 119])
    k = 0
    print(s, " ", k)
    if s == 0:
        for k in range(25):
            print(k)
            # y = y + (c_i[s]*Calendar.iloc[k+95:k+96, s+1:s+2].values.astype(float)*np.exp(-Calendar.iloc[k+95:k+96, s+1:s+2].values.astype(float)*(T_i[s] - (k)/12))/12) + \
            y = y + (((k)/12)-T_i[s])*c_i[s]*np.exp(-Calendar.iloc[k+95:k+96, s+1:s+2].values.astype(float)*(T_i[s] - (k)/12))*x_pca[s:s+1, k+94:k+95]
    if s == 1:
        for k in range(37):
            # y = y + (c_i[s] *Calendar.iloc[k+83:k+84, s+1:s+2].values.astype(float)*np.exp(-Calendar.iloc[k+83:k+84, s+1:s+2].values.astype(float)*(T_i[s]-(k)/12))/12) + \
            y = y + (((k)/12)-T_i[s])*c_i[s]*np.exp(-Calendar.iloc[k+83:k+84, s+1:s+2].values.astype(float)*(T_i[s]-(k)/12))*x_pca[s:s+1, k+82:k+83]
    if s == 2:
        for k in range(49):
            # y = y + (c_i[s] * Calendar.iloc[k+71:k+72, s+1:s+2].values.astype(float) * np.exp(-Calendar.iloc[k+71:k+72, s+1:s+2].values.astype(float)*(T_i[s]-(k)/12))/12) + \
             y = y + (((k)/12)-T_i[s])*c_i[s]*np.exp(-Calendar.iloc[k+71:k+72, s+1:s+2].values.astype(float)*(T_i[s]-(k)/12))*x_pca[s:s+1, k+70:k+71]
    if s == 3:
        for k in range(61):
            # y = y + (c_i[s]*Calendar.iloc[k+59:k+60, s+1:s+2].values.astype(float) * np.exp(-Calendar.iloc[k+59:k+60, s+1:s+2].values.astype(float)*(T_i[s]-((k)/12))) / 12) + \
            y = y + (((k)/12)-T_i[s]) * c_i[s]*np.exp(-Calendar.iloc[k+59:k+60, s+1:s+2].values.astype(float)*(T_i[s]-((k)/12)))*x_pca[s:s+1, k+58:k+59]
    print(y)
    sum = sum + y
print("res", sum)
print("stdev", np.std(sum))
print("mean", np.mean(sum))

