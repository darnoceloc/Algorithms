""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
matplotlib.use('Qt5Agg')

plt.close('all') #close any open plots
"""
===============================================================================
============================ Question 1 =======================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, legend=None, title=None, M=0, sigma=0):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo', label=legend[0]) #plot training datahttps://www.hackerrank.com/challenges/swap-nodes-algo/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=search&h_r=next-challenge&h_v=zen
    p2 = plt.plot()
    p3 = plt.plot()
    if(x2 is not None):
        p2 = plt.plot(x2, t2, 'g', label=legend[1]) #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r', label=legend[2]) #plot training data

    #add title, legend and axes labels
    plt.title(title)
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x2 is None and x3 is None):
        plt.legend(p1[0], legend)
    if(x2 is None):
        plt.legend((p1[0], p2[0]), legend)
    if (x3 is None):
        plt.legend(legend)
    else:
        plt.legend()
    plt.xlim(-4.5, 4.5)
    plt.ylim(-2, 2)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if(M != 0 and sigma != 0):
        textstr = '\n'.join((
            r'$\mathrm{Model \ Order}=%.2f$' % (M,),
            r'$\sigma=%.2f$' % (sigma,)))
        plt.text(4, -1.75, textstr, fontsize=14, horizontalalignment='right', verticalalignment='bottom', bbox=props)
        plt.show()
        return
    if(M != 0 and sigma == 0):
        textstr = r'$\mathrm{Model \ Order}=%.2f$' % (M,)
        plt.text(4, -1.75, textstr, fontsize=14, horizontalalignment='right', verticalalignment='bottom', bbox=props)
        plt.show()
        return
    if(M == 0 and sigma == 0):
        plt.show()
        return

def fitdata(x, t, M):
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X)@X.T@t
    return w

def fitdata2(x, t, M, sigma=0):
    X2 = np.zeros((x.size, M))
    mus = np.random.choice(np.random.permutation(x), size=M, replace=False)
    for i in range(x.size):
        for j in range(M):
            if(sigma == 0):
                X2[i][j] = np.exp(gamma * ((x[i] - mus[j])**2))
            if(sigma != 0):
                X2[i][j] = np.exp(-((x[i] - mus[j]) ** 2)/2*(sigma ** 2))
    if np.linalg.det(X2.T @ X2) > 1e-10:
        w2 = np.linalg.inv(X2.T@X2) @ X2.T @ t
        return w2, X2, mus
    else:
        idxs = np.diag_indices(M)
        X2_reg = X2.T@X2
        X2_reg[idxs] += 1e-13
        w2 = np.linalg.inv(X2_reg)@X2.T@t
        return w2, X2, mus

def fitdata3(x, t, M, sigma=0):
    X3 = np.zeros((x.size, M))
    mus_even = np.array(np.array_split(np.arange(-4.0, 4.0, 0.001), M))
    for i in range(x.size):
        for j in range(M):
            if(sigma == 0):
                X3[i][j] = np.exp(gamma * ((x[i] - np.mean(mus_even[j]))**2))
            if(sigma != 0):
                X3[i][j] = np.exp(-((x[i] - np.mean(mus_even[j]))**2)/2*(sigma ** 2))
    if np.linalg.det(X3.T@X3) > 1e-10:
        w3 = np.linalg.inv(X3.T @ X3) @ X3.T @ t
        return w3, X3, mus_even
    else:
        idxs = np.diag_indices(M)
        X3_reg = X3.T@X3
        X3_reg[idxs] += 1e-13
        w3 = np.linalg.inv(X3_reg) @ X3.T @ t
        return w3, X3, mus_even

def compare_models(x1, t1, x1_test, t1_test, M=0, sigma=0):
    # fit regression model
    w_train_pbf = fitdata(x1, t1, M)
    rbf_random = fitdata2(x1, t1, M, sigma)  # regression weights
    rbf_even = fitdata3(x1, t1, M, sigma)  # regression weights

    X_test_pbf = np.array([x1_test**m for m in range(w_train_pbf.size)]).T

    X_test_random = np.zeros((x1_test.size, M))
    for i in range(x1_test.size):
        for j in range(M):
            X_test_random[i][j] = np.exp(- ((x1_test[i] - rbf_random[2][j]) ** 2)/2*(sigma ** 2))

    X_test_even = np.zeros((x1_test.size, M))
    for i in range(x1_test.size):
        for j in range(M):
            X_test_even[i][j] = np.exp(-((x1_test[i] - np.mean(rbf_even[2][j])) ** 2)/2*(sigma ** 2))

    # estimate labels
    est_test_pbf = X_test_pbf@w_train_pbf
    est_test_random = X_test_random@rbf_random[0]  # get estimated labels of test data
    est_test_even = X_test_even@rbf_even[0]  # get estimated labels of training data

    # calculate instantaneous error
    inst_error_test_pbf = (est_test_pbf - t1_test)
    inst_error_test_random = (est_test_random - t1_test)
    inst_error_test_even = (est_test_even - t1_test)

    # calculate mean absolute error
    abs_error_test_pbf[M-1] = np.mean(np.absolute(inst_error_test_pbf))
    abs_error_test_random[M-1] = np.mean(np.absolute(inst_error_test_random))
    abs_error_test_even[M-1] = np.mean(np.absolute(inst_error_test_even))

    return abs_error_test_pbf, abs_error_test_random, abs_error_test_even

""" ======================  Variable Declaration ========================== """
M = 32 #regression model order
maxM = 50 #Max regression model order
sigma = 0.58
maxsigma = 1.00
gamma = -1 / (2 * (sigma**2))

""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:, 0]
t1 = data_uniform[:, 1]

x2 = np.arange(x1[0], x1[-1], 0.001)
t2 = np.sinc(x2)
plotData(x1, t1, x2=x2, t2=t2, legend=['Training Data', 'True Function', 'Estimated\nPolynomial'], title='Provided Training Data')

""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """

w1 = fitdata(x1, t1, M=M)
XT1 = np.array([x2**m for m in range(w1.size)]).T
t3 = XT1@w1 #compute the predicted value
plotData(x1, t1, x2=x2, t2=t2, x3=x2, t3=t3, legend=['Training Data', 'True Function', 'Estimated\nPolynomial'], title='Training Polynomial Basis Functions', M=M)

params_train_random = fitdata2(x1, t1, M=M)
t3_random = params_train_random[1]@params_train_random[0] #compute the predicted value
plotData(x1, t1, x2=x2, t2=t2, x3=x1, t3=t3_random, legend=['Training Data', 'True Function', 'Estimated\nRBF'], title='Training Random Uniform Means RBF ', M=M, sigma=sigma)

params_train_even = fitdata3(x1, t1, M=M)
t3_even = params_train_even[1]@params_train_even[0] #compute the predicted value
plotData(x1, t1, x2=x2, t2=t2, x3=x1, t3=t3_even, legend=['Training Data', 'True Function', 'Estimated\nRBF'], title='Training Evenly Spaced Means RBF', M=M, sigma=sigma)

""" ======================== Load Test Data  and Test the Model =========================== """
"""This is where you should load the testing data set. You should NOT re-train the model   """

data_uniform = np.load('TestData.npy')
x1_test = data_uniform[:, 0]
t1_test = data_uniform[:, 1]

x3 = np.arange(x1_test[0], x1_test[-1], 0.001)  #get equally spaced points in the xrange

t4 = np.sinc(x3)
plotData(x1_test, t1_test, x2=x3, t2=t4, legend=['Test Data', 'True Function', 'Estimated\nPolynomial'], title='Provided Test Data')

XT2 = np.array([x3**m for m in range(w1.size)]).T
t5 = XT2@w1 #compute the predicted value
plotData(x1_test, t1_test, x2=x3, t2=t4, x3=x3, t3=t5, legend=['Test Data', 'True Function', 'Estimated\nPolynomial'], title='Test Polynomial Basis Functions', M=M)

X_phi = np.zeros((x1_test.size, M))
for i in range(x1_test.size):
    for j in range(M):
        X_phi[i][j] = np.exp(gamma*((x1_test[i] - params_train_random[2][j])**2))
t6 = X_phi@params_train_random[0] #compute the predicted value
plotData(x1_test, t1_test, x2=x3, t2=t4, x3=x1_test, t3=t6, legend=['Test Data', 'True Function', 'Estimated\nRBF'],title='Test Random Uniform Means RBF', M=M, sigma=sigma)

X_phi_even = np.zeros((x1_test.size, M))
for i in range(x1_test.size):
    for j in range(M):
        X_phi_even[i][j] = np.exp(gamma*((x1_test[i] - np.mean(params_train_even[2][j]))**2))
t6_even = X_phi_even@params_train_even[0] #compute the predicted value
plotData(x1_test, t1_test, x2=x3, t2=t4, x3=x1_test, t3=t6_even, legend=['Test Data', 'True Function', 'Estimated\nRBF'],title='Test Evenly Spaced Means RBF', M=M, sigma=sigma)

""" ======================== Evaluate model over range of M and s values =========================== """
srange = np.linspace(0.1, maxsigma, 10)

best_M_rbf1 = defaultdict()
best_M_rbf2 = defaultdict()

abs_error_test_pbf = np.zeros((maxM))

for sig in srange:
    abs_error_test_random = np.zeros((maxM))
    abs_error_test_even = np.zeros((maxM))
    for M in range(1, (maxM+1)):
        predicts = compare_models(x1, t1, x1_test, t1_test, M, sig)
    best_M_rbf1[sig] = np.min(predicts[1])
    best_M_rbf2[sig] = np.min(predicts[2])
    best_M_rbf1[sig] = [best_M_rbf1[sig], np.argmin(predicts[1])+1]
    best_M_rbf2[sig] = [best_M_rbf2[sig], np.argmin(predicts[2])+1]
    # Plot the error vs model order
    plt.figure()
    p1_m = plt.plot(np.arange(1, (maxM + 1)), predicts[0], marker='o', color='b', fillstyle='none')
    p2_m = plt.plot(np.arange(1, (maxM+1)), predicts[1], marker='o', color='r', fillstyle='none')
    p3_m = plt.plot(np.arange(1, (maxM+1)), predicts[2], marker='o', color='g', fillstyle='none')
    # plt.ylim(0, 3)
    plt.xlabel("Model Order")
    plt.ylabel("Eabs")
    plt.title("RBF Regression Error for Varying Model Orders")
    plt.legend(("PBF Test Error", "Random Uniform Means RBF Test Error", "Evenly Spaced Means RBF Test Error"),
               bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = r'$\sigma=%.2f$' % (sig,)
    plt.text(x=3, y=0.1, s=textstr, fontsize=14, bbox=props)
    plt.show()

best_M_pbf = np.argmin(abs_error_test_pbf)+1

print("Best M for pbf", best_M_pbf)

print("Ordered models rbf random", sorted(best_M_rbf1.items(), key=lambda x: x[1]))
print("Ordered models rbf even", sorted(best_M_rbf2.items(), key=lambda x: x[1]))

print("Best Model rbf random", sorted(best_M_rbf1.items(), key=lambda x: x[1])[0][0], sorted(best_M_rbf1.items(), key=lambda x: x[1])[0][1][1])
print("Best Model for rbf even", sorted(best_M_rbf2.items(), key=lambda x: x[1])[0][0], sorted(best_M_rbf2.items(), key=lambda x: x[1])[0][1][1])
