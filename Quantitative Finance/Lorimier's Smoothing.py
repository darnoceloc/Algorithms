import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sci

alpha = 0.1
Times = np.array([2, 3, 4, 5, 7, 10, 20, 30])
Times_T = np.transpose(Times)
Yields = np.array([-0.0079, -0.0073, -0.0065, -0.0055, -0.0033, -0.0004, 0.0054, 0.0073])

Betas = np.zeros([len(Times)+1, 1])
RHS = np.zeros([len(Times)+1, 1])
A = np.zeros([len(Times)+1, len(Times)+1])

A[0][0] = 0
RHS[0][0] = 0

for i in range(1, len(Times)+1):
    A[0][i] = Times[i-1]
    RHS[i][0] = alpha*Times[i-1]*Yields[i-1]
    A[i][0] = alpha*Times[i-1]


def lorimier_dot_prod(h_i, h_j):
    ''' Calculates dot product of h_i, h_j  based on lorimeir definition'''
    v1 = Times[h_i]
    v2 = Times[h_j]
    res = v1 * v2 + 0.5 * min(v1, v2) ** 2 * max(v1, v2) - min(v1, v2) ** 3 / 6.
    res = (alpha * res + (1.0 if v1 == v2 else 0.0))
    return res


for i in range(1, len(Times)+1):
    for j in range(1, len(Times)+1):
        A[i][j] = lorimier_dot_prod(i-1, j-1)

print(A)
beta = np.linalg.solve(A, RHS)
print(beta)

def h_i(t_i, t):
    ''' calculates hi(u) '''
    return t_i + t_i*min(t_i, t) - 0.5*min(t_i, t)**2


def integ_hi(ti, t):
    return ti * t + 0.5 * min(ti, t) ** 2 * max(ti, t) - min(ti, t) ** 3 / 6.


def yield_integ(t):
    result = np.zeros([len(Times)+1, 1])
    for k in range(0, len(Times)):
        result[k] = beta[k+1]*(integ_hi(Times[k], t))
    return result

def y_t(t):
    y_ts = np.zeros([len(t), 1])
    k = 0
    for x in t:
        y_ts[k] = beta[0] + np.sum(yield_integ(x))/x
        k = k + 1
    return y_ts

Times_n = np.append(Times, 6)
print(Times_n)
print(100*y_t(Times_n))
smooth = np.arange(0, 33)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.scatter(Times, Yields, c='b', marker='o', label='Actual Yield Data')
ax.plot(y_t(smooth), c='r', lw=1, markersize=10, label='Lorimiers Spline Fit')
plt.title("Lorimier's Smoothing Yield Curve", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity")
plt.ylabel("Yield (%)")
ax.legend(loc=0)
plt.show()

def f0(t):
        ''' calculates f0 @ time t based on provided beta '''
        # TODO: test
        # TODO: implement plots in parent discount method
        for tau, beta in zip(Times, Betas):
            hi = h_i(tau, t)
            res = res + beta*hi
        return res / t


def forward(loc, prev_loc):
    ''' Calculates forward rate based on f0 '''
    ti = Times[loc]
    prev_ti = Times[prev_loc]
    weights = np.zeros(2)
    weights[0] = prev_ti / ti
    mat = Times[loc]
    f0 = f0(mat)
    forward = (f0 * ti - prev_loc * prev_ti) / (ti - prev_ti)
    return forward, f0
