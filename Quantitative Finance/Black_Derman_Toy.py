import numpy as np
import pandas as pd
import math

# node 수
num_node = 10
# volatility
vol = 0.05
p = 0.5
up = np.exp(vol)
down = 1/up
fv = 1
#YTM
ytm = [0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.0355, 0.036, 0.0365, 0.037]
adj = np.zeros(num_node)
pv = [fv/(1+ytm[i])**(i+1) for i in range(num_node)]

spt = np.zeros((num_node, num_node))
for i in range(num_node):
    for j in range(num_node):
        if i == 0 and j == 0:
            spt[i, j] = ytm[0]
        elif spt[i, j] > 0:
            spt[i, j] = spt[i, j]
        else:
            if i > j:
                spt[i, j] = spt[i-1, j]*up
            elif i == j:
                spt[i, j] = spt[i-1, j-1] * down
            else:
                spt[i, j] = 0

print(pd.DataFrame(spt.T))


def pvr(n):
    y = np.zeros((num_node+10, num_node+10, num_node+10))
    for i in range(num_node+1):
        for a in range(i+1, 0, -1):
            for b in range(i+1, 0, - 1):
                c = i
                if int(a) == c+1:
                    y[a, b, c] = fv
                elif y[a, b, c] > 0:
                    y[a, b, c] = y[a, b, c]
                elif int(a) >= int(b):
                    y[a, b, c] = (y[a + 1, b, c] / (1 + spt[a-1, b-1] + adj[a - 1])) * p + (y[a + 1, b + 1, c] /
                                                                            (1 + spt[a-1, b-1] + adj[a - 1])) * (1 - p)
                else:
                    y[a, b, c] = 0
    return y[1, 1, n]


def cost(n):
    return pvr(n+1) - pv[n]


for i in range(num_node):
    print(cost(i))

adj = np.zeros(num_node)
for i in range(1, num_node):
    tick = 0.0000000002
    k = 0
    while abs(cost(i)) > 0.01:
        k += 1
        d1 = cost(i)
        adj[i] += 0.000001
        d = (cost(i) - d1)/0.000001
        adj[i] -= 0.000001
        adj[i] -= tick*d
        if k > 50000:
            break

index = [str(i+1) + " year" for i in range(num_node)]
print(pd.Series(adj, index=index))
adjspt = np.zeros((num_node, num_node))
for i in range(num_node):
    for j in range(num_node):
        if i >= j:
            adjspt[i, j] = spt[i, j] + adj[i]
print(pd.DataFrame(adjspt.T))

adjpv = [pvr(i+1) for i in range(num_node)]
print(pd.Series(adjpv, index=index))

for i in range(num_node):
    print(cost(i))