from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_excel('StructuredCredit_PSet4.xlsx', index_col=1, header=1, engine='openpyxl')
data = np.asarray(data['Default Prob']).astype(np.float64)
print(data)
print(len(data))
P = np.zeros(shape=(20, 21))
P[0][0] = 1 - data[1]
P[0][1] = data[1]

for i in range(1, len(data)-1):
    j = i+1
    while j > 0:
        P[i][j] = P[i-1][j-1]*data[i] + P[i-1][j]*(1 - data[i])
        j = j - 1
    P[i][0] = P[i-1][0]*(1 - data[i])
    print("i = ", i)
    print("j = ", j)
    print(P[i])

print(P[19][2:21])
total_losses = 0
variance = 0
for x in range(0, len(data)):
    total_losses += x*P[19][x]
for x in range(0, len(data)):
    variance += ((x - total_losses)**2)*P[19][x]
tranche_02 = P[19][1] + 2*np.sum(P[19][2:21])
tranche_24 = P[19][3] + 2*np.sum(P[19][4:21])
tranche_420 = total_losses - tranche_02 - tranche_24
print(P[19][3])
print(total_losses)
print(variance)
print(tranche_02)
print(tranche_24)
print(tranche_420)

