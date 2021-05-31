import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import numpy as np
import datetime as dt
import os

L01 = 0.03
F013 = 0.05
Rswap = 0.04

P01 = 1/(1+L01)
P03 = P01/(1+(F013*2))
L03 = ((1/P03) - 1)/3
L02 = 0.5*(L01+L03)
P02 = 1/(1+(2*L02))
P04 = (1 - Rswap*(P01+P02+P03))/(1+Rswap)
print(P04)
print(P02)
print(P01)
PV = 15*(P01+P02+P03) + 115*P04
print(PV)

os.chdir("/home/darnoc/Documents/Personal/Personal Finance/Interest_Rate_Models/")
cwd = os.getcwd()
# import calendar and obtain a vector "days" that counts the day to each cashflow.
Calendar = pd.read_excel(cwd+'/Bootstrap_data.xls', header=4, usecols="C:F, G:I")
Calendar = Calendar[pd.notnull(Calendar)]
Calendar = Calendar.reset_index(drop=True)
t_0 = dt.datetime(2012, 10, 3) # spot date
t_i = [0] * 39
days = [0] * 39
for i in range(0, 39):
    days[i] = Calendar.Tenor[i] - t_0
    days[i] = days[i].days
days = pd.DataFrame({"Days": days})
Calendar = Calendar.join(days)

# import rates
Rates = Calendar[['Maturity Dates', 'Market Quotes', 'Source']]
Rates = Rates.replace(to_replace=['None', '-'], value=np.nan).dropna()
# define global variables
N = Calendar.shape[0]
n = Rates.shape[0]

# initializing vectors
P = [0] * N  # discount Factors
F = [0] * N  # Forward rates
L = [0] * N  # Simple Rates
Prices = [0] * n # 17 prices

# Build MATRIX C in three steps, MM, Futures and Swaps

# Build LIBOR rows for C matrix under C_L
Rates_L = Rates[Rates['Source'] == 'LIBOR']
Rates_L = Rates_L.reset_index(drop=True)
Calendar_L = Calendar[(Calendar['Tenor'] > t_0) & (Calendar['Tenor'] < t_0 + dt.timedelta(days=181))]
Calendar_L = Calendar_L.reset_index(drop=True)
n_L = Rates_L.shape[0]
C_L = np.zeros([n_L, N]) ## 4 x 39
for i in range(n_L):
    for t in range(N):
        if Calendar['Tenor'][t] == Rates_L['Maturity Dates'][i]:
            C_L[i, t] = 1 + (Rates_L['Market Quotes'][i]*0.01)*(Calendar_L['Days'][i]/360)


# Build Futures rows for C matrix under C_F
Rates_F = Rates[Rates['Source'] == 'Futures']
Rates_F = Rates_F.reset_index(drop=True)
Calendar_F = Calendar[(Calendar['Tenor'] > t_0 + dt.timedelta(days=180)) & (Calendar['Tenor'] < t_0 + dt.timedelta(days=370))]
Calendar_F = Calendar_F.reset_index(drop=True)
print(Calendar_F)
n_F = Rates_F.shape[0]
C_F = np.zeros([n_F, N])
for i in range(n_F-1):
    start = Calendar_F['Maturity Dates'][i]  # start with the first date
    end = Calendar_F['Maturity Dates'][i+1]  # second date, then it goes into a loop
    for t in range(N):
        if Calendar['Tenor'][t] == start:
            C_F[i, t] = -1  # because the FIRST cash flow for the futures is -1 (when you buy it)
        elif Calendar['Tenor'][t] == end:
            C_F[i, t] = 1 + ((1 - 0.01*Rates_F['Market Quotes'][i])*(Calendar_F['Days'][i + 1] - Calendar_F['Days'][i])/360)


# Build Swap rows for C matrix under C_S
Rates_S = Rates[Rates['Source'] == 'Swap']
Rates_S = Rates_S.reset_index(drop=True)
Calendar_S = Calendar[pd.to_numeric(Calendar['Swaps'], errors='coerce').notnull()]
Calendar_S = Calendar_S.reset_index(drop=True)
n_S = Rates_S.shape[0]
C_S = np.zeros([n_S, N])

for i in range(n_S):
    Maturity = Rates_S['Maturity Dates'][i]
    DaysToMaturity = int(Calendar_S[Calendar_S['Tenor'] == Rates_S['Maturity Dates'][i]]['Days']) # takes the number from the days column coinciding with the dates in both tables
    for t in range(N):
        if Calendar['Days'][t] < DaysToMaturity:
            PrevDate = 0
            if Calendar['Swaps'][t] == 1.0:
                PrevDate = 0
                C_S[i, t] = Rates_S['Market Quotes'][i]*0.01*(Calendar['Days'][t] - PrevDate)/360
            if Calendar['Swaps'][t] > 1.0:
                PrevDate = tuple(Calendar_S[Calendar_S['Days'] < Calendar['Days'][t]]['Days'].tail(1))[0] # take the previous to last number of days
                C_S[i, t] = Rates_S['Market Quotes'][i]*0.01*(Calendar['Days'][t] - PrevDate)/360
        elif Calendar['Days'][t] == DaysToMaturity:
            PrevDate = tuple(Calendar_S[Calendar_S['Days'] < DaysToMaturity]['Days'].tail(1))[0]
            C_S[i, t] = 1 + Rates_S['Market Quotes'][i]*0.01*(DaysToMaturity - PrevDate)/360
        else:
            C_S[i, t] = 0


# Put 3 Matrices Together and Print
C = np.vstack((C_L, C_F))
C = np.vstack((C, C_S))
C = pd.DataFrame(C)
C.columns = Calendar['Tenor']
pd.set_option('display.max_columns', None)
C
print(C)

# Build Prices Vector - using the instructions in the slides
Prices = np.zeros(n)
idx = 0
for r in Rates['Source']:
    if r == 'LIBOR':
        Prices[idx] = 1
    elif r == 'Futures':
        Prices[idx] = 0
    elif r == 'Swap':
        Prices[idx] = 1
    idx += 1

# Pseudoinverse calculation
# Create W Matrix
W = np.zeros([N, N])  # W needs to be NxN and it is a diagonal matrix
W[0, 0] = 1/np.sqrt(Calendar['Days'][0]/360)  # first entry
for i in range(1, N): # note that this starts from 1
    W[i, i] = 1 / np.sqrt((Calendar['Days'][i] - Calendar['Days'][i - 1])/360)

# Create M Matrix
M = np.zeros([N, N])
for i in range(N): # this starts from 0
    M[i, i] = 1
    if i < N - 1:
        M[i + 1, i] = -1
foo = np.zeros(N)  # this is the vector (1,0,0,0,0,0,0...,0)
foo[0] = 1
A = (C.to_numpy()@(np.linalg.inv(M))@np.linalg.inv(W))

term1 = np.transpose(A)@np.linalg.pinv(A@A.T)
term2 = Prices - C@np.linalg.inv(M)@foo.T
delta = term1@term2
# Below is the part where the bond prices (discount factors) are extracted from the delta vector
P[0] = delta[0] * W[0, 0] + 1
for i in range(1, N):
    P[i] = delta[i] * W[i, i] + P[i - 1]
fig, ax = plt.subplots()
plt.plot(P, c='b', lw=2, marker='.', markerfacecolor='green', markersize=15)
plt.title("ZCB Price Curve", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity (T - to)")
plt.ylabel("Price [P(T0, T)]")
plt.show()

F[0] = 0.095
for i in range(1, N):
    F[i] = (P[i - 1]/P[i] - 1)/((Calendar['Days'][i] - Calendar['Days'][i - 1])/360)*100
plt.plot(Calendar['Days'] / 365, F, c='r', lw=1.5, marker='D', markerfacecolor='purple', markersize=8)
plt.title("Forward Rate Curve", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity (T - to)")
plt.ylabel("Simple Forward Rate F(T0, T1, T2)]")
plt.show()

# using this to estimate the discount bond prices (so d has to be compared to P)
d = np.linalg.inv(M)@(np.linalg.inv(W)@delta + foo.T)

# here we estimate the fwd rates using d instead of P
fwd = [0] * N
fwd[0] = 0.095 # this is in percentage points
for f in range(1, N):
    fwd[f] = ((d[f-1]/d[f])-1)/((Calendar['Days'][f] - Calendar['Days'][f-1])/360)*100
plt.plot(Calendar['Days']/365, fwd, c='g', lw=1.5, marker='*', markerfacecolor='red', markersize=8)
plt.title("Simple Forward Rate Curve", y=1.02, fontsize=22)
plt.xlabel("Time to Maturity (T - to)")
plt.ylabel("Simple Forward Rate F(T0, T1, T2)]")
plt.xticks(rotation=30)
plt.show()

## Q2
print(round(fwd[38], 5))
print(round(F[38], 5))
