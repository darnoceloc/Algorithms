import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import datetime as dt
import os
from math import isnan

os.chdir("/home/darnoc/Documents/Personal/Personal Finance/Interest_Rate_Models/")
# import calendar and obtain a vector "days" that counts the day to each cashflow.
cwd = os.getcwd()
Calendar = pd.read_excel(cwd+'/Bootstrap_data.xls', header=4, usecols="C:F, G:I, J, L:N")
Calendar = Calendar[pd.notnull(Calendar)]
Calendar = Calendar.reset_index(drop=True)

t_0 = dt.datetime(2012, 10, 3) # spot date
t_swap = dt.datetime(2013, 10, 3)
new_col = dict(zip(Calendar[pd.notnull(Calendar['Tenor'])]['Tenor'], np.zeros(Calendar['Discount bond price'].size)))
swaps = dict(zip(Calendar[pd.notnull(Calendar['Swaps'])]['Swaps'], np.zeros(Calendar[pd.notnull(Calendar['Swaps'])]['Swaps'].size)))
for a in Calendar[pd.notnull(Calendar['Swaps'])]['Swaps']:
    if a in Calendar['Swaps']:
        swaps[a] = tuple(Calendar[Calendar['Swaps'] == a]['Tenor'])[0]
swaps[0] = pd.Timestamp(t_0)
print(sorted(swaps.items(), key=lambda item: item[0]))
j = 0
k = 0
t = 2

while k < Calendar['Maturity Dates'].size:
    if Calendar['Source'][k] == 'LIBOR':
        if Calendar['Tenor'][j] == Calendar['Maturity Dates'][k]:
            new_col[Calendar['Tenor'][j]] = 1/(1+0.01*(Calendar['Market Quotes'][k]*((
                    Calendar['Tenor'][j] - t_0)/dt.timedelta(days=1)/360)))
            j = j+1
            k = k+1
        if Calendar['Tenor'][j] != Calendar['Maturity Dates'][k]:
            q = (Calendar['Tenor'][k] - Calendar['Tenor'][j])/(Calendar['Maturity Dates'][k] - Calendar['Tenor'][k-1])
            L = q*Calendar['Market Quotes'][k-1] + (1-q)*Calendar['Market Quotes'][k]
            new_col[Calendar['Tenor'][j]] = 1/(1+0.01*L*((Calendar['Tenor'][j] - t_0)/dt.timedelta(days=1)/360))
            j = j+1
    if Calendar['Source'][k] == 'Futures':
        if Calendar['Tenor'][j] == Calendar['Maturity Dates'][k]:
            f = 1 - 0.01*Calendar['Market Quotes'][k]
            new_col[Calendar['Tenor'][j]] = new_col[tuple(Calendar[(Calendar['Tenor'] > Calendar['Tenor'][j] - dt.timedelta(days=94)) &
                    (Calendar['Tenor'] < Calendar['Tenor'][j] - dt.timedelta(days=89))]['Tenor'])[0]]/(1+(f*(
                    tuple((Calendar['Tenor'][j] - Calendar[(Calendar['Tenor'] > Calendar['Tenor'][j] - dt.timedelta(days=94)) &
                      (Calendar['Tenor'] < Calendar['Tenor'][j] - dt.timedelta(days=88))]['Tenor']))[0]/dt.timedelta(days=1)/360)))
            j = j+1
            k = k+1
        if Calendar['Tenor'][j] != Calendar['Maturity Dates'][k]:
            q = (Calendar['Tenor'][k] - Calendar['Tenor'][j])/(Calendar['Maturity Dates'][k] - Calendar['Tenor'][k-1])
            L = q*Calendar['LIBOR Spot Rates'][k-1] + (1-q)*Calendar['LIBOR Spot Rates'][k]
            f = ((1+Calendar['LIBOR Spot Rates'][j])**((Calendar['Tenor'][j] - t_0)/dt.timedelta(days=1)/360)/
                    ((1+Calendar['LIBOR Spot Rates'][j-1])**((Calendar['Tenor'][j-1] - t_0)/dt.timedelta(days=1)/360)))**(
                    1/((Calendar['Tenor'][j] - Calendar['Tenor'][j-1])/dt.timedelta(days=1)/360)) - 1
            new_col[Calendar['Tenor'][j]] = new_col[Calendar['Tenor'][j-1]]/(1+f*((Calendar['Tenor'][j]
                                            - Calendar['Tenor'][j-1])/dt.timedelta(days=1)/360))
            j = j+1
    if Calendar['Source'][k] == 'Swap':
        if Calendar['Tenor'][j] == Calendar['Maturity Dates'][k]:
            swap_dict = {g: v for (g, v) in swaps.items() if g < t}
            summation = 0
            for x in range(1, t):
                summation += ((swap_dict[x] - swap_dict[x-1])/dt.timedelta(days=1)/360)*new_col[swap_dict[x]]
            new_col[Calendar['Tenor'][j]] = (1 - 0.01*Calendar['Market Quotes'][k]*summation)/((1 +
                    0.01*Calendar['Market Quotes'][k]*(Calendar['Tenor'][j] - swap_dict[t-1])/dt.timedelta(days=1)/360))
            j = j+1
            t = t+1
            k = k+1
            if j == 39:
                break
        if Calendar['Tenor'][j] != Calendar['Maturity Dates'][k]:
            q = (Calendar['Maturity Dates'][k] - Calendar['Tenor'][j])/(Calendar['Maturity Dates'][k] - Calendar['Maturity Dates'][k-1])
            R = q*Calendar['Market Quotes'][k-1] + (1-q)*Calendar['Market Quotes'][k]
            swap_dict = {g: v for (g, v) in swaps.items() if g < t}
            summation = 0
            for x in range(1, t):
                summation += ((swap_dict[x] - swap_dict[x-1])/dt.timedelta(days=1)/360)*new_col[swap_dict[x]]
            new_col[Calendar['Tenor'][j]] = (1 - 0.01*R*summation)/((1 +
                    0.01*R*(Calendar['Tenor'][j] - swap_dict[t-1])/dt.timedelta(days=1)/360))
            j = j+1
            t = t+1
            if j == 39:
                break

Calendar['New ZCB Prices'] = (sorted(new_col.items(), key=lambda item: item[1]))
Calendar['New ZCB Prices'] = Calendar['Tenor'].apply(lambda x: new_col.get(x))
print(Calendar['New ZCB Prices'])

new_col[swaps[0]] = 1
print(sorted(new_col.items(), key=lambda item: item[0]))
forward = ((new_col[swaps[29]]/new_col[swaps[30]]) - 1)/((swaps[30] - swaps[29])/dt.timedelta(days=1)/360)
print(forward, "final forward rate")

x, y = zip(*sorted(new_col.items(), key=lambda item: item[1]))
fig, ax = plt.subplots()
ax.plot(x, y)
plt.ylabel("Price [P(T0, T)]", labelpad=15, fontsize=16)
plt.xlabel("Date (dd-mm-YY)", labelpad=15, fontsize=16)
plt.title("Bootstrapped ZCB Price Curve", y=1.02, fontsize=22)
plt.xticks(fontsize=10)
ax.xaxis.set_major_locator(dates.YearLocator(2))
ax.xaxis.set_major_formatter(dates.DateFormatter('\n%m-%Y'))
ax.xaxis.set_minor_locator(dates.MonthLocator([1, 4, 7, 10]))
plt.show()

Calendar['Tenor'] = (Calendar['Tenor'] - t_0)/dt.timedelta(days=1)/360
Calendar.plot('Tenor', 'New ZCB Prices', lw=2, colormap='jet', marker='.', markersize=10, title='ZCB Price Curve Bootstrap Method')
plt.xlabel("Time to Maturity (T - to)")
plt.ylabel("Price [P(T0, T)]")
ax.xaxis.set_major_formatter(dates.DateFormatter('\n%m-%Y'))
ax.xaxis.set_minor_locator(dates.MonthLocator([1, 4, 7, 10]))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.show()