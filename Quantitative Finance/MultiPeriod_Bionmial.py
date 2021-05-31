"""
# T: maturity
# n: # option periods
# N: # futures periods
# S: initial stock price
# r: continuously-compounded interest rate
# c: dividend yield
# sigma: annualized volatility 
# K: strike price
# cp: +1/-1 with regards to call/put
"""

from __future__ import division
from math import exp, sqrt
import numpy as np
import math

T = 0.25
n = 15 # option periods
N = 15 # futures periods
S = 100 #initial stock price
r = 0.02 #continuously-compounded interest rate
c = 0.01 #dividend yield
sigma = 0.3 #annualized volatility 
K = 110 #strike price
cp = -1 #with regards to call/put


def Parameter(T,n,sigma,r,c):

    """Parameter calculation"""    

    dt = T/n
    u = exp(sigma * sqrt(dt))
    d = 1/u
    
    q1 = (exp((r-c)*dt)-d)/(u-d)
    q2 = 1-q1
    R = exp(r*dt)
    
    return (u, d, q1, q2, R)

# =============================================================================

def GenerateTree(T,n,S,sigma,r,c):
    """generate stock tree"""    
    u, d, q1, q2, R = Parameter(T,n,sigma,r,c)
    
    stockTree = np.zeros((n+1, n+1))  
    
    # compute the stock tree
    stockTree[0,0] = S
    for i in range(1,n+1):
        stockTree[0,i] = stockTree[0, i-1]*u
        for j in range(1,n+1):
            stockTree[j,i] = stockTree[j-1, i-1]*d
    
    return stockTree

# =============================================================================

def StockOptionAM(T,n,S,r,c,sigma,K,cp):
    """first return: American Stock Option Pricing"""
    """second return: when is the earliest time to exercise""" 
    """Though it's never optimal to early exercise AM call"""
    """It matters for AM put"""
          
    u, d, q1, q2, R = Parameter(T,n,sigma,r,c)
    
    stockTree = GenerateTree(T,n,S,sigma,r,c)
    optionTree = np.zeros((n+1,n+1))

    
    # compute the option tree
    for j in range(n+1):
        optionTree[j, n] = max(0, cp * (stockTree[j, n]-K))
        
    flag = 0 
    list = []
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            optionTree[j, i] = max((q1 * optionTree[j, i+1] + q2 * optionTree[j+1, i+1])/R,   
                               cp * (stockTree[j, i] - K))                        
            if (optionTree[j, i] - cp * (stockTree[j, i] - K)) < 1e-10:
                flag += 1
                list.append(i)
    
    when = n
    if(flag):  when = list[-1]

    print(optionTree, when)
    return (optionTree[0,0], when)

   
z = StockOptionAM(T,n,S,r,c,sigma,K,cp)

option_maturity = 10

class bs_bin_tree:

    def __init__(self,T,s0,r,sigma,c,K,n):
        self.T = T
        self.r = r
        self.c = c
        self.sigma = sigma
        self.K = K
        self.s0 = s0
        self.n = n
        self.u = math.exp(self.sigma*np.sqrt(self.T/self.n))
        self.q = (math.exp((self.r-self.c)*T/self.n)-(1/self.u))/(self.u-(1/self.u))
        self.R = math.exp(self.r*self.T/self.n)
        self.__print_param__()

    def __print_param__(self):
        print('Time',self.T)
        print('Starting Price',self.s0)
        print('r',self.r)
        print('volatility',self.sigma)
        print('dividend yield',self.c)
        print('strike',self.K)
        print('# period',self.n)
        
    
    def generate_price(self):
        arr=[[self.s0]]
        for i in range(self.n):
            arr_to_add=[]
            for j in range(len(arr[i])):
                arr_to_add.append(arr[i][j]/self.u)
                if j == (len(arr[i])-1):
                    arr_to_add.append(arr[i][j]*self.u)
            arr.append(arr_to_add)
        return arr

    def neutral_pricing(self,p1,p2):
        price = ((1-self.q)*p1 + (self.q)*p2)/self.R
        return price

    def eu_put(self):
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(self.K-arr_rev[i][j],0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    #a = max(arr_rev[i][j]-strike,0)
                    #a = max(a,price)
                    a = price
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return res[::-1]

    def eu_call(self):
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(arr_rev[i][j]-self.K,0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    #a = max(arr_rev[i][j]-strike,0)
                    #a = max(a,price)
                    a = price
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return res[::-1]

    def us_call(self):
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(arr_rev[i][j]-self.K,0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    a1 = max(arr_rev[i][j]-self.K,0)
                    a = max(a1,price)
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return res[::-1]

    def us_call_price(self):
        return self.us_call()[0][0]

    def us_put(self):
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(self.K-arr_rev[i][j],0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    a1 = max(self.K - arr_rev[i][j],0)
                    a = max(a1,price)
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return res[::-1]

    def us_put_price(self):
        return self.us_put()[0][0]

    def us_put_early_ex(self):
        early_ex = False
        early_ex_earning = 0
        early_ex_time = self.n
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(self.K-arr_rev[i][j],0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    a1 = max(self.K-arr_rev[i][j],0)
                    if a1 > price:
                        if early_ex_time == self.n - i:
                            early_ex_earning = max(early_ex_earning,a1)
                        else:
                            early_ex_earning = a1
                        early_ex =True
                        early_ex_time = self.n - i


                    a = max(a1,price)
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return {early_ex_time:early_ex_earning} if early_ex == True else False

    def us_put_call_parity(self):
        LHS = self.us_put_price() + self.s0 * math.exp(-self.c * self.T)
        RHS = self.us_call_price() + self.K * math.exp(-self.r * self.T)
        print('Put Side',LHS)
        print('Call Side',RHS)
        return LHS==RHS

    def generate_future_price(self):
        arr = self.generate_price()
        arr_rev = arr[::-1]
        res=[]
        for i in range(len(arr_rev)):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    res_to_add.append(arr_rev[i][j])
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])*self.R
                    res_to_add.append(price)
                
            res.append(res_to_add)
        return res[::-1]

    def option_on_future(self,option_maturity):
        arr = self.generate_future_price()[0:option_maturity+1]
        arr_rev = arr[::-1]
        res=[]
        for i in range(option_maturity+1):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(arr_rev[i][j]-self.K,0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    a1 = max(arr_rev[i][j]-self.K,0)
                    a = max(a1,price)
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return res[::-1]

    def option_price_on_future(self,option_maturity):
        return self.option_on_future(option_maturity)[0][0]

    def option_on_future_early_ex(self,option_maturity):
        arr = self.generate_future_price()[0:option_maturity+1]
        arr_rev = arr[::-1]
        res=[]
        early_ex = False
        early_ex_earning = 0
        early_ex_time = self.n
        for i in range(option_maturity+1):
            res_to_add = []
            for j in range(len(arr_rev[i])):
                if i == 0:
                    a = max(arr_rev[i][j]-self.K,0)
                    res_to_add.append(a)
                else:
                    price = self.neutral_pricing(res[i-1][j], res[i-1][j+1])
                    a1 = max(arr_rev[i][j]-self.K,0)
                    if a1 > price:
                        if early_ex_time == option_maturity - i:
                            early_ex_earning = max(early_ex_earning,a1)
                        else:
                            early_ex_earning = a1
                        early_ex =True
                        early_ex_time =  len(arr_rev) - i -1


                    a = max(a1,price)
                    res_to_add.append(a)
                
            res.append(res_to_add)
        return {early_ex_time:early_ex_earning} if early_ex == True else False

    def nCr(self,n,r):
        f = math.factorial
        return f(n) / f(r) / f(n-r)
    
    def chooser_option_price(self,option_expire):
        call = self.eu_call()[option_expire]
        put = self.eu_put()[option_expire]
        res=[]
        for i in range(len(call)):
            res.append(max(call[i],put[i]))
        result=0
        for j in range(0,len(res)):
            result += self.nCr(option_expire,j)* (self.q**(j)) * (1-self.q)**(option_expire-j) * res[j]
        return (result/self.R**(option_expire))

tree = bs_bin_tree(T, 100, r, sigma, c, K, n)
print(tree.us_call())
print(tree.us_call_price())
print(tree.us_put())
print(tree.us_put_price())
print(tree.us_put_early_ex())
print(tree.us_put_call_parity())
print(tree.option_on_future(option_maturity))
print(tree.option_price_on_future(option_maturity))
print(tree.option_on_future_early_ex(option_maturity))
print(tree.chooser_option_price(10))
