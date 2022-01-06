import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.integrate as sciIntegr
from scipy import integrate, optimize
from scipy.integrate._ivp.dop853_coefficients import B
from scipy.optimize import minimize
from scipy.stats import norm

## Set model parameters
r0 = 0.08
theta = 0.09 #mean reversion level
k = 0.86 #speed of mean reversion
sigma = 0.0148 #volatility
phi = -1
S = 1.0 #ZC bond maturity
t = 0 #valuation date
T = 1 #ZC option maturity

#Zero-coupon bond prices in Vasicek model, Brigo, IRM, p59
class short_rate_model(object):
    def __init__(self, r0, kappa, theta, sigma, tau, model):
        self.r0 = r0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.tau = tau
        self.model = model

    def zcb_price(self):
        if (model == 'Vasicek'):
            self.B = (1/self.kappa)*(1-np.exp(-self.kappa*self.tau))
            self.A = (self.theta - 0.5*self.sigma**2/self.kappa**2)*(self.B - self.tau) - (self.sigma**2) / (4*self.kappa) * (self.B**2)
            n2 = len(self.A)
            A_ = self.A
            B_ = self.B
            r_ = np.repeat(self.r0, n2)
            p = np.exp(A_- B_*r_)
            return p
        if (model == 'CIR'):
            g = np.sqrt(self.kappa**2 + 2*self.sigma**2)
            tmp = 2*self.kappa*self.theta/self.sigma**2
            tmp1 = self.kappa*(self.tau)/2
            tmp2 = g*(self.tau)/2

            self.A = tmp*np.log(np.exp(tmp1)/ (np.cosh(tmp2) + (self.kappa/g)*np.sinh(tmp2)))
            tanh = np.tanh(g*(self.tau)/2)
            self.B = 2. * tanh / (self.kappa*tanh + g)
            n2 = len(self.A)
            A_ = self.A
            B_ = self.B
            r_ = np.repeat(self.r0, n2)
            p = np.exp(A_- B_*r_)
            return p
        else:
            print('zcb model must be Vasicek or CIR')
            return -1
    
    def short_rate(self):
        P = self.zcb_price()
        r = (self.A - np.log(P))/self.B
        return r

def swapRates(tau, p, mat):
    tmax = mat[-1]

    ttemp = np.arange(0.5, tmax+0.5, 0.5)
    ptemp = np.interp(ttemp, tau, p)

    dis = np.cumsum(ptemp)

    pmat = np.interp(mat, tau, p)

    index = (2*mat).astype(int) - 1
    S = 100 * 2 * (1 - pmat)/dis[index]

    return S

def liborRates(tau, p, mat):
    pmat = np.interp(mat, tau, p)
    L = 100*(1. / pmat- 1)/mat
    return L

def objFunc1(params, tau, LIBOR, SWAP, model):
    r0 = params[0]
    kappa = params[1]
    theta = params[2]
    sigma = params[3]

    p = short_rate_model(r0, kappa, theta, sigma, tau, model).zcb_price()
    S = swapRates(tau, p, SWAP[:, 0])
    L = liborRates(tau, p, LIBOR[:, 0])

    rel1 = (S - SWAP[:, 1])/ SWAP[:, 1]
    rel2 = (L - LIBOR[:, 1])/ LIBOR[:, 1]
    mae = np.sum(rel1**2) + np.sum(rel2**2)

    return mae

def calibration_func(fun, param_0, tau, LIBOR, SWAP, model):
    opt = {'maxiter':10000, 'maxfev':5e4}
    sol = minimize(fun, param_0, args=(tau, LIBOR, SWAP, model), method='Nelder-Mead', options=opt)
    print(sol.message)
    par = np.array(sol.x)
    print('parameters =' + str(par))
    r_star = par[0]
    kappa_star = par[1]
    theta_star = par[2]
    sigma_star = par[3]
    p = short_rate_model(r_star, kappa_star, theta_star, sigma_star, tau, model).zcb_price()
    L = liborRates(tau, p, LIBOR[:, 0])
    S = swapRates(tau, p, SWAP[:, 0])
    return p, L, S

LIBOR = np.array([
    [1/12, 1.49078, 2.2795],
    [2/12, 1.52997, 2.33075],
    [3/12, 1.60042, 2.43631],
    [6/12, 1.76769, 2.63525],
    [12/12, 2.04263, 2.95425]
])
SWAP = np.array([
    [2, 2.013, 3.0408],
    [3, 2.1025, 3.1054],
    [5, 2.195, 3.1332],
    [7, 2.2585, 3.1562],
    [10, 2.3457, 3.199],
    [15, 2.4447, 3.2437],
    [30, 2.5055, 3.227]
])

T = np.arange(0, 30 + 1/12, 1/12)
params = np.array([r0, k, theta, sigma])

params1 = np.array([0.25, 5, 0.2, 0.1])
params2 = np.array([0.25, 5, 0.2, 0.1])

model = 'Vasicek'

whichOne = 1
p11, L11, S11 = calibration_func(objFunc1, params1, T, LIBOR[:, [0, whichOne]], SWAP[:, [0, whichOne]], model)
whichOne = 2
p12, L12, S12 = calibration_func(objFunc1, params1, T, LIBOR[:, [0, whichOne]], SWAP[:, [0, whichOne]], model)

model = 'CIR'
whichOne = 1
p21, L21, S21 = calibration_func(objFunc1, params1, T, LIBOR[:, [0, whichOne]], SWAP[:, [0, whichOne]], model)
whichOne = 2
p22, L22, S22 = calibration_func(objFunc1, params1, T, LIBOR[:, [0, whichOne]], SWAP[:, [0, whichOne]], model)

fig = plt.figure()
plt.scatter(LIBOR[:, 0], LIBOR[0:, 1], c=np.sqrt(LIBOR[:, 0]), label='Market')
plt.plot(LIBOR[:, 0], L11, 'o--', label='Vasicek')
plt.plot(LIBOR[:, 0], L21, 'r--', label='CIR')
plt.xlabel('Maturities (Years)')
plt.ylabel('LIBOR Rates')
plt.legend()
plt.title('Model vs. Market Rates')
plt.show()
plt.scatter(LIBOR[:, 0], LIBOR[0:, 2], c=np.sqrt(LIBOR[:, 0]), label='Market')
plt.plot(LIBOR[:, 0], L12, 'o--', label='Vasicek')
plt.plot(LIBOR[:, 0], L22, 'r--', label='CIR')
plt.xlabel('Maturities (Years)')
plt.ylabel('LIBOR Rates')
plt.legend()
plt.title('Model vs. Market Rates')
plt.show()
plt.scatter(SWAP[:, 0], SWAP[0:, 1], c=np.sqrt(SWAP[:, 0]), label='Market')
plt.plot(SWAP[:, 0], S11, 'o--', label='Vasicek')
plt.plot(SWAP[:, 0], S21, 'r--', label='CIR')
plt.xlabel('Maturities (Years)')
plt.ylabel('Swap Rates')
plt.legend()
plt.title('Model vs. Market Rates')
plt.show()
plt.scatter(SWAP[:, 0], SWAP[0:, 2], c=np.sqrt(SWAP[:, 0]), label='Market')
plt.plot(SWAP[:, 0], S12, 'o--', label='Vasicek')
plt.plot(SWAP[:, 0], S22, 'r--', label='CIR')
plt.legend()
plt.xlabel('Maturities (Years)')
plt.ylabel('Swap Rates')
plt.title('Model vs. Market Rates')
plt.show()
plt.plot(T, p11, 'r--', label='Vasicek')
plt.scatter(T, p21, label='CIR')
plt.legend()
plt.xlabel('Maturities (Years)')
plt.ylabel('ZCB Prices')
plt.title('Vasicek vs. CIR ZCB Prices')
plt.show()
plt.plot(T, p12, 'r--', label='Vasicek')
plt.scatter(T, p22, label='CIR')
plt.legend()
plt.xlabel('Maturities (Years)')
plt.ylabel('ZCB Prices')
plt.title('Vasicek vs. CIR ZCB Prices')
plt.show()
plt.plot(T, p11, 'r--', label='12-14-17')
plt.scatter(T, p12, label='10-11-18')
plt.legend()
plt.xlabel('Maturities (Years)')
plt.ylabel('ZCB Prices')
plt.title('Vasicek ZCB Prices')
plt.show()
plt.plot(T, p21, 'r--', label='12-14-17')
plt.scatter(T, p22, label='10-11-18')
plt.legend()
plt.xlabel('Maturities (Years)')
plt.ylabel('ZCB Prices')
plt.title('CIR ZCB Prices')
plt.show()

periods = np.linspace(0.25, 30, 120)
discountFactors = short_rate_model(r0, k, theta, sigma, periods, 'Vasicek').zcb_price()
periodLength = 0.25

forwardRates = []
for i in range(len(discountFactors) - 1):
    forwardRates.append((1/periodLength)*((discountFactors[i]/discountFactors[i+1]) - 1))
print(forwardRates)
print(len(forwardRates))
print(len(discountFactors))

print(short_rate_model(0.06, 0.86, 0.08, 0.01, np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25]), 'Vasicek').zcb_price())
print(short_rate_model(0.06, 0.86, 0.08, 0.01, np.array([0, 1.25]), 'Vasicek').zcb_price())

print(short_rate_model(0.06, 0.86, 0.08, np.exp(-0.86)*0.01, np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25]), 'Vasicek').zcb_price())
print(short_rate_model(0.06, 0.86, 0.08, np.exp(-0.86*1.25)*0.01, np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25]), 'Vasicek').zcb_price())

def atmStrikeRate():
	# From the cap-floor parity, we infer that ATM caps strike rates are equal to a swap rate with the same tenor structure
	return (discountFactors[0] - discountFactors[-1]) / (periodLength * sum(discountFactors[1:]))

def d1(periodIndex, strike=atmStrikeRate(), sigma=0.0148):
    vol = np.exp(strike*periods[periodIndex])*sigma
    return (np.log(forwardRates[periodIndex] / strike) +  (0.5 * vol**2 * periods[periodIndex-1])) / (vol * np.sqrt(periods[periodIndex-1])) 

def d2(periodIndex, strike=atmStrikeRate(), sigma=0.0148):
    vol = np.exp(strike*periods[periodIndex])*sigma
    return d1(periodIndex, strike, vol) - vol * np.sqrt(periods[periodIndex-1])

def capletPrice(periodIndex, strike=atmStrikeRate(), sigma=0.0148):
    vol = np.exp(strike*periods[periodIndex])*sigma
    return periodLength * discountFactors[periodIndex+1] * (forwardRates[periodIndex] * norm.cdf(d1(periodIndex, strike, vol)) - strike * norm.cdf(d2(periodIndex, strike, vol)))

def capPrice(num):
    return sum([capletPrice(p) for p in range(1, num-1)])

def d1_p(periodIndex, strike=atmStrikeRate(), vol=0.0148):
    strike = discountFactors[periodIndex]/discountFactors[periodIndex-1]
    vol = np.exp(-strike*periods[periodIndex])*vol
    fact = (vol**2/strike**2) * (np.exp(-strike*periods[periodIndex-1]) - np.exp(-strike*periods[periodIndex]))**2*(np.exp(2*strike*periods[periodIndex-1])-1)/(2*strike)
    return (np.log(discountFactors[periodIndex] / strike*discountFactors[periodIndex-1]) +  (0.5 * fact))/np.sqrt(fact)

def d2_p(periodIndex, strike=atmStrikeRate(), vol=0.0148):
    strike = discountFactors[periodIndex]/discountFactors[periodIndex-1]
    vol = np.exp(-strike*periods[periodIndex])*vol
    fact = (vol**2/strike**2) * (np.exp(-strike*periods[periodIndex-1]) - np.exp(-strike*periods[periodIndex]))**2*(np.exp(2*strike*periods[periodIndex-1])-1)/(2*strike)
    return (np.log(discountFactors[periodIndex] / strike*discountFactors[periodIndex-1]) -  (0.5 * fact))/np.sqrt(fact)

def capletPrice_p(periodIndex, strike=atmStrikeRate(), vol=0.0148):
    strike = discountFactors[periodIndex]/discountFactors[periodIndex-1]
    vol = np.exp(-strike*periods[periodIndex])*vol
    return ((strike * discountFactors[periodIndex] * norm.cdf(-d2_p(periodIndex, strike, vol))) - discountFactors[periodIndex+1]*norm.cdf(-d1_p(periodIndex, strike, vol)))*(1 + periodLength*strike)

def capPrice_p(num):
    return sum([capletPrice_p(p) for p in range(1, num-1)])

print('zcbs', discountFactors)
print('atmstrike', atmStrikeRate())
print('caplets', [capletPrice(p) for p in range(1, len(periods)-1)])
print(capletPrice(44))
print(capletPrice(45))
print('final cap', capPrice(120))

class Payoff(object):
    def __init__(self, K):
        self.K = K
    def f(self):
        pass
    
class ZCBOption(Payoff):
    def f(self, S):
        return np.maximum(S - self.K, 0)

class ZCBValue(Payoff):
    def f(self, S):
        return S


class Bond(object):
    def __init__(self, theta, kappa, sigma, r0=0.05):
        '''
        r0    is the current level of rates
        kappa is the speed of convergence
        theta is the long term rate
        sigma is the volatility    
        '''
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.r0 = r0
        self.dt = 0.0001
    
    def B(self, t, T):
        pass
    
    def A(self, t, T): 
        pass
    
    def Exact_zcb(self, t, T):
        pass
    
    def Euler(self, M, I, T):
        pass
    
    def Yield(self, t, T, rate):
        return -1/(T-t)*self.A(t, T) + 1/(T-t)*self.B(t, T)*rate
    
    def SpotRate(self, t, T):
        price = self.Exact_zcb(t, T)
        time = T - t
        return (-np.log(price)/time)
    
    # Different ways to calculate the PV and cc Forward Rates 
    # 1    
    def ForwardRate1(self, time):
        up   = np.max(time + self.dt, 0)
        down = np.max(time - self.dt, 0)

        r = self.SpotRate(0, time)
        dr = self.SpotRate(0, up) - self.SpotRate(0, down)
        dr_dt = dr/(2*self.dt)
        fwd_rate = r + time * dr_dt
        return fwd_rate
    def ZCB_Forward_Integral1(self, t, T):    
        val = integrate.quad(self.ForwardRate1, t, T)[0]
        return np.exp(-val)
    # 2
    def ForwardRate2(self, time):            
        up   = np.max(time + self.dt, 0)
        down = np.max(time - self.dt, 0)

        dP = self.Exact_zcb(0, down) - self.Exact_zcb(0, up)
        dP_dt = dP/(2*self.dt)
        P = self.Exact_zcb(0, time)
        fwd_rate = dP_dt/P
        return fwd_rate

    def ZCB_Forward_Integral2(self, t, T):                    
        val = integrate.quad(self.ForwardRate2, t, T)[0]
        return np.exp(-val)
    # 3
    def ForwardRate3(self, time):            
        up = np.max(time + self.dt, 0)
        down = np.max(time - self.dt, 0)

        dP = np.log(self.Exact_zcb(0, down)) - np.log(self.Exact_zcb(0, up))     
        fwd_rate = dP/(2*self.dt)
        return fwd_rate 
    
    def ZCB_Forward_Integral3(self, t, T):                    
        val = integrate.quad(self.ForwardRate3, t, T)[0]
        return np.exp(-val)
    
    # Pricing with a sde
    def StochasticPrice(self, VectorRates, VectorTime):  
        # VectorRates is a two dimensional array:
        # with simulated rates in columns and timesteps rates in rows
        
        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0        
        VectorRates = VectorRates[:, :]
        VectorTime  = VectorTime[:]
        
        No_Sim = VectorRates.shape[0]
         
        price = np.zeros(No_Sim)
        for i in range(No_Sim):    
            Rates = VectorRates[i, :]
            price[i] = np.exp(-(sciIntegr.simpson(Rates, VectorTime)))
        
        RangeUp_Down = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean = np.mean(price)
                
        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down
        
    def FutureZCB(self, M, I, T_0, T_M, Bond):
        
        self.Euler(M, I, T_0)        
        bond = Bond    
        
        # we do not need VectorRates and VectorTime at the beginning of the simulation as it is r0        
        VectorRates = self.rates[:, :]
        VectorTime  = self.times[:]
        
        No_Sim = VectorRates.shape[0]
         
        price = np.zeros(No_Sim)
        priceUntil_FO = np.zeros(No_Sim)
        Payoff = np.zeros(No_Sim)
        R0 = np.zeros(No_Sim)
        for i in range(No_Sim):    
            Rates = VectorRates[i, :]
            R0[i] = Rates[-1]
            Yield = self.Yield(T_0, T_M, Rates[-1])
            ZCBPrice = np.exp(-Yield*(T_M-T_0))
            Payoff[i] = bond.f(ZCBPrice)
            price[i] = np.exp(-(sciIntegr.simpson(Rates, VectorTime)))*Payoff[i]
            priceUntil_FO[i] = np.exp(-(sciIntegr.simpson(Rates, VectorTime)))
        
        RangeUp_Down = np.sqrt(np.var(price))*1.96 / np.sqrt(No_Sim)
        Mean = np.mean(price)
        MeanRate = np.mean(R0)
        FWDValue = np.mean(Payoff)
        MeanValueUntil_FO = np.mean(priceUntil_FO)
        return Mean,  Mean + RangeUp_Down, Mean - RangeUp_Down, FWDValue, MeanValueUntil_FO, MeanRate
    
    def ExpectedRate(self, t, T):
        pass
    
    def VarianceRate(self, t, T):
        pass

    def PlotEulerSim(self, Text, No_of_Sim):
        # We plot the first No_of_Sim simulated paths
        [plt.plot(self.times, self.rates[k], lw=1.5) for k in range(No_of_Sim)]
        plt.xlabel('time - yrs')  
        plt.ylabel('rates level')
        plt.grid(True)
        plt.title(Text + ' - the first {}'.format(No_of_Sim) + " Simulated Paths")
        plt.show()

    def PlotEulerSim_Stats(self, Text):        
        # We plot and compare the average simulation +-2 sd of all time steps vs. what we expect from the model
        
        SimAverage  = [np.mean(self.rates[:,i]) for i in range(len(self.rates[0, :]))]
        print("SimAvg", SimAverage[-1])
        print("Simmean", np.mean(SimAverage))
        SimSD = [np.std(self.rates[:,i]) for i in range(len(self.rates[0, :]))]
        print("SimSD", SimSD[-1])
        AnalyAverage = self.ExpectedRate(self.r0)
        print("AnalyAvg", AnalyAverage[-1])
        print("Analymean", np.mean(AnalyAverage))
        AnalySD  = np.asarray([np.sqrt(self.VarianceRate(0, i)) for i in self.times])
        plt.plot(self.times, SimAverage, lw=1.5, label='Sim Mean', linestyle=':')
        #plt.plot(self.times, SimAverage + 2*SimSD, lw=1.5, label ='Sim Mean + 2*SD',linestyle=':')
        #plt.plot(self.times, SimAverage - 2*SimSD, lw=1.5, label ='Sim Mean - 2*SD',linestyle=':')
        plt.plot(self.times, AnalyAverage, lw=1.5, label='Analy Mean', linestyle='-')
        plt.plot(self.times, AnalySD, lw=1.5, label='Analy SD', linestyle='-')
        # plt.plot(self.times, SimSD, lw=1.5, label='Sim SD', linestyle='-')
        #plt.plot(self.times, AnalyAverage + 2*AnalySD, lw=1.5, label ='Analy Mean + 2*SD',linestyle='-.')
        #plt.plot(self.times, AnalyAverage - 2*AnalySD, lw=1.5, label ='Analy Mean - 2*SD',linestyle='-.')
                
        plt.legend()
        plt.xlabel('time - yrs')  
        plt.ylabel('rates')
        plt.grid(True)        
        plt.title(Text)
        plt.show()


class Vasicek(Bond):
    def __init__(self, theta, kappa, sigma, M_, N_, tau_, r0=0.05):
        Bond.__init__(self, theta, kappa, sigma, r0)
        self.times = np.linspace(0, tau_, num=M_+1)
        self.rates = np.zeros((N_, M_+1))
        self.M = M_
        self.N = N_
        self.tau = tau_
        self.dt = tau_ / float(M_)

    def B(self, t, T):
        return (1 - np.exp(-self.kappa*(T-t))) / self.kappa
    
    def A(self, t, T): 
        return ((self.theta-(self.sigma**2)/(2*(self.kappa**2))) *(self.B(t, T)-(T-t)) \
                   -  (self.sigma**2)/(4*self.kappa)*(self.B(t, T)**2))
                
    def Exact_zcb(self, t, T):      
        B = self.B(t, T)
        A = self.A(t, T)
        return np.exp(A-self.r0*B)

    def Euler(self, M, N, tau):
        # I is the number of simulation, M is the number of time steps until maturity 
        dt = np.linspace(0, tau, M+1)
        xh = np.zeros((M+1))
        xh[0] = self.r0
        for i in range(N):
            for t in range(1, M+1):
                xh[t] = xh[t - 1] + (self.kappa * (self.theta - xh[t - 1]))*dt[t] + \
                        (self.sigma * npr.normal(0, 1, 1))
            self.rates[i, :] = xh
        return self.rates
    
    def ExpectedRate(self, r, T=1, M=365):
        dt = np.linspace(0, T, M+1)
        temp1 = np.exp(-self.kappa*dt)
        result = r*temp1 + self.theta*(1 -temp1)
        return result
    
    def VarianceRate(self, t_, T_):
        result = self.sigma**2/(self.kappa * 2) * (1 - np.exp(- 2*self.kappa*(T_- t_)))
        return result

""" Simulate interest rate path by the Vasicek model """


# def vasicek_t(r0, k, theta, sigma, T=1.0, M=365, N=9000, seed=777):
#     np.random.seed(seed)
#     dt = np.linspace(0, T, M+1)    
#     rates = np.zeros([N, M+1])
#     rates[:, 0] = [r0]*N 
#     for i in range(N):
#         for j in range(1, M+1):
#             rates[i, j] = rates[i, j-1] + k*(theta-rates[i,j-1])*dt[j] + sigma*npr.normal(0, 1, 1)
#     return range(M+1), rates

# print("exp_test2", np.mean(vasicek_t(r0, k, theta, sigma)[1][:, -1]), np.std(vasicek_t(r0, k, theta, sigma)[1][:, -1]))
# plt.plot(vasicek_t(r0, k, theta, sigma)[0], [np.mean(vasicek_t(r0, k, theta, sigma)[1][:, i]) for i in range(366)], label='SimAvg')
# plt.show()

I = 8000  # no. of simulations
T = 1
M = T * 365  # trading day per annum

if __name__ == "__main__":
    vasicek = Vasicek(theta, k, sigma, M, I, T, r0)
    npr.seed(1500)
    vasicek.Euler(M, I, T)
    print("future_sr", vasicek.ExpectedRate(r0, T, M)[-1])
    print(vasicek.ForwardRate1(1), vasicek.ForwardRate2(1), vasicek.ForwardRate3(1))
    print(vasicek.ZCB_Forward_Integral1(0, 1), vasicek.ZCB_Forward_Integral2(0, 1), vasicek.ZCB_Forward_Integral3(0, 1))
    print("Volatility", np.sqrt(vasicek.VarianceRate(t, T)))
    print("Analytical ZCB", vasicek.Exact_zcb(t, T))
    vals = vasicek.StochasticPrice(vasicek.rates, vasicek.times)
    print("the MC ZCB price is:", np.round(vals, 4))
    vasicek.PlotEulerSim_Stats("Vasicek: Simulation vs Analytical")
    vasicek.PlotEulerSim("Vasicek: Simulations", I)
    print("Mean1", np.mean(vasicek.rates[:, -2:-1]))
    print("Mean2", np.mean(vasicek.rates))
    npr.seed(704)
    BondValue = ZCBValue(0)  # an instance of class ZCBValue, to model the payoff value at year 1
    vals = vasicek.FutureZCB(M, I, 0, 1, BondValue)
    print("The MC Expected short term rate in", 1, "year is ", np.round(vals[-1], 6))