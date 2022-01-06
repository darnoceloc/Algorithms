import numpy as np
from scipy.optimize import minimize, fmin
from scipy import optimize
from scipy.stats import norm


periodLength = 0.25

periods = np.array([0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.])/2.0
forwardRates = [0.06, 0.08, 0.09, 0.10, 0.10, 0.10, 0.09, 0.09]

# maturities: 1Y, 2Y, 3Y, 4Y; semi-annual cashflows; first reset date at 6m
capMarketPrices = [0.002, 0.008, 0.012, 0.016]

simpleRates = [forwardRates[0]]
for i in range(1, len(periods)):
	simpleRates.append((((1 + forwardRates[i])**(periodLength)) * ((1 + simpleRates[i - 1])**(periods[i-1]))**(1/periods[i])) - 1)
		
discountFactors = []
discountFactors.append(1/((periods[0]*forwardRates[0]) + 1))
for i in range(1, len(periods)):
	discountFactors.append(discountFactors[i-1] / (1. + (periodLength * forwardRates[i])))
	
def atmStrikeRate(years, freq):
	# From the cap-floor parity, we infer that ATM caps strike rates
	# are equal to a swap rate with the same tenor structure
	return (discountFactors[0] - discountFactors[freq * years - 1]) / (periodLength * sum([discountFactors[i] for i in range(1, freq * years)]))

def capletVega(periodIndex, strike, vol):
	return periodLength * discountFactors[periodIndex] * forwardRates[periodIndex] * np.sqrt(periods[periodIndex - 1]) * norm.pdf(d1(periodIndex, strike, vol))

def vega(year, model, freq):
	vol = impliedVol(year, model, freq)
	strike = atmStrikeRate(year)
	return sum([capletVega(periodIndex, strike, vol) for periodIndex in range(1, freq * year)])

def d1(periodIndex, strike, vol):
	return (np.log(forwardRates[periodIndex] / strike) +  0.5 * vol**2 * periods[periodIndex - 1]) / (vol * np.sqrt(periods[periodIndex - 1])) 

def d2(periodIndex, strike, vol):
	return d1(periodIndex, strike, vol) - vol * np.sqrt(periods[periodIndex - 1])

def capletPrice(periodIndex, strike, vol):
	return periodLength * discountFactors[periodIndex] * (forwardRates[periodIndex] * norm.cdf(d1(periodIndex, strike, vol)) - strike * norm.cdf(d2(periodIndex, strike, vol)))

def D_bach(periodIndex, strike, vol):
	return (forwardRates[periodIndex] - strike)/(vol*np.sqrt(periods[periodIndex-1]))

def caplPrice(periodIndex, strike, vol):
	return  periodLength * discountFactors[periodIndex] * vol * np.sqrt(periods[periodIndex-1]) * (D_bach(periodIndex, strike, vol) * norm.cdf(D_bach(periodIndex, strike, vol)) + norm.pdf(D_bach(periodIndex, strike, vol)))

def capPrice(year, strike, vol, model, freq):
	if model == 'Black':
		return sum([capletPrice(periodIndex, strike, vol) for periodIndex in range(1, freq * year)])
	if model == 'Bachelier':
		return sum([caplPrice(periodIndex, strike, vol) for periodIndex in range(1, freq * year)])

def impliedVol(year, model, freq):
    strike = atmStrikeRate(year, freq)
    f = lambda vol: capPrice(year, strike, vol, model, freq) - 0.01
    return optimize.brentq(f, -2, 2)

print(impliedVol(2, 'Black', 4), 'Black')
print(capPrice(2, atmStrikeRate(2, 4), impliedVol(2, 'Bachelier', 4), 'Bachelier', 4))
print(impliedVol(2, 'Bachelier', 4), 'Bachelier')
print(capPrice(2, atmStrikeRate(2, 4), 0.141, 'Black', 4))
vegas = [(vega(year, 'Black', 2)) for year in range(1, 5)]
impliedVols = [(impliedVol(year, 'Black', 2)) for year in range(1, 5)]

def hjmVol(beta, v, maturity):
	return v**2 * (1 + np.exp(-beta) - 2*np.exp(-0.5*beta)) * np.exp(-2*beta*(maturity-0.5)) * (1/(2*beta)) * (np.exp(2*beta*(maturity-0.5)) - 1)

def modelD1(periodIndex, strike, beta, v):
	return (np.log((1 + periodLength * strike) * discountFactors[periodIndex]/discountFactors[periodIndex - 1]) + 0.5 * hjmVol(beta, v, periods[periodIndex])) / np.sqrt(hjmVol(beta, v, periods[periodIndex]))

def modelD2(periodIndex, strike, beta, v):
	return (np.log((1 + periodLength * strike) * discountFactors[periodIndex]/discountFactors[periodIndex - 1]) - 0.5 * hjmVol(beta, v, periods[periodIndex])) / np.sqrt(hjmVol(beta, v, periods[periodIndex]))

def modelCapletPrice(periodIndex, strike, beta, v):
	return discountFactors[periodIndex - 1]*norm.cdf(-modelD2(periodIndex, strike, beta, v)) - (1 + periodLength * strike) * discountFactors[periodIndex]*norm.cdf(-modelD1(periodIndex, strike, beta, v))

def modelCapPrice(year, beta, v):
	strike = atmStrikeRate(year)
	return sum([modelCapletPrice(periodIndex, strike, beta, v) for periodIndex in range(1, 2 * year)])

def calibrateModel():
	f = lambda p: sum([((1/vegas[year-1]**2) * (modelCapPrice(year, p[0], p[1]) - capMarketPrices[year - 1])**2) for year in range(1, 5)])
	return minimize(f, [1.5, 0.05],  tol=1e-19, method='BFGS')

print('vegas', vegas)
print('impvols', impliedVols)
print('atmstrikes',[atmStrikeRate(i) for i in range(1, 5)])
print('zcb', discountFactors)
print(calibrateModel())