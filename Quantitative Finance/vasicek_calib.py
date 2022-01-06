import numpy as np
import QuantLib as ql
from collections import namedtuple
import math


def create_swaption_helpers(data_, index_, term_structure_, engine_):
    swaptions_ = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data_:
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(d.volatility))
        helper = ql.SwaptionHelper(ql.Period(d.start, ql.Months),
                                   ql.Period(d.length, ql.Months),
                                   vol_handle,
                                   index_,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure_,
                                   ql.BlackCalibrationHelper.RelativePriceError,
                                   ql.nullDouble(),
                                   1.,
                                   ql.Normal,
                                   0.
                                   )
        helper.setPricingEngine(engine_)
        swaptions_.append(helper)
    return swaptions_


def calibration_report(swaptions_, data_):
    print("-" * 82)
    print("%15s %15s %15s %15s %15s" % \
          ("Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error"))
    print("-" * 82)
    cum_err = 0.0
    for i, s in enumerate(swaptions_):
        model_price = s.modelValue()
        market_vol = data_[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price / black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-6, 500, 0.0, 0.50)
        rel_error2 = implied_vol / market_vol - 1.0
        cum_err += rel_error2 * rel_error2

        print("%15.5f %15.5f %15.5f %15.5f %15.5f" % \
              (model_price, black_price, implied_vol, market_vol, rel_error))
    print("-" * 82)
    print("Cumulative Error : %15.5f" % math.sqrt(cum_err))


today = ql.Date().todaysDate()
# ql.Settings.instance().evaluationDate = today
# crv = ql.ZeroCurve([today, settlement], [0.05, 0.05], ql.Actual365Fixed())
crv = ql.FlatForward(today, 0.05, ql.Actual365Fixed())
yts = ql.YieldTermStructureHandle(crv)
vol = ql.QuoteHandle(ql.SimpleQuote(0.1))
model = ql.Vasicek(r0=0.05, a=0.2, b=0.05, sigma=0.1)
engine = ql.BlackCallableFixedRateBondEngine(vol, yts)
# engine = ql.JamshidianSwaptionEngine(model, yts)
index = ql.Euribor1Y(yts)

CalibrationData = namedtuple("CalibrationData", 
                             "start, length, volatility")

data = [CalibrationData(0, 1, 0.1), CalibrationData(1, 1, 0.1), CalibrationData(3, 1, 0.1), CalibrationData(4, 1, 0.1)]
swaptions = create_swaption_helpers(data, index, yts, engine)

optimization_method = ql.LevenbergMarquardt(1.0e-8, 1.0e-8, 1.0e-8)
end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
model.calibrate(swaptions, optimization_method, end_criteria)

calibration_report(swaptions, data)
a, b, sigma, lam = model.params()
print('%6.5f' % a, '%6.5f' % b, '%6.5f' % sigma, '%6.5f' % lam)
