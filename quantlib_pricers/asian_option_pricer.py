import os
import numpy as np
import QuantLib as ql
from time import time
from joblib import Parallel, delayed


class asian_option_pricer():
    def __init__(self,seed=123):
        self.seed = seed
        if seed == None:
            self.seed = 0
        self.rng = "pseudorandom" # could use "lowdiscrepancy"
        self.numPaths = 100000

    def day_count(self):
        return 

    def asian_option_price(self,
        s,k,r,g,w,
        averaging_type,n_fixings,fixing_frequency,past_fixings,
        kappa,theta,rho,eta,v0
        ):

        s = float(s)
        k = float(k)
        r = float(r)
        g = float(g)
        if w == 'call':
            option_type = ql.Option.Call 
        elif w == 'put':
            option_type = ql.Option.Put
        t = n_fixings*fixing_frequency
        calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        periods = np.arange(fixing_frequency,t+1,fixing_frequency).astype(int)
        fixing_periods = [ql.Period(int(p),ql.Days) for p in periods]
        fixing_dates = [calculation_date + p for p in fixing_periods]
        expiration_date = calculation_date + fixing_periods[-1]
        riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),ql.Actual365Fixed()))
        dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),ql.Actual365Fixed()))

        hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, ql.QuoteHandle(ql.SimpleQuote(s)), v0, kappa, theta, eta, rho)
        hestonModel = ql.HestonModel(hestonProcess)
        vanillaPayoff = ql.PlainVanillaPayoff(option_type, float(k))
        europeanExercise = ql.EuropeanExercise(expiration_date)
        
        if averaging_type == 'geometric':
            geometric_engine = ql.MCDiscreteGeometricAPHestonEngine(hestonProcess, self.rng, requiredSamples=self.numPaths,seed=self.seed)
            geometricAverage = ql.Average().Geometric
            geometricRunningAccumulator = 1.0
            discreteGeometricAsianOption = ql.DiscreteAveragingAsianOption(
                geometricAverage, geometricRunningAccumulator, past_fixings,
                fixing_dates, vanillaPayoff, europeanExercise)
            discreteGeometricAsianOption.setPricingEngine(geometric_engine)
            geometric_price = float(discreteGeometricAsianOption.NPV())
            return geometric_price
            
        else:
            arithmetic_engine = ql.MCDiscreteArithmeticAPHestonEngine(hestonProcess, self.rng, requiredSamples=self.numPaths,seed=self.seed)
            arithmeticAverage = ql.Average().Arithmetic
            arithmeticRunningAccumulator = s
            discreteArithmeticAsianOption = ql.DiscreteAveragingAsianOption(
                arithmeticAverage, arithmeticRunningAccumulator, past_fixings, 
                fixing_dates, vanillaPayoff, europeanExercise)
            discreteArithmeticAsianOption.setPricingEngine(arithmetic_engine)
            arithmetic_price = float(discreteArithmeticAsianOption.NPV())
            return arithmetic_price

    def row_asian_option_price(self,row):
        try:
            tic = time()
            asian_price = self.asian_option_price(
                row['spot_price'],
                row['strike_price'],
                row['risk_free_rate'],
                row['dividend_rate'],
                row['w'],
                row['averaging_type'],
                row['fixing_frequency'],
                row['n_fixings'],
                row['past_fixings'],
                row['kappa'],
                row['theta'],
                row['rho'],
                row['eta'],
                row['v0']
            )
            asian_cpu = time() - tic
            return {'asian_price':asian_price,'asian_cpu':asian_cpu}
        except Exception as e:
            print(f"Error with row: {row}\nException: {e}")
            return {'asian_price':np.nan,'asian_cpu':np.nan}

    def df_asian_option_price(self, df):
        max_jobs = os.cpu_count() // 4
        max_jobs = max(1, max_jobs)
        return Parallel(n_jobs=max_jobs)(delayed(self.row_asian_option_price)(row) for _, row in df.iterrows())
