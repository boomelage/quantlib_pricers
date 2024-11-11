import os
import numpy as np
import QuantLib as ql
from joblib import Parallel, delayed
from scipy.stats import norm

class vanilla_pricer():
	def __init__(self,day_count_name=None,steps=None,rng=None,numPaths=None,seed=1312):
		
		if day_count_name == None:
			self.day_count_name = 'Actual365Fixed'

		elif day_count_name == 'Thirty360.USA':
			self.day_count_name = 'Thirty360.USA'
		else:
			raise(f"{day_count_name} not supported")
		
		self.steps = steps
		if steps == None:
			self.steps = 10

		self.rng = rng
		if rng == None:
			self.rng = "pseudorandom" # could use "lowdiscrepancy"

		self.numPaths = numPaths
		if numPaths == None:
			self.numPaths = 100000
		
		self.seed = seed
		if seed == None:
			self.seed = 0

	def day_count(self):
		if self.day_count_name == 'Actual365Fixed':
			return ql.Actual365Fixed()
		elif self.day_count_name == 'Thirty360.USA':
			return ql.Thirty360(ql.Thirty360.USA)
		else:
			raise(f"{self.day_count_name} not supported")

	def numpy_black_scholes(self, s, k, t, r, volatility,w):
		if w == 'call':
			w = 1
		elif w == 'put':
			w = -1
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")            
		d1 = (np.log(s/k)+(r+volatility**2/2)*t/365)/(volatility*np.sqrt(t/365))
		d2 = d1-volatility*np.sqrt(t/365)
		return max(0,w*(s*norm.cdf(w*d1)-k*np.exp(-r*t/365)*norm.cdf(w*d2)))

	def row_numpy_black_scholes(self, row):
		return self.numpy_black_scholes(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['volatility'],
			row['w']
		)


	def df_numpy_black_scholes(self, df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_numpy_black_scholes)(row) for _, row in df.iterrows())

	def mc_heston_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
		):
		calculation_date = ql.Date.todaysDate()
		s = float(s)
		k = float(k)
		t = int(t)
		r = float(r)
		g = float(g)
		kappa = float(kappa)
		theta = float(theta)
		rho = float(rho)
		eta = float(eta)
		v0 = float(v0)
		ql.Settings.instance().evaluationDate = calculation_date
		expiration_date = calculation_date + ql.Period(t,ql.Days)
		ts_r = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),self.day_count()))
		ts_g = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),self.day_count()))

		if w == 'call':
			option_type = ql.Option.Call
		elif w == 'put':
			option_type = ql.Option.Put
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")

		payoff = ql.PlainVanillaPayoff(option_type, k)
		exercise = ql.EuropeanExercise(expiration_date)
		european_option = ql.VanillaOption(payoff, exercise)

		heston_process = ql.HestonProcess(

		ts_r, ts_g, 

		ql.QuoteHandle(ql.SimpleQuote(s)), 

		v0, kappa, theta, eta, rho

		)
		engine = ql.MCEuropeanHestonEngine(heston_process, self.rng, self.steps, requiredSamples=self.numPaths,seed=self.seed)
		european_option.setPricingEngine(engine)
		return max(european_option.NPV(),0)

	def row_mc_heston_price(self,row):
		return self.mc_heston_price(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['dividend_rate'],
			row['w'],
			row['kappa'],
			row['theta'],
			row['rho'],
			row['eta'],
			row['v0'],
			)

	def df_mc_heston_price(self,df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_mc_heston_price)(row) for _, row in df.iterrows())



	def heston_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
	):
		s = float(s)
		k = float(k)
		t = int(t)
		r = float(r)
		g = float(g)
		kappa = float(kappa)
		theta = float(theta)
		rho = float(rho)
		eta = float(eta)
		v0 = float(v0)

		calculation_date = ql.Date.todaysDate()
		ql.Settings.instance().evaluationDate = calculation_date
		expiration_date = calculation_date + ql.Period(int(t),ql.Days)
		ts_r = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),self.day_count()))
		ts_g = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),self.day_count()))

		if w == 'call':
			option_type = ql.Option.Call
		elif w == 'put':
			option_type = ql.Option.Put
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")

		payoff = ql.PlainVanillaPayoff(option_type, k)
		exercise = ql.EuropeanExercise(expiration_date)
		european_option = ql.VanillaOption(payoff, exercise)

		heston_process = ql.HestonProcess(

		ts_r, ts_g, 

		ql.QuoteHandle(ql.SimpleQuote(s)), 

		v0, kappa, theta, eta, rho

		)

		heston_model = ql.HestonModel(heston_process)

		engine = ql.AnalyticHestonEngine(heston_model)

		european_option.setPricingEngine(engine)

		return max(european_option.NPV(),0)

	def row_heston_price(self,row):
		return self.heston_price(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['dividend_rate'],
			row['w'],
			row['kappa'],
			row['theta'],
			row['rho'],
			row['eta'],
			row['v0'],
			)

	def df_heston_price(self, df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_heston_price)(row) for _, row in df.iterrows())


	def bates_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
		lambda_, nu, delta
	):
		calculation_date = ql.Date.todaysDate()
		ql.Settings.instance().evaluationDate = calculation_date
		expiration_date = calculation_date + ql.Period(int(t),ql.Days)
		ts_r = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),self.day_count()))
		ts_g = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),self.day_count()))


		if w == 'call':
			option_type = ql.Option.Call
		elif w == 'put':
			option_type = ql.Option.Put
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")
		bates_process = ql.BatesProcess(
		ts_r, 
		ts_g,
		ql.QuoteHandle(ql.SimpleQuote(s)),
		v0, kappa, theta, eta, rho,
		lambda_, nu, delta,
		)
		engine = ql.BatesEngine(ql.BatesModel(bates_process))

		payoff = ql.PlainVanillaPayoff(option_type, float(k))
		europeanExercise = ql.EuropeanExercise(expiration_date)
		european_option = ql.VanillaOption(payoff, europeanExercise)
		european_option.setPricingEngine(engine)

		bates = european_option.NPV()
		return bates