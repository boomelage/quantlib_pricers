import os
import numpy as np
import QuantLib as ql
from joblib import Parallel, delayed

class barrier_option_pricer():
	def barrier_price(self,
	        s,k,t,r,g,w,
	        barrier_type_name,barrier,rebate,
	        kappa,theta,rho,eta,v0
	        ):
		s = float(s)
		k = float(k)
		t = int(t)
		r = float(r)
		g = float(g)
		barrier = float(barrier)
		rebate = float(rebate)
		kappa = float(kappa)
		theta = float(theta)
		rho = float(rho)
		eta = float(eta)
		v0 = float(v0)
		calculation_date = ql.Date.todaysDate()
		ql.Settings.instance().evaluationDate = calculation_date
		expiration_date = calculation_date + ql.Period(int(t),ql.Days)
		riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(r),ql.Actual365Fixed()))
		dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date,float(g),ql.Actual365Fixed()))
		if w == 'call':
			option_type = ql.Option.Call
		elif w == 'put':
			option_type = ql.Option.Put
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")
		if barrier_type_name == 'UpOut':
			barrierType = ql.Barrier.UpOut
		elif barrier_type_name == 'DownOut':
			barrierType = ql.Barrier.DownOut
		elif barrier_type_name == 'UpIn':
			barrierType = ql.Barrier.UpIn
		elif barrier_type_name == 'DownIn':
			barrierType = ql.Barrier.DownIn
		else:
			raise KeyError('barrier flag error')


		heston_process = ql.HestonProcess(
			riskFreeTS,dividendTS, 
			ql.QuoteHandle(ql.SimpleQuote(s)), 
			v0, kappa, theta, eta, rho
		)

		heston_model = ql.HestonModel(heston_process)
		engine = ql.FdHestonBarrierEngine(heston_model)
		
		exercise = ql.EuropeanExercise(expiration_date)
		payoff = ql.PlainVanillaPayoff(option_type, k)

		barrierOption = ql.BarrierOption(
		barrierType, barrier, rebate, payoff, exercise)

		barrierOption.setPricingEngine(engine)

		try:
			return min(max(barrierOption.NPV(),0),s)
		except Exception as e:
			print(e)
			return np.nan


	def row_barrier_price(self,row):
		return self.barrier_price(
	        row['spot_price'],
	        row['strike_price'],
	        row['days_to_maturity'],
	        row['risk_free_rate'],
	        row['dividend_rate'],
	        row['w'],
	        row['barrier_type_name'],
	        row['barrier'],
	        row['rebate'],
	        row['kappa'],
	        row['theta'],
	        row['rho'],
	        row['eta'],
	        row['v0']
			)

	def df_barrier_price(self, df):
	    max_jobs = os.cpu_count() // 4
	    max_jobs = max(1, max_jobs)
	    return Parallel(n_jobs=max_jobs)(delayed(self.row_barrier_price)(row) for _, row in df.iterrows())