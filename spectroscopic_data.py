import os
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Splatalogue_query import get_query, generate_smart_table


class partition_function:
	
	def __init__(self, file, check_fit=False):
		self.file = file
		self.T, self.Q = np.loadtxt(self.file, unpack=True)
		self.fit = self.T.max() < 400 or self.T.size < 15
		if self.fit:
			self.popt, pcov = curve_fit(self.pl_model, self.T, self.Q, p0=[10, 2])
			perr = np.sqrt(np.diag(pcov))
			r_err = perr / self.popt
			assert np.all(r_err < 0.1)
			print('Data used for partition function: {:s}'.format(os.path.abspath(self.file)))
			print('Data are fitted by a power-law function due to poor data points:')
			print('fitted parameters: Q at 10 K = {:.2f}, power-law index = {:.2f}'.format(*self.popt))
			print('Relative error on those parameters: {:.2f}, {:.2f}'.format(*r_err))
			if check_fit:
				plt.plot(self.T, self.Q, label='data')
				plt.plot(self.T, self.pl_model(self.T, *self.popt), label='model')
				plt.xlabel('Temperature [K]')
				plt.ylabel('Partition function')
				plt.legend()
				plt.show()
		else:
			self.f = interp1d(self.T, self.Q)
			print('Data used for partition function: {:s}'.format(os.path.abspath(self.file)))
			print('Data are linearly interpolated.')
	
	@staticmethod
	def pl_model(T, q0, p):
		return q0 * (T/10) ** p
		
	def __call__(self, T):
		if isinstance(T, u.Quantity):
			T = T.to(u.K).value
		if self.fit:
			return np.array(self.pl_model(T, *self.popt))
		else:
			return self.f(T)


NH3_pf = partition_function(file='./data/partition_function/NH3_partition_function_fromRotConst.dat')
ortho_NH3_pf = partition_function(file='./data/partition_function/o-NH3_partition_function_fromRotConst.dat')
para_NH3_pf = partition_function(file='./data/partition_function/p-NH3_partition_function_fromRotConst.dat')
NH2D_pf = partition_function(file='./data/partition_function/NH2D_partition_function.dat')


def get_spectroscopic_data(trans, hfs=False):
    q = get_query(trans, hfs=hfs)
    return generate_smart_table(trans, q, hfs=hfs)
    