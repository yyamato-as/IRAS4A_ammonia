import numpy as np 
import astropy.units as u 
import astropy.constants as ac
from astropy.convolution import Gaussian1DKernel, convolve
from matplotlib import colors

# constants
c = ac.c
h = ac.h
k_B = ac.k_B
pi = np.pi

# plot colormap
freeze = np.loadtxt("../qDisk/cmap_freeze.txt")
freeze /= 255.0
cpal = colors.ListedColormap(freeze, name="freeze")

def convolve_spectrum(velax, model, width=1*u.km/u.s):
    sigma = FWHM_to_sigma(width)
    kernel = Gaussian1DKernel(sigma/np.diff(velax)[-1])
    convolved = convolve(model, kernel)
    return convolved
	
def sigma_to_FWHM(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def FWHM_to_sigma(FWHM):
    return FWHM / np.sqrt(8 * np.log(2))

def J_nu(nu, T):
	return h * nu / k_B / (np.exp(h * nu / (k_B * T)) - 1)

def get_velocity_offset(nu0, ref_nu0):
    dv = - (nu0 - ref_nu0) / ref_nu0 * c
    return dv

def get_median_uncertainty(sample, q=[16, 50, 84]):
    percentiles = np.nanpercentile(sample, q=q)
    median = percentiles[1]
    uc_lower = percentiles[1] - percentiles[0]
    uc_upper = percentiles[2] - percentiles[1]
    return median, uc_lower, uc_upper
