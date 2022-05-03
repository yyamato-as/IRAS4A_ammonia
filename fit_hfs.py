# import module
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from basic_utils_v2 import pi, FWHM_to_sigma, partition_function, J_nu, h, k_B, c
import splatalogue_query_molecular_data as sqmd
import matplotlib.pyplot as plt
from uncertainties import ufloat
import pickle
from mcmc_tools import (
    log_prior,
    emcee_run_wrapper,
    plot_walker,
    condition,
    plot_corner,
    plot_chain,
)
from astropy.convolution import convolve, Gaussian1DKernel
from multiprocessing import Pool
from classes import AmmoniaLTEModel
from numpy.random import default_rng
from spectroscopic_data import (
    NH3_pf,
    para_NH3_pf,
    ortho_NH3_pf,
    NH2D_pf,
    get_spectroscopic_data,
)
from ammonia_hfs_model import NH3_model, NH2D_model

# initial difinition of the variables and parameters
source = "IRAS4A2"
transition = ["NH3_33", "NH2D_33", "NH2D_44"]
fit_type = "OPR_fixed"
initial_state = {
    "OPR_fixed": [
        2.5,
        3.5,
        4,
        1,
        1,
        7,
        7,
        7,
        7,
        7,
        18,
        17,
        160,
        0.30,
    ],
    "OPR_free": [
        2.5,
        3.5,
        4,
        1,
        1,
        7,
        7,
        7,
        7,
        7,
        18,
        1,
        17,
        160,
        0.30,
    ],
}
bound = {
    "OPR_fixed": [
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (14, 21),
        (14, 21),
        (50, 350),
        (0.01, 0.6),
    ],
    "OPR_free": [
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (0.1, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (4, 10),
        (14, 21),
        (1e-2, 1e2),
        (14, 21),
        (50, 350),
        (0.01, 0.6),
    ],
}

model_func = {"NH3": NH3_model,
              "NH2D": NH2D_model}

print("Importing the spectroscopic data...")
rotdata = {trans: get_spectroscopic_data(trans, hfs=False) for trans in transition}
hfsdata = {trans: get_spectroscopic_data(trans, hfs=True) for trans in transition}
if fit_type == "OPR_fixed":
    pf = {trans: NH3_pf if "NH3" in trans else NH2D_pf for trans in transition}
    

def fetch_params(param, transition, fit_type):
    ntrans = len(transition)

    dv_FWHM = {}
    for i, trans in enumerate(transition):
        dv_FWHM[trans] = param[i]

    v0 = {}
    for i, trans in enumerate(transition):
        v0[trans] = param[i + ntrans]

    if fit_type == "OPR_fixed":
        logN_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        return dv_FWHM, v0, logN_NH3, logN_NH2D, T, s

    elif fit_type == "OPR_free":
        logN_NH3, OPR_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        return dv_FWHM, v0, logN_NH3, OPR_NH3, logN_NH2D, T, s


# paths for required data
fitfilepath = "/raid/work/yamato/IRAS4A_ammonia_analysis/analysis_data/line_fit/"
spectrum_path = "/raid/work/yamato/IRAS4A_ammonia_analysis/analysis_data/spectrum/"

# beam data in arcsec
beam = {
    "Choi": (0.3, 0.3),
    "NH3_33": (1.31, 0.98),
    "NH3_44": (1.25, 0.95),
    "NH3_55": (1.27, 1.02),
    "NH2D_33": (1.52, 1.23),
    "NH2D_44": (1.26, 0.92),
}

######## Kernel for emitting region size ############
print("Setting the prior kernel...")

def Gaussian_kernel(mu, stddev):
    def g(x):
        return (
            1
            / (np.sqrt(2 * pi) * stddev)
            * np.exp(-((x - mu) ** 2) / (2 * stddev**2))
        )

    return g

source_size_kernel = Gaussian_kernel(
    0.30, 0.03
)  # based on the MCMC fit for peak brightness temperature of our data and Choi's data


######## import the spectrum ########
print("Importing the observed spectra...")
# unit for spectrum
unit = "K"

# import
velax = {}
obs_spectrum = {}
obs_error = {}
for trans in transition:
    spectrum_file = (
        spectrum_path
        + "{:s}_{:s}.robust2.0.dv1.0.clean.image.pbcor.fits.spectrum.pickle".format(
            source, trans
        )
    )
    with open(spectrum_file, "rb") as f:
        data = pickle.load(f)
    vel = data.coord.pixel_to_world(np.arange(data.spectrum.size)).quantity.to(
        u.km / u.s, equivalencies=u.doppler_radio(data.restfreq)
    )
    spec = data.spectrum.to(
        unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam)
    )
    error = data.rms.to(
        unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam)
    )
    obs_spectrum[trans] = spec[error != 0.0]  # flag the bad channels
    obs_error[trans] = error[error != 0.0]
    velax[trans] = vel[error != 0.0]
    print("Imported {:s}".format(spectrum_file))


####### define MCMC-related function #######

def log_prior(param, bound):
    for p, b in zip(param, bound):
        if not condition(p, b):
            return -np.inf
    return np.log(source_size_kernel(param[-1]))

def log_likelihood(param):
    if fit_type == "OPR_fixed":
        dv_FWHM, v0, logN_NH3, logN_NH2D, T, s = fetch_params(param, transition, fit_type)
    else:
        dv_FWHM, v0, logN_NH3, OPR_NH3, logN_NH2D, T, s = fetch_params(param, transition, fit_type)
        
    
    ll = 0
    for trans in transition:
        model = model_func[trans.split("_")[0]](velax=velax[trans], beam=beam[trans], invdata=rotdata[trans], hfsdata=hfsdata[trans], dv_FWHM=dv_FWHM[trans], pf=)

    ll = 0
    N_NH3 = 10 ** logN_NH3
    N_NH2D = 10 ** logN_NH2D
    # for (3,3)
    trans = "NH3_33"
    model = NH3_N_model(
        velax=velax[trans],
        beam=beam[trans],
        invdata=invdata[trans],
        hfsdata=hfsdata[trans],
        dv_FWHM=dv_FWHM_NH3_33 * u.km / u.s,
        pf=ortho_NH3_pf,
        v0=v0_NH3_33 * u.km / u.s,
        N=N_NH3/u.cm**2 * OPR_NH3 / (1 + OPR_NH3),
        T=T * u.K,
        s=s,
    )
    ll += -0.5 * np.nansum(
        (obs_spectrum[trans] - model) ** 2 / obs_error[trans] ** 2
        + np.log(2 * pi * obs_error[trans].value ** 2)
    )

