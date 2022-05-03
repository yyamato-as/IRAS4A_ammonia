# import module
import numpy as np
import astropy.units as u
# from scipy.optimize import curve_fit
# from scipy.interpolate import interp1d
# from basic_utils_v2 import pi, FWHM_to_sigma, partition_function, J_nu, h, k_B, c
# import splatalogue_query_molecular_data as sqmd
import matplotlib.pyplot as plt
# from uncertainties import ufloat
import pickle
from mcmc_tools import (
    log_prior,
    emcee_run_wrapper,
    plot_walker,
    condition,
    plot_corner,
    plot_chain,
)
# from astropy.convolution import convolve, Gaussian1DKernel
from multiprocessing import Pool
# from classes import AmmoniaLTEModel
# from numpy.random import default_rng
from spectroscopic_data import (
    NH3_pf,
    para_NH3_pf,
    ortho_NH3_pf,
    NH2D_pf,
    get_spectroscopic_data,
)
from ammonia_hfs_model import NH3_model, NH2D_model

plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"]})

# initial difinition of the variables and parameters
source = "IRAS4A1"
transition = ["NH3_33", "NH2D_33", "NH2D_44"]
fit_type = "OPR_fixed"
initial_state = {
    "OPR_fixed": [
        2.5,
        1,
        1,
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
        1,
        1,
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

model_func = {"NH3": NH3_model, "NH2D": NH2D_model}

print("Importing the spectroscopic data...")
rotdata = {trans: get_spectroscopic_data(trans, hfs=False) for trans in transition}
hfsdata = {trans: get_spectroscopic_data(trans, hfs=True) for trans in transition}
if fit_type == "OPR_fixed":
    pf = {trans: NH3_pf if "NH3" in trans else NH2D_pf for trans in transition}
elif fit_type == "OPR_free":
    pf = {
        trans: ortho_NH3_pf
        if trans == "NH3_33"
        else para_NH3_pf
        if "NH3" in trans
        else NH2D_pf
        for trans in transition
    }


def fetch_params(param, transition, fit_type):
    ntrans = len(transition)

    dv_FWHM = {}
    for i, trans in enumerate(transition):
        dv_FWHM[trans] = param[i] * u.km / u.s

    v0 = {}
    for i, trans in enumerate(transition):
        v0[trans] = param[i + ntrans] * u.km / u.s

    if fit_type == "OPR_fixed":
        logN_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        N_NH3 = 10 ** logN_NH3 / u.cm ** 2
        N_NH2D = 10 ** logN_NH2D / u.cm ** 2
        N = {trans: N_NH3 if "NH3" in trans else N_NH2D for trans in transition}
        return dv_FWHM, v0, N, T*u.K, s

    elif fit_type == "OPR_free":
        logN_NH3, OPR_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        N_oNH3 = 10 ** logN_NH3 * OPR_NH3 / (1. + OPR_NH3) / u.cm ** 2
        N_pNH3 = 10 ** logN_NH3 * 1. / (1. + OPR_NH3) / u.cm ** 2
        N_NH2D = 10 ** logN_NH2D / u.cm ** 2
        N = {trans: N_oNH3 if trans == "NH3_33" else N_pNH3 if "NH3" in trans else N_NH2D for trans in transition}
        return dv_FWHM, v0, N, T*u.K, s


# paths for required data
fitfilepath = "./data/fit/"
spectrum_path = "./data/spectrum/"

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
            / (np.sqrt(2 * np.pi) * stddev)
            * np.exp(-((x - mu) ** 2) / (2 * stddev**2))
        )

    return g

if source == "IRAS4A1":
    source_size_kernel = Gaussian_kernel(0.25, 0.06)

if source == "IRAS4A2":
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
print("Constructing the likelihood functions...")


def log_prior(param, bound):
    for p, b in zip(param, bound):
        if not condition(p, b):
            return -np.inf
    return np.log(source_size_kernel(param[-1]))


def log_likelihood(param):
    dv_FWHM, v0, N, T, s = fetch_params(param, transition, fit_type)

    ll = 0
    for trans in transition:
        model = model_func[trans.split("_")[0]](
            velax=velax[trans],
            beam=beam[trans],
            invdata=rotdata[trans],
            hfsdata=hfsdata[trans],
            dv_FWHM=dv_FWHM[trans] ,
            v0=v0[trans],
            N=N[trans],
            pf=pf[trans],
            T=T,
            s=s
        )
        ll += -0.5 * np.nansum(
        (obs_spectrum[trans] - model) ** 2 / obs_error[trans] ** 2
        + np.log(2 * np.pi * obs_error[trans].value ** 2)
        )
    return ll

def log_probability(param):
    lp = log_prior(param, bound)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(param)
    return lp + ll


############ execute MCMC fit ###################
print("Executing MCMC sampling...")
initial_state = initial_state[fit_type]
bound = bound[fit_type]
nwalker = 200
# nstep = 2500
nstep = 250
# nburnin = 2500
nburnin = 250
with Pool(processes=16) as pool:
    _, sample = emcee_run_wrapper(
        log_probability,
        initial_state,
        nwalker=nwalker,
        nstep=nstep,
        nburnin=nburnin,
        relative_error=1e-4,
        get_sample=True,
        discard=False,
        pool=pool,
    )

print("Plotting the results...")
corner_fig = plot_corner(sample, nburnin=nburnin)
plot_walker(sample, nburnin=nburnin)
plt.show()

print("Saving the results...")
import pickle
savefile = fitfilepath + "{:s}_{:s}_{:s}.pkl".format(source, "_".join(transition), fit_type)
with open(savefile, 'wb') as f:
    pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
