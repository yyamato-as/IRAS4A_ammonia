import numpy as np
import astropy.units as u
import astropy.constants as ac
from analysis_utils import J_nu, convolve_spectrum, FWHM_to_sigma, get_velocity_offset
from scipy.interpolate import interp1d

c = ac.c
h = ac.h
k_B = ac.k_B
pi = np.pi


def calculate_beam_filling_factor(source_size, beam):
	return source_size / np.sqrt(source_size ** 2 + beam[0] ** 2 ) * source_size / np.sqrt(source_size ** 2 + beam[1] ** 2)

def NH3_model(
    velax, beam, rotdata, hfsdata, dv_FWHM, v0, N, T, s, pf, Tbg=2.73 * u.K
):
    lpf = 0
    ref_nu0 = rotdata["nu0 [GHz]"] * u.GHz
    velax_h = (
        np.arange(velax.min().value - 1.0, velax.max().value + 1.0, 0.01) * velax.unit
    )
    for r, nu0 in zip(hfsdata["hfs ratio"], hfsdata["nu0 [GHz]"] * u.GHz):
        deltav = get_velocity_offset(nu0, ref_nu0)
        sigma_v = FWHM_to_sigma(dv_FWHM)
        lpf += r * np.exp(-((velax_h - v0 - deltav) ** 2) / (2 * sigma_v ** 2))
    f = calculate_beam_filling_factor(s, beam)
    nu0 = rotdata["nu0 [GHz]"] * u.GHz
    A_ul = 10 ** rotdata["logA [s^-1]"] / u.s
    g_u = rotdata["g_u"]
    E_u = rotdata["E_u [K]"] * u.K
    expterm = (np.exp(h * nu0 / (k_B * T)) - 1) / (np.exp(h * nu0 / (k_B * T)) + 1)
    N_u = g_u / pf(T) * np.exp(-E_u / T) * N
    tau0 = (
        1
        / (np.sqrt(2 * pi) * sigma_v)
        * c ** 3
        * A_ul
        / (8 * pi * nu0 ** 3)
        * expterm
        * N_u
    )
    model = f * (J_nu(ref_nu0, T) - J_nu(ref_nu0, Tbg)) * (1 - np.exp(-tau0 * lpf))
    model = convolve_spectrum(velax_h, model, width=np.diff(velax)[-1])
    model_f = interp1d(velax_h, model)
    model = model_f(velax) * model.unit
    return model.decompose()

def NH2D_model(
    velax, beam, rotdata, hfsdata, dv_FWHM, v0, N, T, s, pf, Tbg=2.73 * u.K
):
    lpf = 0
    ref_nu0 = rotdata["nu0 [GHz]"] * u.GHz
    velax_h = (
        np.arange(velax.min().value - 1.0, velax.max().value + 1.0, 0.01) * velax.unit
    )
    for r, nu0 in zip(hfsdata["hfs ratio"], hfsdata["nu0 [GHz]"] * u.GHz):
        deltav = get_velocity_offset(nu0, ref_nu0)
        sigma_v = FWHM_to_sigma(dv_FWHM)
        lpf += r * np.exp(-((velax_h - v0 - deltav) ** 2) / (2 * sigma_v ** 2))
    f = calculate_beam_filling_factor(s, beam)
    nu0 = rotdata["nu0 [GHz]"] * u.GHz
    A_ul = 10 ** rotdata["logA [s^-1]"] / u.s
    g_u = rotdata["g_u"]
    E_u = rotdata["E_u [K]"] * u.K
    expterm = np.exp(h * nu0 / (k_B * T)) - 1
    N_u = g_u / pf(T) * np.exp(-E_u / T) * N
    tau0 = (
        1
        / (np.sqrt(2 * pi) * sigma_v)
        * c ** 3
        * A_ul
        / (8 * pi * nu0 ** 3)
        * expterm
        * N_u
    )
    model = f * (J_nu(ref_nu0, T) - J_nu(ref_nu0, Tbg)) * (1 - np.exp(-tau0 * lpf))
    model = convolve_spectrum(velax_h, model, width=np.diff(velax)[-1])
    model_f = interp1d(velax_h, model)
    model = model_f(velax) * model.unit
    return model.decompose()


def get_NH3_tau0(rotdata, dv_FWHM, T, pf, N):
    sigma_v = FWHM_to_sigma(dv_FWHM)
    nu0 = rotdata["nu0 [GHz]"] * u.GHz
    A_ul = 10 ** rotdata["logA [s^-1]"] / u.s
    g_u = rotdata["g_u"]
    E_u = rotdata["E_u [K]"] * u.K
    expterm = (np.exp(h * nu0 / (k_B * T)) - 1) / (np.exp(h * nu0 / (k_B * T)) + 1)
    N_u = g_u / pf(T) * np.exp(-E_u / T) * N
    tau0 = (
        1
        / (np.sqrt(2 * pi) * sigma_v)
        * c ** 3
        * A_ul
        / (8 * pi * nu0 ** 3)
        * expterm
        * N_u
    )
    return tau0


def get_NH2D_tau0(rotdata, dv_FWHM, T, pf, N):
    sigma_v = FWHM_to_sigma(dv_FWHM)
    nu0 = rotdata["nu0 [GHz]"] * u.GHz
    A_ul = 10 ** rotdata["logA [s^-1]"] / u.s
    g_u = rotdata["g_u"]
    E_u = rotdata["E_u [K]"] * u.K
    expterm = np.exp(h * nu0 / (k_B * T)) - 1
    N_u = g_u / pf(T) * np.exp(-E_u / T) * N
    tau0 = (
        1
        / (np.sqrt(2 * pi) * sigma_v)
        * c ** 3
        * A_ul
        / (8 * pi * nu0 ** 3)
        * expterm
        * N_u
    )
    return tau0


def populate_params(param, transition, fit_type):
    ntrans = len(transition)

    dv_FWHM = {}
    for i, trans in enumerate(transition):
        dv_FWHM[trans] = param[i] * u.km / u.s

    v0 = {}
    for i, trans in enumerate(transition):
        v0[trans] = param[i + ntrans] * u.km / u.s

    

    if fit_type == "OPR_fixed":
        logN_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        N_NH3 = 10**logN_NH3 / u.cm**2
        N_NH2D = 10**logN_NH2D / u.cm**2
        N = {trans: N_NH3 if "NH3" in trans else N_NH2D for trans in transition}
        return dv_FWHM, v0, N, T * u.K, s

    elif fit_type == "OPR_free":
        logN_NH3, OPR_NH3, logN_NH2D, T, s = param[2 * ntrans :]
        N_oNH3 = 10**logN_NH3 * OPR_NH3 / (1.0 + OPR_NH3) / u.cm**2
        N_pNH3 = 10**logN_NH3 * 1.0 / (1.0 + OPR_NH3) / u.cm**2
        N_NH2D = 10**logN_NH2D / u.cm**2
        N = {
            trans: N_oNH3 if trans == "NH3_33" else N_pNH3 if "NH3" in trans else N_NH2D
            for trans in transition
        }
        return dv_FWHM, v0, N, T * u.K, s