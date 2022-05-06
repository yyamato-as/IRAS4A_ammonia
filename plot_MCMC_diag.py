from mcmc_tools import plot_corner
from ammonia_hfs_model import get_NH3_tau0, get_NH2D_tau0
from spectroscopic_data import (
    get_spectroscopic_data,
    NH3_pf,
    ortho_NH3_pf,
    para_NH3_pf,
    NH2D_pf,
)
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import get_param_labels, get_blob_labels
import pickle

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Times"]}
)

source = "IRAS4A2"
transition = ["NH3_33", "NH2D_33", "NH2D_44"]
fit_type = "OPR_fixed"
nburnin = 250

tau_func = {"NH3": get_NH3_tau0, "NH2D": get_NH2D_tau0}

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


fitfile = "./data/fit/{:s}_{:s}_{:s}.pkl".format(source, "_".join(transition), fit_type)
with open(fitfile, "rb") as f:
    sample = pickle.load(f)

# tau and NH2D/NH3
print("Calculating tau and D/H ratio...")
blob_arr = np.empty((*sample.shape[:-1], len(transition) + 1))
dv_FWHM, _, N, T, s = fetch_params(np.rollaxis(sample, axis=2, start=0), transition, fit_type)
for j, trans in enumerate(transition):
    tau = tau_func[trans.split("_")[0]](
        rotdata=rotdata[trans],
        dv_FWHM=dv_FWHM[trans],
        T=T,
        pf=pf[trans],
        N=N[trans],
    )
    blob_arr[..., j] = np.log10(tau)

blob_arr[..., -1] = N["NH2D_33"] / N["NH3_33"] if fit_type == "OPR_fixed" else N["NH2D_33"] / (N["NH3_33"] + N["NH3_44"])

# concatenate
sample = np.concatenate((sample, blob_arr), axis=2)

print("Generating corner plot...")
labels = get_param_labels(transition, OPR_NH3_free="free" in fit_type) + get_blob_labels(transition)

fig = plot_corner(sample, nburnin=nburnin, labels=labels)

fig.savefig(
    "./figure/{:s}_{:s}_{:s}_corner.pdf".format(source, "_".join(transition), fit_type),
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.show()


