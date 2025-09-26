import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import astropy.units as u
from astropy.coordinates import SkyCoord
import pickle
import emcee
from sourcedictionary import source_dict
from plot_utils import source_label, trans_label, get_param_names
from ammonia_hfs_model import NH3_model, NH2D_model
from spectroscopic_data import (
    NH3_pf,
    NH2D_pf,
    ortho_NH3_pf,
    para_NH3_pf,
    get_spectroscopic_data,
)
from qdisk.classes import FitsImage

imagepath = "/raid/work/yamato/IRAS4A_ammonia/"
spectrumpath = "./data/spectrum/"
fitdatapath = "./data/fit/"

sources = ["IRAS4A1", "IRAS4A2"]
transition = ["NH3_33", "NH3_44", "NH3_55", "NH2D_33", "NH2D_44"]
fit_type = ["OPR_free", "OPR_fixed"]

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["Times"],
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

colors = {"OPR_fixed": "tab:blue", "OPR_free": "tab:orange"}
unit = "K"

nburnin = 100

beam = {'NH3_33': (1.31, 0.98),
	    'NH3_44': (1.25, 0.95),
	    'NH3_55': (1.27, 1.02),
	    'NH2D_33': (1.52, 1.23),
	    'NH2D_44': (1.26, 0.92),
	   }

fig, axes = plt.subplots(
    nrows=2, ncols=5, sharex=True, sharey=True, squeeze=False, figsize=(10, 3.5)
)

for i, source in enumerate(sources):
    # peak position
    peak_coord = source_dict["IRAS4A"][source]["radec"]

    for j, trans in enumerate(transition):
        # observed spectra
        spectrum_file = (
            spectrumpath
            + "{:s}_{:s}.robust2.0.dv1.0.clean.image.pbcor.fits.spectrum.pickle".format(
                source, trans
            )
        )
        with open(spectrum_file, "rb") as f:
            data = pickle.load(f)

        # unit conversion
        vel = data.coord.pixel_to_world(np.arange(data.spectrum.size)).quantity.to(
            u.km / u.s, equivalencies=u.doppler_radio(data.restfreq)
        )
        spec = data.spectrum.to(
            unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam)
        )
        error = data.rms.to(
            unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam)
        )

        # flag bad channels
        vel = vel[error != 0.0]
        spec = spec[error != 0.0]

        # plot
        ax = axes[i, j]
        ax.plot(vel, spec, color="black", drawstyle="steps-mid")
        ax.axhline(y=0.0, color="grey", ls="dashed")
        ax.set(xlim=(-15.8, 29.8), ylim=(-1.8, 11))

        # get spectroscopic data
        rotdata = get_spectroscopic_data(trans, hfs=False)
        hfsdata = get_spectroscopic_data(trans, hfs=True)

        # high resolution velocity axis
        vel_hires = (
            np.arange(vel.min().value - 1.0, vel.max().value + 1.0, 0.01) * vel.unit
        )

        for k, ft in enumerate(fit_type):
            samplerfile = (
                fitdatapath
                + f"{source}_NH3_33_NH3_44_NH3_55_NH2D_33_NH2D_44_{ft}_v2.h5"
            )
            sampler = emcee.backends.HDFBackend(samplerfile)
            bestparam = sampler.get_chain(discard=nburnin, flat=True)[
                np.argmax(sampler.get_log_prob(discard=nburnin, flat=True))
            ]

            OPR_NH3_free = "free" in ft
            param_names = get_param_names(transition, OPR_NH3_free=OPR_NH3_free)
            color = colors[ft]

            # get parameter
            dv_FWHM = (
                bestparam[param_names.index("dv_FWHM_{:s}".format(trans))] * u.km / u.s
            )
            v0 = bestparam[param_names.index("v_sys_{:s}".format(trans))] * u.km / u.s
            Tex = bestparam[param_names.index("Tex")] * u.K
            s = bestparam[param_names.index("theta_s")]
            if "NH3" in trans:
                modelf = NH3_model
                if OPR_NH3_free and trans == "NH3_33":
                    OPR_NH3 = bestparam[param_names.index("OPR_NH3")]
                    N = (
                        10 ** bestparam[param_names.index("logN_NH3")]
                        * OPR_NH3
                        / (1 + OPR_NH3)
                        / u.cm**2
                    )
                    pf = ortho_NH3_pf
                elif OPR_NH3_free:
                    OPR_NH3 = bestparam[param_names.index("OPR_NH3")]
                    N = (
                        10 ** bestparam[param_names.index("logN_NH3")]
                        * 1
                        / (1 + OPR_NH3)
                        / u.cm**2
                    )
                    pf = para_NH3_pf
                else:
                    N = 10 ** bestparam[param_names.index("logN_NH3")] / u.cm**2
                    pf = NH3_pf
            if "NH2D" in trans:
                modelf = NH2D_model
                N = 10 ** bestparam[param_names.index("logN_NH2D")] / u.cm**2
                pf = NH2D_pf
            model = modelf(
                velax=vel_hires,
                beam=beam[trans],
                rotdata=rotdata,
                hfsdata=hfsdata,
                dv_FWHM=dv_FWHM,
                pf=pf,
                v0=v0,
                N=N,
                T=Tex,
                s=s,
            )
            ax.plot(vel_hires, model, color=color)
            ax.axvline(x=v0.value, color="grey", ls="dotted")

        if i == 0:
            ax.set_title(trans_label[trans])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(4))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        # ax.text(0.05, 0.95, trans_label[trans], ha='left', va='top', transform=ax.transAxes, color='black')

    # source label
    axes[i, 0].text(
        0.05,
        0.95,
        source_label[source],
        ha="left",
        va="top",
        transform=axes[i, 0].transAxes,
        color="black",
        fontweight="extra bold",
        fontsize="large",
    )

# axis label
fig.supxlabel(r"$v_\mathrm{LSR}$ [km s$^{-1}$]", y=0.0)
fig.supylabel(r"$T_\mathrm{b}$ [K]", x=0.08)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
