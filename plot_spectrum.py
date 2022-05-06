import numpy as np
import pickle
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib import ticker
from numpy.random import default_rng
from plot_utils import source_label, trans_label, get_param_names
from ammonia_hfs_model import NH3_model, NH2D_model
from spectroscopic_data import NH3_pf, NH2D_pf, ortho_NH3_pf, para_NH3_pf, get_spectroscopic_data
from classes import Spectrum


spectrumpath = './data/spectrum/'
samplepath = "./data/fit/"

sources = ["IRAS4A1", "IRAS4A2"]
transition = ['NH3_33', 'NH3_44', 'NH3_55', 'NH2D_33', 'NH2D_44']
fit_type = ["OPRfree", "OPRfixed"]


# plot setting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"]})

unit = 'K'
ylabel = r'$T_\mathrm{b}$ [K]'

ymax = {'IRAS4A1': 8.5,
	    'IRAS4A2': 11}

colors = {'OPRfixed': "tab:blue",
         'OPRfree': 'tab:orange'}

beam = {'NH3_33': (1.31, 0.98),
	    'NH3_44': (1.25, 0.95),
	    'NH3_55': (1.27, 1.02),
	    'NH2D_33': (1.52, 1.23),
	    'NH2D_44': (1.26, 0.92),
	   }

nburnin = 2500

# figure setup
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey="row", squeeze=False, figsize=(10, 3.5))

for i, source in enumerate(sources):

    # load the mcmc sample and seed the random number
    sample_dict = {}
    for ft in fit_type:
        filename = samplepath + "{}_{}.pkl".format(source, ft)
        with open(filename, 'rb') as f:
            sample = pickle.load(f)
        sample_dict[ft] = sample[nburnin:].reshape(-1, sample.shape[-1])


    rng = default_rng(714)
    rint = rng.integers(low=0, high=sample.shape[0], size=20)

    for j, trans in enumerate(transition):
        # select axis
        ax = axes[i, j]

        # load observed spectra
        print("Loading observed spectrum for {}...".format(trans))
        spectrum_file = spectrumpath + '{:s}_{:s}.robust2.0.dv1.0.clean.image.pbcor.fits.spectrum.pickle'.format(source, trans)
        with open(spectrum_file, 'rb') as f:
            data = pickle.load(f)
        
        # unit conversion
        vel = data.coord.pixel_to_world(np.arange(data.spectrum.size)).quantity.to(u.km/u.s, equivalencies=u.doppler_radio(data.restfreq))
        spec = data.spectrum.to(unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam))
        error = data.rms.to(unit, equivalencies=u.brightness_temperature(data.restfreq, data.beam))

        # flag bad channels
        vel = vel[error != 0.0]
        spec = spec[error != 0.0]
        error = error[error != 0.0]

        # plot 
        ax.plot(vel, spec, color='black', drawstyle='steps-mid')
        ax.axhline(y=0.0, color='grey', ls='dashed')
        ax.set(xlim=(-15.8, 29.8), ylim=(-1.8, 11))
        if i == 0:
            ax.set_title(trans_label[trans])

        # get spectroscopic data
        rotdata = get_spectroscopic_data(trans, hfs=False)
        hfsdata = get_spectroscopic_data(trans, hfs=True)

        # high resolution velocity axis
        vel_hires = np.arange(vel.min().value-1.0, vel.max().value+1.0, 0.01) * vel.unit

        # prepare model spectra from mcmc sample
        for ft in fit_type:

            OPR_NH3_free = "free" in ft
            param_names = get_param_names(transition, OPR_NH3_free=OPR_NH3_free)
            sample = sample_dict[ft]
            color = colors[ft]

            for n in rint:

                # get parameter values
                dv_FWHM = sample[n, param_names.index("dv_FWHM_{:s}".format(trans))] * u.km / u.s
                v0 = sample[n, param_names.index("v_sys_{:s}".format(trans))] * u.km / u.s
                Tex = sample[n, param_names.index("Tex")] * u.K
                s = sample[n, param_names.index("theta_s")]

                if source == "IRAS4A1":
                    s = 0.25

                if "NH3" in trans:
                    modelf = NH3_model
                    if OPR_NH3_free and trans == 'NH3_33':
                        OPR_NH3 = sample[n, param_names.index("OPR_NH3")]
                        N = 10 ** sample[n, param_names.index("logN_NH3")] * OPR_NH3 / (1 + OPR_NH3) / u.cm ** 2
                        pf = ortho_NH3_pf
                    elif OPR_NH3_free:
                        OPR_NH3 = sample[n, param_names.index("OPR_NH3")]
                        N = 10 ** sample[n, param_names.index("logN_NH3")] * 1 / (1 + OPR_NH3) / u.cm ** 2
                        pf = para_NH3_pf
                    else:
                        N = 10 ** sample[n, param_names.index("logN_NH3")] / u.cm ** 2
                        pf = NH3_pf
                if "NH2D" in trans:
                    modelf = NH2D_model
                    N = 10 ** sample[n, param_names.index("logN_NH2D")] / u.cm **2 
                    pf = NH2D_pf
                model = modelf(velax=vel_hires,
                                beam=beam[trans],
                                rotdata=rotdata,
                                hfsdata=hfsdata,
                                dv_FWHM=dv_FWHM,
                                pf=pf,
                                v0=v0,
                                N=N,
                                T=Tex,
                                s=s
                                )
                ax.plot(vel_hires, model, color=color, alpha=0.3, lw=0.5)

        # adjust appearance 
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        # ax.text(0.05, 0.95, trans_label[trans], ha='left', va='top', transform=ax.transAxes, color='black')

    # source label
    axes[i, 0].text(0.05, 0.95, source_label[source], ha='left', va='top', transform=axes[i, 0].transAxes, color='black', fontweight="extra bold", fontsize="large")

# axis label
fig.supxlabel(r"$v_\mathrm{LSR}$ [km s$^{-1}$]", y=0.0)
fig.supylabel(r"$T_\mathrm{b}$ [K]", x=0.08)
    
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("./figure/IRAS4A_spectrum_fit.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()



        


        