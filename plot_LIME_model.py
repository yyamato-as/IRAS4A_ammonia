import numpy as np
import matplotlib.pyplot as plt
#sys.path.append('../IRAS4A_ammonia_analysis')
from classes import Spectrum
import astropy.units as u
import astropy.constants as ac
import pickle
from plot_utils import trans_label, fiducial_fit, get_param_names, nburnin, source_label
from matplotlib import ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"]})

obsspecpath = './data/spectrum/'
modelspecpath = "./data/LIME/"
fitpath = "./data/fit/"
transition = ['NH2D_33', 'NH2D_44']
source = ['IRAS4A1', 'IRAS4A2']

fit_sample = {src: fitpath + "{}_{}.pkl".format(src, fiducial_fit[src]) for src in source}


Mstar = [0.08, 1.0]
color = {0.08: '#FF4500', 1.0: '#1E8FFF'}

def load_spectrum(spectrumfile, vsys=0.0*u.km/u.s):
    with open(spectrumfile, 'rb') as f:
        data = pickle.load(f)
    nu = data.coord.pixel_to_world(np.arange(data.spectrum.size))
    v = (ac.c * (1 - nu / data.restfreq)).to(u.km/u.s) + vsys
    return v, data.spectrum

# get v_sys estimated from hyperfine fits
transition_all = ["NH3_33", "NH3_44", "NH3_55", "NH2D_33", "NH2D_44"]
vsys = {}
for src in source:

    OPR_NH3_free = "free" in fiducial_fit[src]
    param_names = get_param_names(transition_all, OPR_NH3_free=OPR_NH3_free)

    with open(fit_sample[src], "rb") as f:
        sample = pickle.load(f)
        sample = sample[nburnin:].reshape(-1, sample.shape[-1])

    vsys[src] = {}
    for trans in transition:
        vsys[src][trans] = np.nanpercentile(sample[:,param_names.index("v_sys_{:s}".format(trans))], q=50) * u.km / u.s

print(vsys)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,3.5), squeeze=False, sharex=True, sharey=True)

for i, src in enumerate(source):
    for j, trans in enumerate(transition):

        ax = axes[i, j]

        # observed spectrum
        spectrumfile = obsspecpath + '_'.join([src, trans]) + '.robust2.0.dv1.0.clean.image.pbcor.fits.spectrum.pickle'
        v, spec = load_spectrum(spectrumfile)
        
        ax.plot(v, spec*1e3, color="black", label=src, drawstyle='steps-mid') # in mJy / beam

        # model
        for M in Mstar:
            spectrumfile = modelspecpath + 'IRAS4A_{:s}Msun_{:s}_postprocessed.fits.spectrum.pickle'.format(str(M), trans)
            v, spec = load_spectrum(spectrumfile, vsys=vsys[src][trans])

            ax.plot(v, spec*1e3, color=color[M], label='Model ($M_\\star = $' + str(M) + '$M_\\odot$)', lw=1.5)

        ax.set(xlim=(-15.8, 29.8), ylim=(-0.8, None))
        if i == 0:
            ax.set_title(trans_label[trans])
        ax.axhline(y=0.0, color='grey', ls='dashed', zorder=-100)
        # ax.legend(loc='upper right', fontsize=6)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

    axes[i, 0].text(0.05, 0.95, source_label[src], ha='left', va='top', transform=axes[i, 0].transAxes, color='black', fontweight="extra bold", fontsize="large")

fig.supxlabel(r"$v_\mathrm{LSR}$ [km s$^{-1}$]", y=0.0)
fig.supylabel(r"$I_\nu$ [mJy beam$^{-1}$]", x=0.04)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig("./figure/IRAS4A_NH2D_LIME_model.pdf", bbox_inches="tight", pad_inches=0.04)
plt.show()