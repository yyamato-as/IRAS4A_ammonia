import numpy as np
import matplotlib.pyplot as plt
from sourcedictionary import source_dict
from matplotlib import colors, ticker
from qdisk.plot import Map
from qdisk.classes import FitsImage
from astropy.coordinates import SkyCoord

# plot setting
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Times"]}
)

# colormap
freeze = np.loadtxt("../qDisk/cmap_freeze.txt")
freeze /= 255.0
cpal = colors.ListedColormap(freeze, name="freeze")

transition = ["NH3_11", "NH3_22", "NH3_33", "NH3_44", "NH3_55", "NH2D_33", "NH2D_44"]
# mom0path = "./data/mom0_old/"
mom0path = "/raid/work/yamato/IRAS4A_ammonia/"

# preset
distance = 300  # in pc
center_coord = source_dict["IRAS4A"]["IRAS4A2"]["radec"]
# center_coord = "03h29m10.43475s 31d13m32.00371s"
xlim = (-2.5, 4)
ylim = (-4, 2.5)

# label
trans_label = {'NH3_11': 'NH$_3$ (1,1)', 
               'NH3_22': 'NH$_3$ (2,2)', 
               'NH3_33': 'NH$_3$ (3,3)', 
               'NH3_44': 'NH$_3$ (4,4)', 
               'NH3_55': 'NH$_3$ (5,5)',
               'NH2D_33': 'NH$_2$D 3$_{1,3}$--3$_{0,3}$',
               'NH2D_44': 'NH$_2$D 4$_{1,4}$--4$_{0,4}$',
              }

# figure

fig, axes = plt.subplots(2, 4, figsize=(8, 4.5), sharex=True, sharey=True)

for i, trans in enumerate(transition):
    print(trans)

    # # set axis
    # sharex = None if i == 0 else ax
    # sharey = None if i == 0 else ax
    # ax = fig.add_subplot(2, 4, i+1, sharex=sharex, sharey=sharey)
    ax = axes.flatten()[i]

    # names of fits files
    prefix = mom0path + "IRAS4_{:s}.robust2.0.dv1.0.clean.image".format(trans)
    mom0name = prefix + "_M0.fits"
    dmom0name = prefix + "_dM0.fits"

    # measure the rms
    dmom0map = FitsImage(dmom0name, xlim=xlim, ylim=ylim)
    dmom0map.shift_phasecenter_toward(center_coord)
    rms = np.nanmedian(dmom0map.data)
    print("mom0 rms: {:.2f} mJy/beam km/s".format(rms))

    # mom0 = FitsImage(mom0name)
    # mom0.shift_phasecenter_toward(center_coord)
    # mom0.estimate_rms(rmin=40)
    # rms = mom0.rms
    # print("mom0 rms: {:.2f} mJy/beam km/s".format(rms))
    # print("analytic formula: {:.2f} mJy/beam km/s".format(rms_dmom0))

    # plot the mom0 map
    mom0map = Map(mom0name, ax=ax, xlim=xlim, ylim=ylim, center_coord=center_coord)
    mom0map.plot_colormap(method="pcolormesh", cmap=cpal, vmin=0.0, shading='gouraud')
    mom0map.add_colorbar(
        position="top",
        rotation=0.0,
        label="mJy beam$^{-1}$ km s$^{-1}$" if i == 0 else None,
        labelpad=10,
    )
    mom0map.add_beam(color="white", fill=False)
    mom0map.add_scalebar(scale=300 / distance, text="300 au")

    # adjust colorbar ticks
    mom0map.colorbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    mom0map.colorbar.ax.xaxis.set_minor_locator(ticker.MaxNLocator(nbins=4))
    mom0map.colorbar.ax.minorticks_on()

    # axis ticks
    mom0map.set_ticker(minor=True, majornticks=5, minornticks=True)

    # annotation
    ax.text(
        0.02,
        0.98,
        trans_label[trans]
        + "\n$\\sigma = $"
        + " {:.1f}".format(rms)
        + " mJy beam$^{-1}$ km s$^{-1}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
    )

    # continuum peak
    phasecenter = SkyCoord(source_dict["IRAS4A"]["IRAS4A2"]["radec"], frame=source_dict["IRAS4A"]["IRAS4A2"]["frame"])
    for source in source_dict["IRAS4A"].keys():
        peak = SkyCoord(source_dict["IRAS4A"][source]["radec"], frame=source_dict["IRAS4A"]["IRAS4A2"]["frame"])
        offset_x, offset_y = phasecenter.spherical_offsets_to(peak)
        ax.scatter(offset_x.arcsec, offset_y.arcsec, marker='+', color='black', lw=0.5)

    # 3sigma contour
    mom0map.overlay_contour(levels=np.array([3, 6, 9, 12, 15, 18, 21])*rms, color="dimgrey", linestyle="dashed", linewidth=0.5)


axes[1, 0].set(xlabel=r'$\Delta\mathrm{R.A.}$ [$^{\prime\prime}$]', ylabel=r'$\Delta\mathrm{Dec.}$ [$^{\prime\prime}$]')


# remove empty axis
for i in range(i + 1, axes.flatten().size):
    axes.flatten()[i].set_axis_off()

# plt.show()

fig.savefig("./figure/moment0_gallery.pdf", bbox_inches="tight", pad_inches=0.01)
