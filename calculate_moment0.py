from qdisk.product import calculate_moment
import pickle
import numpy as np

transition = ["NH3_11", "NH3_22", "NH3_33", "NH3_44", "NH3_55", "NH2D_33", "NH2D_44"]

integ_range = {"NH3_11": [(4, 9), (-3, 1), (12, 16), (-14, -11), (23, 29)], 
               "NH3_22": [(4, 10), (-13, -8), (21, 26)],
               "NH3_33": [(3, 11), (-16, -11), (25, 30)],
               "NH3_44": [(3, 10)],
               "NH3_55": [(3, 10)],
               "NH2D_33": [(4, 10)],
               "NH2D_44": [(4, 10)]}

imagepath = "/raid/work/yamato/IRAS4A_ammonia/"
for trans in transition:
    print(trans)
    imagename = imagepath + f"IRAS4_{trans}.robust2.0.dv1.0.clean.image.fits"
    with open(imagename.replace(".fits", ".imstat.pickle"), "rb") as f:
        stat = pickle.load(f)
    rms = np.nanmedian(stat["rms"])
    calculate_moment(imagename=imagename, moments=[0], rms=rms, vel_extent=integ_range[trans], save=True)

