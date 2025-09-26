import casatasks
import numpy as np

imagename = "/raid/work/yamato/IRAS4A_ammonia/IRAS4_cont.robust0.5.clean.image.pbcor.fits"
# region = "circle[[3h29m10.440s, 31d13m32.160s], 0.3arcsec]" # for IRAS4A2
region = "circle[[3h29m10.540s, 31d13m30.850s], 0.3arcsec]" # for IRAS4A1
model = "./data/fit/IRAS4_cont.robust0.5.clean.image.pbcor.2DGaussFit.model.image"
residual = "./data/fit/IRAS4_cont.robust0.5.clean.image.pbcor.2DGaussFit.residual.image"
estimates = "./data/fit/contfit_estimates_4A1.txt"
logfile = "./data/fit/contfit_log.txt"
rms = 9.6e-6

casatasks.imfit(imagename=imagename, region=region, model=model, residual=residual, estimates=estimates, logfile=logfile, rms=rms)
