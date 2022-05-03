import pickle 
from plot_utils import get_param_names, get_exp_notation
from analysis_utils import get_median_uncertainty

samplepath = "./data/fit/"

transition = ["NH3_33", "NH3_44", "NH3_55", "NH2D_33", "NH2D_44"]

source = ["IRAS4A1", "IRAS4A2"]
fit_type = ["OPRfixed", "OPRfree"]

fiducial = {"IRAS4A1": "OPRfixed", "IRAS4A2": "OPRfree"}

nburnin = 2500

table_str = ""

for s in source:

    table_str += "\\hline \\multicolumn{12}{c}{" + s + "} \\\\ \\hline \n"
    
    for ft in fit_type:

        row_str_list = []

        OPR_NH3_free = "free" in ft 

        # load the mcmc sample and seed the random number
        filename = samplepath + "{}_{}.pkl".format(s, ft)
        with open(filename, 'rb') as f:
            sample = pickle.load(f)
        sample = sample[nburnin:].reshape(-1, sample.shape[-1])

        param_names = get_param_names(transition, OPR_NH3_free=OPR_NH3_free)

        # OPR
        if OPR_NH3_free:
            q = get_median_uncertainty(sample[:,param_names.index("OPR_NH3")])
            row_str_list.append("${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(*q))
        else:
            row_str_list.append("1 (fixed)")

        # theta_s
        q = get_median_uncertainty(sample[:,param_names.index("theta_s")])
        row_str_list.append("{:.2f}_{{-{:.2f}}}^{{+{:.2f}}}".format(*q))

        # Tex
        q = get_median_uncertainty(sample[:,param_names.index("Tex")])
        row_str_list.append("{:.0f}_{{-{:.0f}}}^{{+{:.0f}}}".format(*q))

        # Delta V (FWHM) 
        for trans in transition:
            q = get_median_uncertainty(sample[:,param_names.index("dv_FWHM_{:s}".format(trans))])
            row_str_list.append("{:.2f}_{{-{:.2f}}}^{{+{:.2f}}}".format(*q))

        # NH3 column density 
        q = get_median_uncertainty(10 ** sample[:,param_names.index("logN_NH3")])
        row_str_list.append("{:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \\times 10^{{{:d}}}".format(*get_exp_notation(*q)))

        # NH2D column density 
        q = get_median_uncertainty(10 ** sample[:,param_names.index("logN_NH2D")])
        row_str_list.append("{:.1f}_{{-{:.1f}}}^{{+{:.1f}}} \\times 10^{{{:d}}}".format(*get_exp_notation(*q)))

        # NH2D/NH3
        ratio_sample = 10 ** sample[:,param_names.index("logN_NH2D")] / 10 ** sample[:,param_names.index("logN_NH3")]
        q = get_median_uncertainty(ratio_sample)
        row_str_list.append("{:.2f}_{{-{:.2f}}}^{{+{:.2f}}}".format(*q))


        # join
        table_str += " & ".join(row_str_list) + " & "

        if ft == fiducial[s]:
            table_str += "\\checkmark"

        table_str += "\\\\ \n"

with open("./data/LaTeX/fit_table.txt", "w") as f:
    f.write(table_str)


