# labels for transitions
trans_label = {'NH3_11': '$\mathrm{NH_3}$ $(1,1)$', 
               'NH3_22': '$\mathrm{NH_3}$ $(2,2)$', 
               'NH3_33': '$\mathrm{NH_3}$ $(3,3)$', 
               'NH3_44': '$\mathrm{NH_3}$ $(4,4)$', 
               'NH3_55': '$\mathrm{NH_3}$ $(5,5)$',
               'NH2D_33': u'$\mathrm{NH_2D}$ $3_{1,3}$\u2013$3_{0,3}$',
               'NH2D_44': u'$\mathrm{NH_2D}$ $4_{1,4}$\u2013$4_{0,4}$',
              }

source_label = {'IRAS4A1': '4A1',
				'IRAS4A2': '4A2',
                'IRAS4B': '4B'}

fiducial_fit = {"IRAS4A1": "OPRfixed", "IRAS4A2": "OPRfree",}

nburnin = 2500


def get_param_names(transition, OPR_NH3_free=True):

    param_names = ["dv_FWHM_{:s}".format(trans) for trans in transition]
    param_names += ["v_sys_{:s}".format(trans) for trans in transition]
    param_names += ["logN_NH3"]
    if OPR_NH3_free:
        param_names += ["OPR_NH3"]
    param_names += ["logN_NH2D", "Tex", "theta_s"]

    return param_names

def get_param_labels(transition, OPR_NH3_free=True):

    param_labels = ["$\\Delta V$ ({:s})\n[km s$^{{-1}}$]".format(trans_label[trans]) for trans in transition] # velocity dispersion (FWHM)
    param_labels += ["$v_\\mathrm{{sys}}$ ({:s})\n[km s$^{{-1}}$]".format(trans_label[trans]) for trans in transition] # systemic velocity
    param_labels += ["$\\log_{10}N(\\mathrm{NH_3})$\n[cm$^{-2}$]"]
    if OPR_NH3_free:
        param_labels += ["$N$(o-NH$_3$) / $N$(p-NH$_3$)"]
    param_labels += ["$\\log_{10}N(\\mathrm{NH_2D})$\n[cm$^{-2}$]", "$T_\\mathrm{ex}$ [K]", "$\\theta_\\mathrm{s}$ [$^{\\prime\\prime}$]"]

    return param_labels


def get_significand_exponent(value):
    exp = len(str(int(value))) - 1
    sign = value / 10 ** exp
    return sign, exp


def get_exp_notation(median, uc_lower, uc_upper):
    s_med, exp = get_significand_exponent(median)
    s_l = uc_lower / 10 ** exp
    s_u = uc_upper / 10 ** exp
    return s_med, s_l, s_u, exp
    