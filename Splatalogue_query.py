import numpy as np
from astroquery.splatalogue import Splatalogue
import astropy.units as u

transition = ['NH3_11', 'NH3_22', 'NH3_33', 'NH3_44', 'NH3_55', 'NH2D_33', 'NH2D_44']
# carefullily set by looking at the result of splatalogue query for CDMS hfs lines
hfs_freq_range = {'NH3_11': np.array([23.692, 23.697])*u.GHz,
                  'NH3_22': np.array([23.72, 23.73])*u.GHz,
                  'NH3_33': np.array([23.86, 23.88])*u.GHz,
                  'NH3_44': np.array([24.13, 24.15])*u.GHz,
                  'NH3_55': np.array([24.53, 24.54])*u.GHz,
                  'NH2D_33': np.array([18.80, 18.81])*u.GHz,
                  'NH2D_44': np.array([25.02, 25.03])*u.GHz,
                 }

# needed data columns 
columns = ('Species','Resolved QNs','Freq-GHz(rest frame,redshifted)','Meas Freq-GHz(rest frame,redshifted)',
           'Log<sub>10</sub> (A<sub>ij</sub>)','S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)',
           'Upper State Degeneracy','E_U (K)')

def get_query(trans, hfs=False):
    if 'NH3' in trans:
        mol = ' NH3 '
    elif 'NH2D' in trans:    
        mol = ' NH2D '
    if hfs:
        ll = ['CDMS']
    elif not hfs and 'NH3' in trans:
        ll = ['JPL']
    elif not hfs and 'NH2D' in trans:
        ll = ['CDMS']
    q = Splatalogue.query_lines(hfs_freq_range[trans][0], hfs_freq_range[trans][1], 
                                chemical_name=mol, show_upper_degeneracy=True, 
                                line_strengths=['ls2','ls4'], line_lists=ll)
    return q
    
def select_and_rename_columns(table):
    table = table[columns]
    table.rename_column('Resolved QNs', 'QNs')
    table.rename_column('Freq-GHz(rest frame,redshifted)', 'nu0 [GHz]')
    table.rename_column('Meas Freq-GHz(rest frame,redshifted)', 'Meas nu0 [GHz]')
    table.rename_column('Log<sub>10</sub> (A<sub>ij</sub>)', 'logA [s^-1]')
    table.rename_column('S<sub>ij</sub>&#956;<sup>2</sup> (D<sup>2</sup>)', 'Smu2 [D^2]')
    table.rename_column('Upper State Degeneracy', 'g_u')
    table.rename_column('E_U (K)', 'E_u [K]')
    return table

def resolve_freq_duplication(table):
    table['nu0 [GHz]'] = table['nu0 [GHz]'].astype(float)
    for i in table.mask['nu0 [GHz]'].nonzero():
        table['nu0 [GHz]'][i] = table['Meas nu0 [GHz]'][i]
    table.remove_column('Meas nu0 [GHz]')
    return table

def resolve_hfs_rot_duplication(table, remove_hfs=False):
    hfs_arg = [i for i, qn in enumerate(table['QNs']) if 'F' in qn]
    rot_arg = [i for i, qn in enumerate(table['QNs']) if not 'F' in qn]
    if remove_hfs:
        table.remove_rows(hfs_arg)
    else:
        table.remove_rows(rot_arg)
    return table

def Smu2_to_hfs_ratio(table):
    table['Smu2 [D^2]'] /= table['Smu2 [D^2]'].sum()
    table.rename_column('Smu2 [D^2]', 'hfs ratio')
    return table

def generate_smart_table(trans, table, hfs=False):
    table = select_and_rename_columns(table)
    table = resolve_freq_duplication(table)
    table = resolve_hfs_rot_duplication(table, remove_hfs=not hfs)
    if hfs:
        table = Smu2_to_hfs_ratio(table)
    if 'NH3' in trans and not hfs:
        J = int(trans.replace('NH3_', '')[0])
        K = int(trans.replace('NH3_', '')[1])
        table.add_columns([[J], [K]], names=['J', 'K'])
    return table
        
    
if __name__ == '__main__':
# trim into each rotational transition
    hfsdata = {}
    for trans in transition:
        q = get_query(trans, hfs=True)
        hfsdata[trans] = generate_smart_table(trans, q, hfs=True)
        
    invdata = {}
    for trans in transition:
        q = get_query(trans, hfs=False)
        invdata[trans] = generate_smart_table(trans, q, hfs=False)
    print(hfsdata, invdata)
    

