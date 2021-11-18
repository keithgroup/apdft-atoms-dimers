# MIT License
# 
# Copyright (c) 2021, Alex M. Maldonado
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy import optimize

from apdft_tools.utils import *
from apdft_tools.utils import  _dimer_poly_pred
from apdft_tools.data import prepare_dfs

json_path_atoms = './json-data/atom-pyscf.apdft-data.posthf.json'
json_path_converged_atom = './tests/tests_data/n.chrg0.mult4-pyscf-uccsdt.augccpv5z.json'
json_path_not_converged_atom = './tests/tests_data/c.chrg-2.mult3-pyscf-uccsdt.augccpv5z.json'
json_path_converged_dimer = './tests/tests_data/c.h.chrg0.mult2.sep1.10-pyscf-uccsdt.ccpv5z.json'
json_path_dimers = './json-data/dimer-pyscf.apdft-data.posthf.json'

df_qc_atom, df_qats_atom = prepare_dfs(
    json_path_atoms, get_CBS=False, only_converged=False
)
df_qc_dimer, df_qats_dimer = prepare_dfs(
    json_path_dimers, get_CBS=False, only_converged=False
)

def test_get_lambda_values_atoms():
    z_ref = np.array([7])
    z_target = np.array([9])
    l_value = get_lambda_value(
        z_ref, z_target, specific_atom=None, direction=None
    )
    assert l_value == 2

    z_ref = np.array([7])
    z_target = np.array([6.342])
    l_value = get_lambda_value(
        z_ref, z_target, specific_atom=None, direction=None
    )
    assert l_value == -0.658

def test_add_energies_to_qats_atoms_7elec():
    # Reducing the size of the dataframes.
    df_qc_sys = df_qc_atom.query('n_electrons == 7')
    df_qats_sys = df_qats_atom.query('n_electrons == 7')
    
    df_qats_refs = add_energies_to_df_qats(
        df_qc_sys, df_qats_sys
    )
    assert len(df_qats_refs) == 30
    systems = list(set(df_qats_refs['system'].values))
    systems.sort()
    assert systems == ['b', 'c', 'f', 'n', 'o']
    for i in range(len(df_qats_refs)):
        row = df_qats_refs.iloc[i]
        assert np.allclose(
            np.array([row['electronic_energy']]),
            np.array(row['poly_coeff'])[0]
        )

def test_get_lambda_values_dimers():
    z_ref = np.array([7, 7])
    z_target = np.array([6, 8])
    specific_atom = None
    direction = None
    try:
        l_value = get_lambda_value(
            z_ref, z_target, specific_atom=specific_atom, direction=direction
        )
    except ValueError as e:
        if 'cannot both be None' in str(e):
            pass
        else:
            raise
    except:
        raise

    z_ref = np.array([7, 7])
    z_target = np.array([6, 8])
    specific_atom = None
    direction = 'counter'
    l_value = get_lambda_value(
        z_ref, z_target, specific_atom=specific_atom, direction=direction
    )
    assert l_value == 1
    
def test_qc_ch_fit_parabola():
    # Get data for system
    df_ch = df_qc_dimer.query(
        'system == "c.h" & charge == 0 & lambda_value == 0 & multiplicity == 2'
    )
    bond_lengths = df_ch['bond_length'].values
    energies = df_ch['electronic_energy'].values
    
    bond_lengths_fit, poly_coeffs = fit_dimer_poly(
        bond_lengths, energies, n_points=2, poly_order=4
    )
    assert np.allclose(
        poly_coeffs, np.array(
            [1.5276269877501092, -7.937293532623535, 15.671709523908662, -13.813859032627777, -33.893061130625625]
        )
    )
    poly_energy_test = np.polyval(poly_coeffs, 1.103)
    poly_energy = _dimer_poly_pred(1.103, poly_coeffs)
    assert poly_energy == poly_energy_test

    init_guess = 1.0
    opt_data_test = optimize.minimize(
        _dimer_poly_pred, init_guess, args=(poly_coeffs)
    )
    eq_bond_length_test = opt_data_test.x[0]
    eq_energy_test = opt_data_test.fun
    eq_bond_length, eq_energy = find_poly_min(
        bond_lengths_fit, poly_coeffs
    )

    assert np.allclose(np.array(eq_bond_length), np.array(eq_bond_length_test))
    assert np.allclose(np.array(eq_energy), np.array(eq_energy_test))

