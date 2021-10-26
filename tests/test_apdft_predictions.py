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

import pytest
import numpy as np

from apdft_tools.data import prepare_dfs, get_qc_df_cbs
from apdft_tools.prediction import *

json_path_atoms = './json-data/atom-pyscf.apdft-data.posthf.json'
json_path_dimers = './json-data/dimer-pyscf.apdft-data.posthf.json'

df_qc_atom, df_apdft_atom = prepare_dfs(
    json_path_atoms, get_CBS=False, only_converged=False
)
df_qc_dimer, df_apdft_dimer = prepare_dfs(
    json_path_dimers, get_CBS=False, only_converged=False
)

@pytest.mark.cbs
def prepare_cbs_atom():
    global df_qc_atom_cbs
    global df_apdft_atom_cbs
    df_qc_atom_cbs = get_qc_df_cbs(
        df_qc_atom, cbs_basis_key='aug', basis_set_lower='aug-cc-pVTZ',
        basis_set_higher='aug-cc-pVQZ'
    )

def test_poly_prediction():
    df_state = df_apdft_atom.query(
        'system == "c" & charge == 0 & multiplicity == 3 & basis_set == "aug-cc-pV5Z"'
    )
    assert len(df_state) == 1
    poly_coeff = df_state.iloc[0]['poly_coeff']
    poly_coef_manual = np.array(  # c.chrg0.mult3
        [-37.819523645273655, -14.692113455939193, -1.5080067837658362,
           0.008594331172654771, 0.0012684372071210721]
    )
    assert np.array_equal(poly_coeff, poly_coef_manual)

    # Check prediction of lambda = 1.
    poly_pred_lambda1_manual = np.array([np.sum(poly_coef_manual)])
    poly_pred_lambda1 = calc_apdft_pred(poly_coeff, 4, 1)
    assert type(poly_pred_lambda1) == np.ndarray
    assert np.allclose(
        np.array(poly_pred_lambda1_manual), np.array(-54.0097811165989083569)
    )
    assert poly_pred_lambda1 == poly_pred_lambda1_manual

#########################################
#####     Ionization Potentials     #####
#########################################

def test_n_ip1_qc_correctness():
    target_label = 'n'
    delta_charge = 1
    target_initial_charge = 0
    change_signs = False
    basis_set = 'aug-cc-pV5Z'
    force_same_method = False

    e_chrg0_ground = -54.56266526988731
    e_chrg1_ground = -54.02830363594568
    ip1_manual = e_chrg1_ground - e_chrg0_ground

    ip1_df = get_qc_change_charge(
        df_qc_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        force_same_method=force_same_method
    )
    assert ip1_manual == ip1_df

def test_n_ip1_apdft_correctness():
    target_label = 'n'
    delta_charge = 1
    target_initial_charge = 0
    change_signs = False
    basis_set = 'aug-cc-pV5Z'
    lambda_specific_atom = None
    lambda_direction = None

    e_chrg0_ground_bref = -54.386726697042256  # b.chrg-2.mult4; lambda = 2
    e_chrg1_ground_bref = -53.854313096059066  # b.chrg-1.mult3; lambda = 2
    e_chrg0_ground_cref = -54.54267918907082  # c.chrg-1.mult4; lambda = 1
    e_chrg1_ground_cref = -54.00871704426481  # c.chrg0.mult3; lambda = 1
    e_chrg0_ground_oref = -54.56757847776784  # o.chrg1.mult4; lambda = -1
    e_chrg1_ground_oref = -54.0331576002892  # o.chrg2.mult3; lambda = -1

    ip1_manual_bref = e_chrg1_ground_bref - e_chrg0_ground_bref  # 0.5324136009831903
    ip1_manual_cref = e_chrg1_ground_cref - e_chrg0_ground_cref  # 0.5339621448060043
    ip1_manual_oref = e_chrg1_ground_oref - e_chrg0_ground_oref  # 0.5344208774786381
    ip1_manual = {
        'b': ip1_manual_bref, 'c': ip1_manual_cref, 'o': ip1_manual_oref
    }

    # Alchemical predictions
    use_fin_diff = False
    ip1_apdft = get_apdft_change_charge(
        df_qc_atom, df_apdft_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_fin_diff=use_fin_diff,
        lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction
    )

    ip1_apdft_keys = [i for i in ip1_apdft.keys()]
    ip1_apdft_keys.sort()
    assert ip1_apdft_keys == ['b', 'c', 'o']
    for key in ['b', 'c', 'o']:
        assert np.array_equal(
            ip1_apdft[key], np.array([ip1_manual[key]], dtype='float64')
        )

    # Finite differences
    poly_coef_chrg0_ground_bref = np.array([-24.516510015879703, -11.703773105206672, -1.5721705908866568, -0.05334061163135098, 0.033582914227281435])  # b.chrg-2.mult4; lambda = 2
    poly_coef_chrg1_ground_bref = np.array([-24.63925817507786, -11.634440896839138, -1.571427391162672, 0.053397380443224535, 0.11683278048716754])  # b.chrg-1.mult3; lambda = 2
    poly_coef_chrg0_ground_cref = np.array([-37.86554794646849, -15.03339026537347, -1.671938887000124, 0.0333497744975375, 0.010669006419069168])  # c.chrg-1.mult4; lambda = 1
    poly_coef_chrg1_ground_cref = np.array([-37.819523645273655, -14.692113455939193, -1.5080067837658362, 0.008594331172654771, 0.0012684372071210721])  # c.chrg0.mult3; lambda = 1
    poly_coef_chrg0_ground_oref = np.array([-74.5384357061857, -21.60310587998424, -1.6286223981865078, 0.004562114241934977, 0.0013357611313343416])  # o.chrg1.mult4; lambda = -1
    poly_coef_chrg1_ground_oref = np.array([-73.2470934104361, -20.716783121350346, -1.5003953162562311, 0.0036310462784664797, 0.001525535253676935])  # o.chrg2.mult3; lambda = -1
    ip1_apdft_fin_diff_manual = {
        'b': np.array(
            [calc_apdft_pred(poly_coef_chrg1_ground_bref, i, 2)[0] - calc_apdft_pred(poly_coef_chrg0_ground_bref, i, 2)[0] for i in range(0, 4+1)]
        ),
        'c': np.array(
            [calc_apdft_pred(poly_coef_chrg1_ground_cref, i, 1)[0] - calc_apdft_pred(poly_coef_chrg0_ground_cref, i, 1)[0] for i in range(0, 4+1)]
        ),
        'o': np.array(
            [calc_apdft_pred(poly_coef_chrg1_ground_oref, i, -1)[0] - calc_apdft_pred(poly_coef_chrg0_ground_oref, i, -1)[0] for i in range(0, 4+1)]
        ),
    }

    use_fin_diff = True
    ip1_apdft_fin_diff = get_apdft_change_charge(
        df_qc_atom, df_apdft_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_fin_diff=use_fin_diff,
        lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction
    )

    ip1_apdft_fin_diff_keys = [i for i in ip1_apdft_fin_diff.keys()]
    ip1_apdft_fin_diff_keys.sort()
    assert ip1_apdft_fin_diff_keys == ['b', 'c', 'o']
    for key in ['b', 'c', 'o']:
        assert np.array_equal(
            ip1_apdft_fin_diff[key], ip1_apdft_fin_diff_manual[key]
        )

"""
def test_ch_ip1_qc_correctness():
    target_label = 'c.h'
    delta_charge = 1
    target_initial_charge = 0
    change_signs = False
    basis_set = 'cc-pV5Z'
    force_same_method = False
    bond_length = 1.1

    e_chrg0_ground = -38.45347656174365
    e_chrg1_ground = -38.06281576385865
    ip1_manual = e_chrg1_ground - e_chrg0_ground
    
    ip1_df = get_qc_change_charge(
        df_qc_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        force_same_method=force_same_method,
        bond_length=bond_length
    )
    assert ip1_manual == ip1_df
"""

def test_ch_ip1_qc_dimer_correctness():
    target_label = 'c.h'
    delta_charge = 1
    change_signs = False
    poly_order = 4
    n_points = 2
    remove_outliers = False

    basis_set = 'cc-pV5Z'
    target_initial_charge = 0
    ignore_one_row = True
    zscore_cutoff = 3.0

    # Code used to get the manual value
    """
    df_qc_sys_chrg0 = df_qc.query(
    'system == "c.h" & charge == 0 & lambda_value == 0 & multiplicity == 2'
    )
    df_qc_sys_chrg1 = df_qc.query(
        'system == "c.h" & charge == 1 & lambda_value == 0 & multiplicity == 1'
    )
    bond_lengths_chrg0_eq, e_chrg0_eq = get_dimer_minimum(
        bond_lengths_chrg0, e_chrg0, n_points=n_points, poly_order=poly_order,
        remove_outliers=False, zscore_cutoff=3.0
    )
    bond_lengths_chrg1_eq, e_chrg1_eq = get_dimer_minimum(
        bond_lengths_chrg1, e_chrg1, n_points=n_points, poly_order=poly_order,
        remove_outliers=False, zscore_cutoff=3.0
    )
    ip1_manual = e_chrg1_eq - e_chrg0_eq  # 0.3904478904319859
    """
    ip1_manual = 0.3904478904319859

    ip1_qc = get_qc_change_charge_dimer(
        df_qc_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        ignore_one_row=ignore_one_row, poly_order=poly_order, n_points=n_points,
        remove_outliers=remove_outliers
    )
    
    assert np.allclose(np.array(ip1_qc), np.array(ip1_manual))

def test_bond_lengths_apdft_ch_from_bh():
    lambda_value = 1
    df_apdft_ref = df_apdft_dimer.query(
        'system == "b.h" & charge == -1 & multiplicity == 2'
    )
    bond_lengths_manual = np.array(
        [0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    )
    energies_manual = np.array(
        [-37.95799552604157, -38.194538493752574, -38.32100427733711,
        -38.38512102920283, -38.41367922472698, -38.42187146508282,
        -38.41835428817463, -38.40852928707839, -38.39561480377705,
        -38.38141483069795, -38.36692691826404, -38.35268868360573,
        -38.33896887440571, -38.32589894564336]
    )
    bond_lengths, energies = get_dimer_curve(
        df_apdft_ref, lambda_value, use_fin_diff=True, apdft_order=2
    )
    assert np.array_equal(bond_lengths_manual, bond_lengths)
    assert np.allclose(energies_manual, energies)

def test_ch_ip1_apdft_dimer_correctness():
    target_label = 'c.h'
    delta_charge = 1
    change_signs = False
    poly_order = 4
    n_points = 2
    remove_outliers = True

    basis_set = 'cc-pV5Z'
    target_initial_charge = 0
    lambda_specific_atom = 0
    lambda_direction = None
    ignore_one_row = True


    # Alchemical predictions.
    use_fin_diff = False
    ip1_manual = {
        'b.h': np.array([0.36382197443953146]),
        'n.h': np.array([0.3773032544777948])
    }
    ip1_apdft = get_apdft_change_charge_dimer(
        df_qc_dimer, df_apdft_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        use_fin_diff=use_fin_diff, lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction, ignore_one_row=ignore_one_row,
        poly_order=poly_order, n_points=n_points
    )

    ip1_apdft_keys = [i for i in ip1_apdft.keys()]
    ip1_apdft_keys.sort()
    assert ip1_apdft_keys == ['b.h', 'n.h']
    for key in ['b.h', 'n.h']:
        assert np.allclose(
            ip1_apdft[key], ip1_manual[key]
        )
    
    # Taylor series predictions.
    use_fin_diff = True
    """
    df_apdft_bh_initial = df_apdft.query(
        'system == "b.h" & charge == -1 & multiplicity == 2'
    )
    df_apdft_bh_final = df_apdft.query(
        'system == "b.h" & charge == 0 & multiplicity == 1'
    )
    df_apdft_nh_initial = df_apdft.query(
        'system == "n.h" & charge == 1 & multiplicity == 2'
    )
    df_apdft_nh_final = df_apdft.query(
        'system == "n.h" & charge == 2 & multiplicity == 1'
    )

    apdft_order = 0
    bh_lambda_value = 1
    nh_lambda_value = -1

    bh_bond_lengths_initial, bh_e_initial = get_dimer_curve(
        df_apdft_bh_initial, bh_lambda_value, use_fin_diff=True, apdft_order=apdft_order
    )
    bh_bond_lengths_final, bh_e_final = get_dimer_curve(
        df_apdft_bh_final, bh_lambda_value, use_fin_diff=True, apdft_order=apdft_order
    )
    bh_bond_lengths_initial_eq, bh_e_initial_eq = get_dimer_minimum(
        bh_bond_lengths_initial, bh_e_initial, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    bh_bond_lengths_final_eq, bh_e_final_eq = get_dimer_minimum(
        bh_bond_lengths_final, bh_e_final, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    ip1_bh = bh_e_final_eq - bh_e_initial_eq


    nh_bond_lengths_initial, nh_e_initial = get_dimer_curve(
        df_apdft_nh_initial, nh_lambda_value, use_fin_diff=True, apdft_order=apdft_order
    )
    nh_bond_lengths_final, nh_e_final = get_dimer_curve(
        df_apdft_nh_final, nh_lambda_value, use_fin_diff=True, apdft_order=apdft_order
    )
    nh_bond_lengths_initial_eq, nh_e_initial_eq = get_dimer_minimum(
        nh_bond_lengths_initial, nh_e_initial, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    nh_bond_lengths_final_eq, nh_e_final_eq = get_dimer_minimum(
        nh_bond_lengths_final, nh_e_final, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    ip1_nh = nh_e_final_eq - nh_e_initial_eq
    """
    ip1_manual = {
        'b.h': np.array(
            [-0.00189905333453666, 0.28264753831965805, 0.38318080118576603, 0.406652661093311, 0.4339253282625464]
        ),
        'n.h': np.array(
            [0.972005617442953, 0.29952171995040544, 0.3890778890175213, 0.3871436111615978, 0.37807340130896705]
        )
    }

    ip1_apdft = get_apdft_change_charge_dimer(
        df_qc_dimer, df_apdft_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        use_fin_diff=use_fin_diff, lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction, ignore_one_row=ignore_one_row,
        poly_order=poly_order, n_points=n_points, remove_outliers=remove_outliers
    )

    ip1_apdft_keys = [i for i in ip1_apdft.keys()]
    ip1_apdft_keys.sort()
    assert ip1_apdft_keys == ['b.h', 'n.h']
    for key in ['b.h', 'n.h']:
        assert np.allclose(
            ip1_apdft[key], ip1_manual[key]
        )



#######################################
#####     Electron Affinities     #####
#######################################

def test_n_ea_qc_correctness():
    target_label = 'n'
    delta_charge = -1
    target_initial_charge = 0
    change_signs = True
    basis_set = 'aug-cc-pV5Z'
    force_same_method = False

    e_chrg0_ground = -54.56266526988731
    e_chrg_neg1_ground = -54.555280878714306
    ea_manual = -(e_chrg_neg1_ground - e_chrg0_ground)

    ea_df = get_qc_change_charge(
        df_qc_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        force_same_method=force_same_method
    )
    assert ea_manual == ea_df

def test_n_ea_apdft_correctness():
    target_label = 'n'
    delta_charge = -1
    target_initial_charge = 0
    change_signs = True
    basis_set = 'aug-cc-pV5Z'
    bond_length = None
    lambda_specific_atom = None
    lambda_direction = None

    e_chrg0_ground_cref = -54.54267918907082  # c.chrg-1.mult4; lambda = 1
    e_chrg_neg1_ground_cref = -54.53648723162306  # c.chrg-2.mult3; lambda = 1
    e_chrg0_ground_oref = -54.56757847776784  # o.chrg1.mult4; lambda = -1
    e_chrg_neg1_ground_oref = -54.559746109512936  # o.chrg0.mult3; lambda = -1
    e_chrg0_ground_fref = -54.5717501791145  # f.chrg2.mult4; lambda = -2
    e_chrg_neg1_ground_fref = -54.56288659430845  # f.chrg1.mult3; lambda = -2

    ea_manual = {
        'c': -(e_chrg_neg1_ground_cref - e_chrg0_ground_cref),  # -0.006191957447754248
        'o': -(e_chrg_neg1_ground_oref - e_chrg0_ground_oref),  # -0.007832368254902633
        'f': -(e_chrg_neg1_ground_fref - e_chrg0_ground_fref)  # -0.00886358480605054
    }
    
    # Alchemical predictions
    use_fin_diff = False
    ea_apdft = get_apdft_change_charge(
        df_qc_atom, df_apdft_atom, target_label, delta_charge, bond_length=bond_length,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_fin_diff=use_fin_diff,
        lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction
    )

    ea_apdft_keys = [i for i in ea_apdft.keys()]
    ea_apdft_keys.sort()
    assert ea_apdft_keys == ['c', 'f', 'o']
    for key in ['c', 'f', 'o']:
        assert np.array_equal(
            ea_apdft[key], np.array([ea_manual[key]], dtype='float64')
        )
    
    # Finite differences
    poly_coef_chrg0_ground_cref = np.array([-37.86554794646849, -15.03339026537347, -1.671938887000124, 0.0333497744975375, 0.010669006419069168])  # c.chrg-1.mult4; lambda = 1
    poly_coef_chrg1_ground_cref = np.array([-37.679969561141824, -15.144953045097864, -1.597639944073137, -1.3682196199719476, 11.982575317167251])  # c.chrg-2.mult3; lambda = 1
    poly_coef_chrg0_ground_oref = np.array([-74.5384357061857, -21.60310587998424, -1.6286223981865078, 0.004562114241934977, 0.0013357611313343416])  # o.chrg1.mult4; lambda = -1
    poly_coef_chrg1_ground_oref = np.array([-75.03715953573213, -22.253246218983946, -1.764760111413466, 0.008784252732615034, -0.00036652162786291836])  # o.chrg0.mult3; lambda = -1
    poly_coef_chrg0_ground_fref = np.array([-97.77796109795331, -24.863702563040846, -1.62636013307349, 0.003370059194670223, 0.0011967316027039487])  # f.chrg2.mult4; lambda = -2
    poly_coef_chrg1_ground_fref = np.array([-99.06063033329923, -25.77866047114199, -1.7558912191617537, 0.004650575628299217, 0.0008996655272615802])  # f.chrg1.mult3; lambda = -2
    ea_apdft_fin_diff_manual = {
        'c': np.array(
            [-(calc_apdft_pred(poly_coef_chrg1_ground_cref, i, 1)[0] - calc_apdft_pred(poly_coef_chrg0_ground_cref, i, 1)[0]) for i in range(0, 4+1)]
        ),
        'o': np.array(
            [-(calc_apdft_pred(poly_coef_chrg1_ground_oref, i, -1)[0] - calc_apdft_pred(poly_coef_chrg0_ground_oref, i, -1)[0]) for i in range(0, 4+1)]
        ),
        'f': np.array(
            [-(calc_apdft_pred(poly_coef_chrg1_ground_fref, i, -2)[0] - calc_apdft_pred(poly_coef_chrg0_ground_fref, i, -2)[0]) for i in range(0, 4+1)]
        ),
    }

    use_fin_diff = True
    ea_apdft_fin_diff = get_apdft_change_charge(
        df_qc_atom, df_apdft_atom, target_label, delta_charge, bond_length=bond_length,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_fin_diff=use_fin_diff,
        lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction
    )
    ip1_apdft_fin_diff_keys = [i for i in ea_apdft_fin_diff.keys()]
    ip1_apdft_fin_diff_keys.sort()
    assert ip1_apdft_fin_diff_keys == ['c', 'f', 'o']
    for key in ['c', 'f', 'o']:
        assert np.array_equal(
            ea_apdft_fin_diff[key], ea_apdft_fin_diff_manual[key]
        )


#######################################
#####     Excitation Energies     #####
#######################################

def test_n_ee_qc_correctness():
    target_label = 'n'
    target_charge = 0
    excitation_level = 1
    basis_set = 'aug-cc-pV5Z'

    e_ground = -54.56266526988731
    e_excited = -54.46422759392469
    ea_manual = e_excited - e_ground

    ea_df = get_qc_excitation(
        df_qc_atom, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set
    )
    assert ea_manual == ea_df

def test_n_ee_apdft_correctness():
    target_label = 'n'
    target_charge = 0
    excitation_level = 1
    basis_set = 'aug-cc-pV5Z'

    e_ground_bref = -54.386726697042256  # b.chrg-2.mult4; lambda = 2
    e_excited_bref = -54.28802831832811  # b.chrg-2.mult2; lambda = 2
    e_ground_cref = -54.54267918907082  # c.chrg-1.mult4; lambda = 1
    e_excited_cref = -54.44426574929463  # c.chrg-1.mult2; lambda = 1
    e_ground_oref = -54.56757847776784  # o.chrg1.mult4; lambda = -1
    e_excited_oref = -54.46912355430326  # o.chrg1.mult2; lambda = -1
    e_ground_fref = -54.5717501791145  # f.chrg2.mult4; lambda = -2
    e_excited_fref = -54.47325984455847  # f.chrg2.mult2; lambda = -2

    ee_manual = {
        'b': e_excited_bref - e_ground_bref,
        'c': e_excited_cref - e_ground_cref,
        'o': e_excited_oref - e_ground_oref,
        'f': e_excited_fref - e_ground_fref,
    }

    # Alchemical predictions
    use_fin_diff = False
    ee_apdft = get_apdft_excitation(
        df_qc_atom, df_apdft_atom, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set,
        use_fin_diff=use_fin_diff
    )

    ee_apdft_keys = [i for i in ee_apdft.keys()]
    ee_apdft_keys.sort()
    assert ee_apdft_keys == ['b', 'c', 'f', 'o']
    for key in ['b', 'c', 'f', 'o']:
        assert np.array_equal(
            ee_apdft[key], np.array([ee_manual[key]], dtype='float64')
        )

    # Finite differences
    poly_coef_ground_bref = np.array([-24.516510015879703, -11.703773105206672, -1.5721705908866568, -0.05334061163135098, 0.033582914227281435])  # b.chrg-2.mult4; lambda = 2
    poly_coef_excited_bref = np.array([ -24.50499686168636, -11.70538276387294, -1.6316110076708412, -0.13452204653911318, 2.0478839365030885 ])  # b.chrg-2.mult2; lambda = 2
    poly_coef_ground_cref = np.array([-37.86554794646849, -15.03339026537347, -1.671938887000124, 0.0333497744975375, 0.010669006419069168])  # c.chrg-1.mult4; lambda = 1
    poly_coef_excited_cref = np.array([ -37.81108471386423, -14.967446960954334, -1.448842561231345, -20.192879301698476, -1333.7607380906982 ])  # c.chrg-1.mult2; lambda = 1
    poly_coef_ground_oref = np.array([-74.5384357061857, -21.60310587998424, -1.6286223981865078, 0.004562114241934977, 0.0013357611313343416])  # o.chrg1.mult4; lambda = -1
    poly_coef_excited_oref = np.array([ -74.40523350914746, -21.569885650943377, -1.6296778545665802, 0.004892941755466987, 0.0012180478847767517 ])  # o.chrg1.mult2; lambda = -1
    poly_coef_ground_fref = np.array([ -97.77796109795331, -24.863702563040846, -1.62636013307349, 0.003370059194670223, 0.0011967316027039487 ])  # f.chrg2.mult4; lambda = -2
    poly_coef_excited_fref = np.array([ -97.61229899821406, -24.83193069401466, -1.6268626619364568, 0.0034881869244903396, 0.0011946591863913152 ])  # f.chrg2.mult2; lambda = -2
    ea_apdft_fin_diff_manual = {
        'b': np.array(
            [calc_apdft_pred(poly_coef_excited_bref, i, 2)[0] - calc_apdft_pred(poly_coef_ground_bref, i, 2)[0] for i in range(0, 4+1)]
        ),
        'c': np.array(
            [calc_apdft_pred(poly_coef_excited_cref, i, 1)[0] - calc_apdft_pred(poly_coef_ground_cref, i, 1)[0] for i in range(0, 4+1)]
        ),
        'o': np.array(
            [calc_apdft_pred(poly_coef_excited_oref, i, -1)[0] - calc_apdft_pred(poly_coef_ground_oref, i, -1)[0] for i in range(0, 4+1)]
        ),
        'f': np.array(
            [calc_apdft_pred(poly_coef_excited_fref, i, -2)[0] - calc_apdft_pred(poly_coef_ground_fref, i, -2)[0] for i in range(0, 4+1)]
        ),
    }

    use_fin_diff = True
    ee_apdft_fin_diff = get_apdft_excitation(
        df_qc_atom, df_apdft_atom, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set,
        use_fin_diff=use_fin_diff
    )
    ee_apdft_fin_diff_keys = [i for i in ee_apdft_fin_diff.keys()]
    ee_apdft_fin_diff_keys.sort()
    assert ee_apdft_fin_diff_keys == ['b', 'c', 'f', 'o']
    for key in ['b', 'c', 'f', 'o']:
        assert np.array_equal(
            ee_apdft_fin_diff[key], ea_apdft_fin_diff_manual[key]
        )


test_ch_ip1_apdft_dimer_correctness()