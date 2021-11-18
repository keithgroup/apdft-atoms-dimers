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

from qa_tools.data import prepare_dfs, get_qc_df_cbs
from qa_tools.prediction import *
from qa_tools.utils import *

json_path_atoms = './json-data/atom-pyscf.qa-data.posthf.json'
json_path_dimers = './json-data/dimer-pyscf.qa-data.posthf.json'

df_qc_atom, df_qats_atom = prepare_dfs(
    json_path_atoms, get_CBS=False, only_converged=False
)
df_qc_dimer, df_qats_dimer = prepare_dfs(
    json_path_dimers, get_CBS=False, only_converged=False
)

@pytest.mark.cbs
def prepare_cbs_atom():
    global df_qc_atom_cbs
    global df_qats_atom_cbs
    df_qc_atom_cbs = get_qc_df_cbs(
        df_qc_atom, cbs_basis_key='aug', basis_set_lower='aug-cc-pVTZ',
        basis_set_higher='aug-cc-pVQZ'
    )

def test_poly_prediction():
    df_state = df_qats_atom.query(
        'system == "c" & charge == 0 & multiplicity == 3 & basis_set == "aug-cc-pV5Z"'
    )
    assert len(df_state) == 1
    poly_coeff = df_state.iloc[0]['poly_coeffs']
    poly_coef_manual = np.array(  # c.chrg0.mult3
        [-37.819523645273655, -14.692113455939193, -1.5080067837658362,
           0.008594331172654771, 0.0012684372071210721]
    )
    assert np.array_equal(poly_coeff, poly_coef_manual)

    # Check prediction of lambda = 1.
    poly_pred_lambda1_manual = np.array([np.sum(poly_coef_manual)])
    poly_pred_lambda1 = qats_prediction(poly_coeff, 4, 1)
    assert type(poly_pred_lambda1) == np.ndarray
    assert np.allclose(
        np.array(poly_pred_lambda1_manual), np.array(-54.0097811165989083569)
    )
    assert poly_pred_lambda1 == poly_pred_lambda1_manual

#########################################
#####     ionization energies     #####
#########################################

def test_n_ie1_qc_correctness():
    target_label = 'n'
    delta_charge = 1
    target_initial_charge = 0
    change_signs = False
    basis_set = 'aug-cc-pV5Z'

    e_chrg0_ground = -54.56266526988731
    e_chrg1_ground = -54.02830363594568
    ie1_manual = e_chrg1_ground - e_chrg0_ground

    ie1_df = energy_change_charge_qc_atom(
        df_qc_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set
    )
    assert ie1_manual == ie1_df

def test_n_ie1_qats_correctness():
    target_label = 'n'
    delta_charge = 1
    target_initial_charge = 0
    change_signs = False
    basis_set = 'aug-cc-pV5Z'

    e_chrg0_ground_bref = -54.386726697042256  # b.chrg-2.mult4; lambda = 2
    e_chrg1_ground_bref = -53.854313096059066  # b.chrg-1.mult3; lambda = 2
    e_chrg0_ground_cref = -54.54267918907082  # c.chrg-1.mult4; lambda = 1
    e_chrg1_ground_cref = -54.00871704426481  # c.chrg0.mult3; lambda = 1
    e_chrg0_ground_oref = -54.56757847776784  # o.chrg1.mult4; lambda = -1
    e_chrg1_ground_oref = -54.0331576002892  # o.chrg2.mult3; lambda = -1

    ie1_manual_bref = e_chrg1_ground_bref - e_chrg0_ground_bref  # 0.5324136009831903
    ie1_manual_cref = e_chrg1_ground_cref - e_chrg0_ground_cref  # 0.5339621448060043
    ie1_manual_oref = e_chrg1_ground_oref - e_chrg0_ground_oref  # 0.5344208774786381
    ie1_manual = {
        'b': ie1_manual_bref, 'c': ie1_manual_cref, 'o': ie1_manual_oref
    }

    # Alchemical predictions
    use_ts = False
    ie1_qats = energy_change_charge_qa_atom(
        df_qc_atom, df_qats_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts,
    )

    ie1_qats_keys = [i for i in ie1_qats.keys()]
    ie1_qats_keys.sort()
    assert ie1_qats_keys == ['b', 'c', 'o']
    for key in ['b', 'c', 'o']:
        assert np.array_equal(
            ie1_qats[key], np.array([ie1_manual[key]], dtype='float64')
        )

    # Finite differences
    poly_coef_chrg0_ground_bref = np.array([-24.516510015879703, -11.703773105206672, -1.5721705908866568, -0.05334061163135098, 0.033582914227281435])  # b.chrg-2.mult4; lambda = 2
    poly_coef_chrg1_ground_bref = np.array([-24.63925817507786, -11.634440896839138, -1.571427391162672, 0.053397380443224535, 0.11683278048716754])  # b.chrg-1.mult3; lambda = 2
    poly_coef_chrg0_ground_cref = np.array([-37.86554794646849, -15.03339026537347, -1.671938887000124, 0.0333497744975375, 0.010669006419069168])  # c.chrg-1.mult4; lambda = 1
    poly_coef_chrg1_ground_cref = np.array([-37.819523645273655, -14.692113455939193, -1.5080067837658362, 0.008594331172654771, 0.0012684372071210721])  # c.chrg0.mult3; lambda = 1
    poly_coef_chrg0_ground_oref = np.array([-74.5384357061857, -21.60310587998424, -1.6286223981865078, 0.004562114241934977, 0.0013357611313343416])  # o.chrg1.mult4; lambda = -1
    poly_coef_chrg1_ground_oref = np.array([-73.2470934104361, -20.716783121350346, -1.5003953162562311, 0.0036310462784664797, 0.001525535253676935])  # o.chrg2.mult3; lambda = -1
    ie1_qats_manual = {
        'b': np.array(
            [qats_prediction(poly_coef_chrg1_ground_bref, i, 2)[0] - qats_prediction(poly_coef_chrg0_ground_bref, i, 2)[0] for i in range(0, 4+1)]
        ),
        'c': np.array(
            [qats_prediction(poly_coef_chrg1_ground_cref, i, 1)[0] - qats_prediction(poly_coef_chrg0_ground_cref, i, 1)[0] for i in range(0, 4+1)]
        ),
        'o': np.array(
            [qats_prediction(poly_coef_chrg1_ground_oref, i, -1)[0] - qats_prediction(poly_coef_chrg0_ground_oref, i, -1)[0] for i in range(0, 4+1)]
        ),
    }

    use_ts = True
    ie1_qats = energy_change_charge_qa_atom(
        df_qc_atom, df_qats_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts,
    )

    ie1_qats_keys = [i for i in ie1_qats.keys()]
    ie1_qats_keys.sort()
    assert ie1_qats_keys == ['b', 'c', 'o']
    for key in ['b', 'c', 'o']:
        assert np.array_equal(
            ie1_qats[key], ie1_qats_manual[key]
        )

def test_ch_ie1_qc_dimer_correctness():
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
    bond_lengths_chrg0_eq, e_chrg0_eq = dimer_minimum(
        bond_lengths_chrg0, e_chrg0, n_points=n_points, poly_order=poly_order,
        remove_outliers=False, zscore_cutoff=3.0
    )
    bond_lengths_chrg1_eq, e_chrg1_eq = dimer_minimum(
        bond_lengths_chrg1, e_chrg1, n_points=n_points, poly_order=poly_order,
        remove_outliers=False, zscore_cutoff=3.0
    )
    ie1_manual = e_chrg1_eq - e_chrg0_eq  # 0.3904478904319859
    """
    ie1_manual = 0.3904478904319859

    ie1_qc = get_qc_change_charge_dimer(
        df_qc_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        ignore_one_row=ignore_one_row, poly_order=poly_order, n_points=n_points,
        remove_outliers=remove_outliers
    )
    
    assert np.allclose(np.array(ie1_qc), np.array(ie1_manual))

def test_bond_lengths_qats_ch_from_bh():
    lambda_value = 1
    df_qats_ref = df_qats_dimer.query(
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
    bond_lengths, energies = dimer_curve(
        df_qats_ref, lambda_value=lambda_value, use_ts=True, qats_order=2,
    )
    assert np.array_equal(bond_lengths_manual, bond_lengths)
    assert np.allclose(energies_manual, energies)

def test_ch_ie1_qats_dimer_correctness():
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
    use_ts = False
    ie1_manual = {
        'b.h': np.array([0.3903041968078611]),
        'n.h': np.array([0.38959475994708725])
    }
    ie1_qats = get_qa_change_charge_dimer(
        df_qc_dimer, df_qats_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        use_ts=use_ts, lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction, ignore_one_row=ignore_one_row,
        poly_order=poly_order, n_points=n_points
    )

    ie1_qats_keys = [i for i in ie1_qats.keys()]
    ie1_qats_keys.sort()
    assert ie1_qats_keys == ['b.h', 'n.h']
    for key in ['b.h', 'n.h']:
        assert np.allclose(
            ie1_qats[key], ie1_manual[key]
        )
    
    # Taylor series predictions.
    use_ts = True
    """
    df_qats_bh_initial = df_qats.query(
        'system == "b.h" & charge == -1 & multiplicity == 2'
    )
    df_qats_bh_final = df_qats.query(
        'system == "b.h" & charge == 0 & multiplicity == 1'
    )
    df_qats_nh_initial = df_qats.query(
        'system == "n.h" & charge == 1 & multiplicity == 2'
    )
    df_qats_nh_final = df_qats.query(
        'system == "n.h" & charge == 2 & multiplicity == 1'
    )

    qats_order = 0
    bh_lambda_value = 1
    nh_lambda_value = -1

    bh_bond_lengths_initial, bh_e_initial = dimer_curve(
        df_qats_bh_initial, lambda_value=bh_lambda_value, use_ts=True, qats_order=qats_order
    )
    bh_bond_lengths_final, bh_e_final = dimer_curve(
        df_qats_bh_final, lambda_value=bh_lambda_value, use_ts=True, qats_order=qats_order
    )
    bh_bond_lengths_initial_eq, bh_e_initial_eq = dimer_minimum(
        bh_bond_lengths_initial, bh_e_initial, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    bh_bond_lengths_final_eq, bh_e_final_eq = dimer_minimum(
        bh_bond_lengths_final, bh_e_final, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    ie1_bh = bh_e_final_eq - bh_e_initial_eq
    print(ie1_bh)

    

    nh_bond_lengths_initial, nh_e_initial = dimer_curve(
        df_qats_nh_initial, lambda_value=nh_lambda_value, use_ts=True, qats_order=qats_order
    )
    nh_bond_lengths_final, nh_e_final = dimer_curve(
        df_qats_nh_final, lambda_value=nh_lambda_value, use_ts=True, qats_order=qats_order
    )
    nh_bond_lengths_initial_eq, nh_e_initial_eq = dimer_minimum(
        nh_bond_lengths_initial, nh_e_initial, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    nh_bond_lengths_final_eq, nh_e_final_eq = dimer_minimum(
        nh_bond_lengths_final, nh_e_final, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=3.0
    )
    ie1_nh = nh_e_final_eq - nh_e_initial_eq
    print(ie1_nh)
    """
    ie1_manual = {
        'b.h': np.array(
            [-0.00189905333453666, 0.2826475383231539, 0.3831808011831157, 0.40665266108710085, 0.4339253282625464]
        ),
        'n.h': np.array(
            [0.9720056174429033, 0.29952171994964516, 0.37311779315795945, 0.3599671271306164, 0.22844150662677265]
        )
    }

    ie1_qats = get_qa_change_charge_dimer(
        df_qc_dimer, df_qats_dimer, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        use_ts=use_ts, lambda_specific_atom=lambda_specific_atom,
        lambda_direction=lambda_direction, ignore_one_row=ignore_one_row,
        poly_order=poly_order, n_points=n_points, remove_outliers=remove_outliers
    )

    ie1_qats_keys = [i for i in ie1_qats.keys()]
    ie1_qats_keys.sort()
    assert ie1_qats_keys == ['b.h', 'n.h']
    for key in ['b.h', 'n.h']:
        assert np.allclose(
            ie1_qats[key], ie1_manual[key]
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

    e_chrg0_ground = -54.56266526988731
    e_chrg_neg1_ground = -54.555280878714306
    ea_manual = -(e_chrg_neg1_ground - e_chrg0_ground)

    ea_df = energy_change_charge_qc_atom(
        df_qc_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set
    )
    assert ea_manual == ea_df

def test_n_ea_qats_correctness():
    target_label = 'n'
    delta_charge = -1
    target_initial_charge = 0
    change_signs = True
    basis_set = 'aug-cc-pV5Z'

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
    use_ts = False
    ea_qats = energy_change_charge_qa_atom(
        df_qc_atom, df_qats_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts
    )

    ea_qats_keys = [i for i in ea_qats.keys()]
    ea_qats_keys.sort()
    assert ea_qats_keys == ['c', 'f', 'o']
    for key in ['c', 'f', 'o']:
        assert np.array_equal(
            ea_qats[key], np.array([ea_manual[key]], dtype='float64')
        )
    
    # Finite differences
    poly_coef_chrg0_ground_cref = np.array([-37.86554794646849, -15.03339026537347, -1.671938887000124, 0.0333497744975375, 0.010669006419069168])  # c.chrg-1.mult4; lambda = 1
    poly_coef_chrg1_ground_cref = np.array([-37.679969561141824, -15.144953045097864, -1.597639944073137, -1.3682196199719476, 11.982575317167251])  # c.chrg-2.mult3; lambda = 1
    poly_coef_chrg0_ground_oref = np.array([-74.5384357061857, -21.60310587998424, -1.6286223981865078, 0.004562114241934977, 0.0013357611313343416])  # o.chrg1.mult4; lambda = -1
    poly_coef_chrg1_ground_oref = np.array([-75.03715953573213, -22.253246218983946, -1.764760111413466, 0.008784252732615034, -0.00036652162786291836])  # o.chrg0.mult3; lambda = -1
    poly_coef_chrg0_ground_fref = np.array([-97.77796109795331, -24.863702563040846, -1.62636013307349, 0.003370059194670223, 0.0011967316027039487])  # f.chrg2.mult4; lambda = -2
    poly_coef_chrg1_ground_fref = np.array([-99.06063033329923, -25.77866047114199, -1.7558912191617537, 0.004650575628299217, 0.0008996655272615802])  # f.chrg1.mult3; lambda = -2
    ea_qats_manual = {
        'c': np.array(
            [-(qats_prediction(poly_coef_chrg1_ground_cref, i, 1)[0] - qats_prediction(poly_coef_chrg0_ground_cref, i, 1)[0]) for i in range(0, 4+1)]
        ),
        'o': np.array(
            [-(qats_prediction(poly_coef_chrg1_ground_oref, i, -1)[0] - qats_prediction(poly_coef_chrg0_ground_oref, i, -1)[0]) for i in range(0, 4+1)]
        ),
        'f': np.array(
            [-(qats_prediction(poly_coef_chrg1_ground_fref, i, -2)[0] - qats_prediction(poly_coef_chrg0_ground_fref, i, -2)[0]) for i in range(0, 4+1)]
        ),
    }

    use_ts = True
    ea_qats = energy_change_charge_qa_atom(
        df_qc_atom, df_qats_atom, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts
    )
    ie1_qats_keys = [i for i in ea_qats.keys()]
    ie1_qats_keys.sort()
    assert ie1_qats_keys == ['c', 'f', 'o']
    for key in ['c', 'f', 'o']:
        assert np.array_equal(
            ea_qats[key], ea_qats_manual[key]
        )


############################################
#####     Bond Curves and Energies     #####
############################################


def test_oh_from_ne_from_qc():
    target_label = 'o.h'
    target_charge = 0
    excitation_level = 0
    specific_atom = 0
    basis_set = 'cc-pV5Z'

    use_ts = False
    qats_order = 2

    df_qc_system = df_qc_dimer.query(
        'system == @target_label'
        '& charge == @target_charge'
    )
    target_n_electrons = df_qc_system.iloc[0]['n_electrons']
    target_atomic_numbers = df_qc_system.iloc[0]['atomic_numbers']

    # Alchemical predictions
    df_selection = 'qc'
    df_references = get_qa_refs(
        df_qc_dimer, df_qats_dimer, target_label, target_n_electrons,
        basis_set=basis_set, df_selection=df_selection,
        excitation_level=excitation_level, specific_atom=specific_atom
    )

    ref_systems = list(set(df_references['system']))
    ref_systems.sort()
    assert ref_systems == ['f.h', 'n.h', 'ne.h']

    df_ref_neh = df_references.query('system == "ne.h"')
    assert df_ref_neh.iloc[0]['multiplicity'] == 2

    neh_energies_manual = np.array([
        -75.37531839629389, -75.58104544895917, -75.67288660526044,
        -75.70648616566774, -75.71023924594704, -75.69903899608731,
        -75.68091242285723, -75.66020154595432, -75.63919901702702,
        -75.62038369356912, -75.60324660673356, -75.58851756031235,
        -75.57635994999121, -75.47142171254076
    ])
    _, neh_energies = dimer_curve(
        df_ref_neh, lambda_value=-2
    )
    assert np.allclose(neh_energies, neh_energies_manual)


def test_oh_eq():
    system_label = 'o.h'
    system_charge = 0
    basis_set = 'cc-pV5Z'
    n_points = 2
    poly_order = 4
    remove_outliers = False
    considered_lambdas = None

    # Quantum chemistry
    calc_type = 'qc'
    use_ts = False
    bl_manual = {'o.h': np.array([0.9675974853219637])}
    e_manual = {'o.h': np.array([-75.7069928879804])}

    bl_test, e_test = dimer_eq(
        df_qc_dimer, system_label, system_charge, calc_type=calc_type, use_ts=use_ts,
        df_qats=df_qats_dimer,
        basis_set=basis_set,
        n_points=n_points, poly_order=poly_order, remove_outliers=remove_outliers,
        zscore_cutoff=3.0, considered_lambdas=considered_lambdas
    )
    
    assert np.allclose(bl_test['o.h'], bl_manual['o.h'])
    assert np.allclose(e_test['o.h'], e_manual['o.h'])
    

    # Quantum alchemy
    calc_type = 'alchemy'
    use_ts = False
    bl_manual = {
        'n.h': np.array([0.9671514714434614]),
        'ne.h': np.array([0.9658039189158778]),
        'f.h': np.array([0.9671063585985051])
    }
    e_manual = {
        'n.h': np.array([-75.69243879972716]),
        'ne.h': np.array([-75.71124458030991]),
        'f.h': np.array([-75.7102058696739])
    }

    bl_test, e_test = dimer_eq(
        df_qc_dimer, system_label, system_charge, calc_type=calc_type,use_ts=use_ts,
        df_qats=df_qats_dimer,
        basis_set=basis_set,
        n_points=n_points, poly_order=poly_order, remove_outliers=remove_outliers,
        zscore_cutoff=3.0, considered_lambdas=considered_lambdas
    )
    for key in e_test.keys():
        assert np.allclose(bl_test[key], bl_manual[key])
        assert np.allclose(e_test[key], e_manual[key])
    


    # QATS
    calc_type = 'alchemy'
    use_ts = True
    bl_manual = {
        'n.h': np.array([1.0378062276064202, 0.9527013676653991, 0.9579026939415882, 1.4, 1.2381960149852107]),
        'ne.h': np.array([1.9, 0.7880503412001761, 0.9328696980518535, 1.151029440166537, 1.1]),
        'f.h': np.array([0.9983097349392865, 0.9080132037980053, 0.979168197421229, 0.9716710513713154, 0.9653585036411326])
    }
    e_manual = {
        'n.h': np.array([-55.1946280081958, -73.93537391224571, -75.67180585291665, -75.63206481933594, -21016.576668143272]),
        'ne.h': np.array([-127.83811520113233, -68.32334222455614, -75.79115628682297, -75.78160775656083, -75.71533779593301]),
        'f.h': np.array([-99.83306112131531, -73.8796453840478, -75.72538825462752, -75.71274196270942, -75.70977712509452])
    }

    bl_test, e_test = dimer_eq(
        df_qc_dimer, system_label, system_charge, calc_type=calc_type, use_ts=use_ts,
        df_qats=df_qats_dimer,
        basis_set=basis_set,
        n_points=n_points, poly_order=poly_order, remove_outliers=remove_outliers,
        zscore_cutoff=3.0, considered_lambdas=considered_lambdas
    )
    for key in ['n.h', 'ne.h', 'f.h']:
        assert np.allclose(bl_test[key], bl_manual[key])
        assert np.allclose(e_test[key], e_manual[key])

def test_ch_bond_lengths_alchemy():
    system_label = 'c.h'
    charge = 0  # Initial charge of the system.
    excitation_level = 0

    basis_set = 'cc-pV5Z'
    specific_atom = 0
    n_points = 2
    poly_order = 4



    use_ts = False
    qats_order = 2

    # Reference QC data
    df_qc_system = df_qc_dimer.query(
        'system == @system_label'
        '& charge == @charge'
        '& lambda_value == 0'
    )
    sys_multiplicity = get_multiplicity(df_qc_system, excitation_level)
    df_qc_system = df_qc_system.query('multiplicity == @sys_multiplicity')
    target_n_electrons = df_qc_system.iloc[0]['n_electrons']
    target_atomic_numbers = df_qc_system.iloc[0]['atomic_numbers']


    qc_system_bl, qc_system_e = dimer_curve(df_qc_system, lambda_value=0)

    qc_eq_bl, qc_eq_e = dimer_minimum(
        qc_system_bl,qc_system_e, n_points=n_points, poly_order=poly_order,
    )

    # QATS predictions with or without Taylor series.
    if use_ts:
        df_selection = 'qats'
    else:
        df_selection = 'qc'
    df_references = get_qa_refs(
        df_qc_dimer, df_qats_dimer, system_label, target_n_electrons, basis_set=basis_set, df_selection=df_selection,
        excitation_level=excitation_level, specific_atom=specific_atom
    )

    ref_systems = tuple(set(df_references['system']))

    pred_system_bl = np.zeros((len(ref_systems), len(qc_system_bl)))
    pred_system_e = np.zeros((len(ref_systems), len(qc_system_bl)))
    pred_lambda_values = np.zeros(len(ref_systems))
    pred_eq_bond_lengths = np.zeros(len(ref_systems))
    pred_eq_energies = np.zeros(len(ref_systems))

    for i in range(len(ref_systems)):
        sys_label = ref_systems[i]
        df_ref_sys = df_references.query('system == @sys_label')
        ref_atomic_numbers = df_ref_sys.iloc[0]['atomic_numbers']
        ref_lambda_value = get_lambda_value(
            ref_atomic_numbers, target_atomic_numbers, specific_atom=specific_atom
        )
        pred_lambda_values[i] = round(ref_lambda_value)
        if use_ts:
            pred_system_bl[i], pred_system_e[i] = dimer_curve(
                df_ref_sys, lambda_value=ref_lambda_value, use_ts=use_ts, qats_order=qats_order
            )
        else:
            pred_system_bl[i], pred_system_e[i] = dimer_curve(
                df_ref_sys, lambda_value=ref_lambda_value, use_ts=use_ts
            )
        pred_eq_bond_lengths[i], pred_eq_energies[i] = dimer_minimum(
            pred_system_bl[i], pred_system_e[i], n_points=n_points, poly_order=poly_order,
        )

    manual_eq_bond_lengths = {
        'n.h': 1.1161870559476388,
        'o.h': 1.1159274391116252,
        'b.h': 1.1096923823911184
    }

    assert np.allclose(qc_eq_bl, 1.1141425854358427)
    for i in range(len(ref_systems)):
        assert np.allclose(
            pred_eq_bond_lengths[i], manual_eq_bond_lengths[ref_systems[i]]
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

def test_n_ee_qats_correctness():
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
    use_ts = False
    ee_qats = get_qa_excitation(
        df_qc_atom, df_qats_atom, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set,
        use_ts=use_ts
    )

    ee_qats_keys = [i for i in ee_qats.keys()]
    ee_qats_keys.sort()
    assert ee_qats_keys == ['b', 'c', 'f', 'o']
    for key in ['b', 'c', 'f', 'o']:
        assert np.array_equal(
            ee_qats[key], np.array([ee_manual[key]], dtype='float64')
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
    ea_qats_manual = {
        'b': np.array(
            [qats_prediction(poly_coef_excited_bref, i, 2)[0] - qats_prediction(poly_coef_ground_bref, i, 2)[0] for i in range(0, 4+1)]
        ),
        'c': np.array(
            [qats_prediction(poly_coef_excited_cref, i, 1)[0] - qats_prediction(poly_coef_ground_cref, i, 1)[0] for i in range(0, 4+1)]
        ),
        'o': np.array(
            [qats_prediction(poly_coef_excited_oref, i, -1)[0] - qats_prediction(poly_coef_ground_oref, i, -1)[0] for i in range(0, 4+1)]
        ),
        'f': np.array(
            [qats_prediction(poly_coef_excited_fref, i, -2)[0] - qats_prediction(poly_coef_ground_fref, i, -2)[0] for i in range(0, 4+1)]
        ),
    }

    use_ts = True
    ee_qats = get_qa_excitation(
        df_qc_atom, df_qats_atom, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set,
        use_ts=use_ts
    )
    ee_qats_keys = [i for i in ee_qats.keys()]
    ee_qats_keys.sort()
    assert ee_qats_keys == ['b', 'c', 'f', 'o']
    for key in ['b', 'c', 'f', 'o']:
        assert np.array_equal(
            ee_qats[key], ea_qats_manual[key]
        )

test_oh_from_ne_from_qc()