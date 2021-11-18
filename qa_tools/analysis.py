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

from qa_tools.utils import *
from qa_tools.prediction import *

def get_alchemical_errors(
    df_qc, n_electrons, excitation_level=0,
    basis_set='aug-cc-pVQZ', bond_length=None, return_energies=False,
    energy_type='total'):
    """
    
    Parameters
    ----------
    bond_length : :obj:`float`, optional

    return_energies : :obj:`bool`, optional
        Return QATS energies instead of errors.
    energy_type : :obj:`str`, optional
        Species the energy type/contributions to examine. Can be ``'total'``
        energies, ``'hf'`` for Hartree-Fock contributions, or ``'correlation'``
        energies.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        System and state labels (e.g., `'c.chrg0.mult1'`) in the order of
        increasing atomic number.
    :obj:`numpy.ndarray`
        Alchemical energy errors due to modeling a target system by changing
        the nuclear charge of a reference system (e.g., c -> n). The rows and
        columns are in the same order as the state labels.
    """
    if energy_type == 'total':
        df_energy_type = 'electronic_energy'
    elif energy_type == 'hf':
        df_energy_type = 'hf_energy'
    elif energy_type == 'correlation':
        df_energy_type = 'correlation_energy'

    df_alch_pes = df_qc[(df_qc.n_electrons == n_electrons) & (df_qc.basis_set == basis_set)]
    sys_labels = list(set(df_alch_pes.system.values))

    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        is_dimer = True
    else:
        is_dimer = False

    # Gets data.
    sys_atomic_numbers = []
    system_labels = []
    calc_labels = []
    lambda_values = []
    energies = []
    true_energies = []
    for sys_label in sys_labels:
        df_sys = df_alch_pes.query('system == @sys_label')
        if is_dimer:
            assert bond_length is not None
            df_sys = df_sys.query('bond_length == @bond_length')

        # Select multiplicity
        df_state = select_state(
            df_sys[df_sys.lambda_value == 0.0], excitation_level,
            ignore_one_row=True
        )
        
        atomic_numbers = df_state.iloc[0]['atomic_numbers']
        if is_dimer:
            sys_atomic_numbers.append(atomic_numbers)
        else:
            sys_atomic_numbers.append(atomic_numbers[0])
        state_mult = df_state.iloc[0]['multiplicity']
        state_chrg = df_state.iloc[0]['charge']
        true_energies.append(df_state.iloc[0][df_energy_type])
        system_labels.append(sys_label)
        calc_labels.append(f'{sys_label}.chrg{state_chrg}.mult{state_mult}')

        df_sys = df_sys.query('multiplicity == @state_mult')

        lambda_values.append(df_sys.lambda_value.values)
        energies.append(df_sys[df_energy_type].values)
    sys_atomic_numbers = np.array(sys_atomic_numbers)
    lambda_values = np.array(lambda_values)
    true_energies = np.array(true_energies)
    energies = np.array(energies)

    # Prepares stuff to organize data
    ## Lambdas
    sys_min_lambda_values = lambda_values.min(axis=1).astype('int')
    global_min_lambda_value = np.min(sys_min_lambda_values)
    adjust_lambdas = global_min_lambda_value - sys_min_lambda_values
    for i in range(len(adjust_lambdas)):
        lambda_values[i] += adjust_lambdas[i]

    if is_dimer:
        sys_atomic_numbers_sum = np.sum(sys_atomic_numbers, axis=1)
        if np.all(sys_atomic_numbers_sum==sys_atomic_numbers_sum.flatten()[0]):
            sort_z = np.argsort(np.min(sys_atomic_numbers, axis=1))
        else:
            sort_z = np.argsort(sys_atomic_numbers_sum)
    else:
        sort_z = np.argsort(sys_atomic_numbers)
    
    sys_energies = []
    for target_idx,true_e_idx in zip(np.flip(sort_z), sort_z):  # Smallest to largest lambda value
        target_lambda = sys_min_lambda_values[target_idx]
        true_energy = true_energies[true_e_idx]
        errors = []
        for ref_idx in sort_z:
            lambda_idx = np.where(lambda_values[ref_idx] == target_lambda)[0]
            if return_energies:
                errors.append(energies[ref_idx][lambda_idx][0])
            else:
                errors.append(energies[ref_idx][lambda_idx][0] - true_energy)
        sys_energies.append(errors)
    sys_energies = np.array(sys_energies)  # Hartree

    return [calc_labels[i] for i in sort_z], sys_energies

def get_qats_errors(
    df_qc, df_qats, n_electrons, qats_order=2, excitation_level=0,
    basis_set='aug-cc-pV5Z', return_energies=False,
    specific_atom=None, direction=None):
    """

    Only atoms are supported.
    
    Parameters
    ----------
    qats_order : :obj:`int`, optional
        Desired order of QATS to use. Defaults to ``2``.
    return_energies : :obj:`bool`, optional
        Return QATS energies instead of errors.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        System and state labels (e.g., `'c.chrg0.mult1'`) in the order of
        increasing atomic number.
    :obj:`numpy.ndarray`
        Alchemical energy errors due to modeling a target system by changing
        the nuclear charge of a reference system (e.g., c -> n). The rows and
        columns are in the same order as the state labels.
    """

    df_alch_pes = df_qc.query(
        'n_electrons == @n_electrons & basis_set == @basis_set'
    )
    mult_sys_test = df_alch_pes.iloc[0]['system']
    state_mult = get_multiplicity(
        df_alch_pes.query('system == @mult_sys_test'), excitation_level,
        ignore_one_row=False
    )
    df_alch_pes = df_alch_pes.query('multiplicity == @state_mult')

    df_sys_info = df_alch_pes.query('lambda_value == 0.0')
    charge_sort = np.argsort(df_sys_info['charge'].values)  # most negative to most positive
    sys_labels = df_sys_info['system'].values[charge_sort]
    sys_atomic_numbers = df_sys_info['atomic_numbers'].values[charge_sort]
    sys_charges = df_sys_info['charge'].values[charge_sort]

    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        raise ValueError('Dimers are not supported.')

    # Gets data.
    calc_labels = []
    lambda_values = []
    alchemical_energies = []
    qats_energies = []

    # Goes through all possible reference systems and calculates QATS-n predictions
    # then computes the alchemical predictions and errors.
    # Loops through all systems.
    for i in range(len(sys_labels)):
        sys_alchemical_energies = []
        sys_qats_energies = []

        target_label = sys_labels[i]
        target_atomic_numbers = sys_atomic_numbers[i]
        target_charge = sys_charges[i]
        calc_labels.append(f'{target_label}.chrg{target_charge}.mult{state_mult}')

        df_qats_ref = get_qa_refs(
            df_qc, df_qats, target_label, n_electrons, basis_set=basis_set,
            df_selection='qats', excitation_level=excitation_level, specific_atom=specific_atom,
            direction=direction, considered_lambdas=None
        )
        
        charge_sort = np.argsort(df_qats_ref['charge'].values)  # most negative to most positive

        # Loops through all QATS references.
        for j in charge_sort:
            qats_row = df_qats_ref.iloc[j]
            ref_sys_label = qats_row['system']
            ref_atomic_numbers = qats_row['atomic_numbers']
            ref_charge = qats_row['charge']
            ref_poly_coeffs = qats_row['poly_coeffs']

            lambda_value = get_lambda_value(
                ref_atomic_numbers, target_atomic_numbers, specific_atom=specific_atom,
                direction=direction
            )

            # Predicted alchemical energy.
            sys_alchemical_energies.append(
                qa_predictions(
                    df_qc, ref_sys_label, ref_charge, excitation_level=excitation_level,
                    lambda_values=[lambda_value], basis_set=basis_set,
                    ignore_one_row=True
                )[0]
            )

            # QATS prediction
            sys_qats_energies.append(
                qats_prediction(
                    ref_poly_coeffs, qats_order, lambda_value
                )[0]
            )
        
        # Adds in alchemical energy and QATS reference
        sys_alchemical_energies.insert(i, np.nan)
        sys_qats_energies.insert(i, np.nan)

        alchemical_energies.append(sys_alchemical_energies)
        qats_energies.append(sys_qats_energies)
    alchemical_energies = np.array(alchemical_energies)
    qats_energies = np.array(qats_energies)

    e_return = qats_energies
    if not return_energies:
        e_return -= alchemical_energies
    
    # Converts nan to 0
    e_return = np.nan_to_num(e_return)

    return calc_labels, e_return

def get_qc_binding_curve(
    df_qc, sys_label, charge, excitation_level=0, basis_set='aug-cc-pVQZ',
    lambda_value=0
):
    """
    
    Returns
    -------
    
    """
    df_sys = df_qc.query(
        'system == @sys_label'
        '& basis_set == @basis_set'
        '& charge == @charge'
    )
    df_state = select_state(
        df_sys[df_sys.lambda_value == 0.0], excitation_level,
        ignore_one_row=True
    )
    state_mult = df_state.iloc[0]['multiplicity']
    df_sys = df_sys.query(
        'multiplicity == @state_mult'
        '& lambda_value == @lambda_value'
    )
    bond_length_idx = np.argsort(df_sys.bond_length.values)
    bond_lengths = []
    energies = []
    for idx in bond_length_idx:
        bond_lengths.append(df_sys.iloc[idx].bond_length)
        energies.append(df_sys.iloc[idx].electronic_energy)
    
    return np.array(bond_lengths), np.array(energies)

def qats_error_change_charge(
    df_qc, df_qats, target_label, delta_charge, change_signs=False,
    basis_set='aug-cc-pVQZ', target_initial_charge=0, use_ts=True,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, compute_difference=False
):
    """Computes QATS errors in change the charge of a system.
    """
    qc_prediction = hartree_to_ev(
        energy_change_charge_qc_atom(
            df_qc, target_label, delta_charge,
            target_initial_charge=target_initial_charge,
            change_signs=change_signs, basis_set=basis_set
        )
    )
    qats_predictions = energy_change_charge_qa_atom(
        df_qc, df_qats, target_label, delta_charge,
        target_initial_charge=target_initial_charge,
        change_signs=change_signs, basis_set=basis_set,
        use_ts=use_ts, ignore_one_row=ignore_one_row, 
        considered_lambdas=considered_lambdas,
        compute_difference=compute_difference
    )
    
    qats_predictions = {
        key:hartree_to_ev(value) for (key,value) in qats_predictions.items()
    }  # Converts to eV
    if use_ts or compute_difference:
        qats_predictions = pd.DataFrame(
            qats_predictions,
            index=[f'QATS-{i}' for i in range(max_qats_order+1)]
        )
    else:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=['QATS']
        )
    if compute_difference:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors

def qats_error_change_charge_dimer(
    df_qc, df_qats, target_label, delta_charge, change_signs=False,
    basis_set='cc-pV5Z', target_initial_charge=0, use_ts=True,
    lambda_specific_atom=None, lambda_direction=None,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, compute_difference=False,
    n_points=2, poly_order=4, remove_outliers=False,
    zscore_cutoff=3.0):
    """Computes QATS errors in change the charge of a system.
    """
    qc_prediction = hartree_to_ev(
        get_qc_change_charge_dimer(
            df_qc, target_label, delta_charge,
            target_initial_charge=target_initial_charge,
            change_signs=change_signs, basis_set=basis_set,
            ignore_one_row=ignore_one_row, n_points=n_points,
            poly_order=poly_order, remove_outliers=remove_outliers,
            zscore_cutoff=zscore_cutoff
        )
    )
    qats_predictions = get_qa_change_charge_dimer(
        df_qc, df_qats, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts,
        lambda_specific_atom=lambda_specific_atom, lambda_direction=lambda_direction,
        ignore_one_row=ignore_one_row, poly_order=poly_order, n_points=n_points,
        remove_outliers=remove_outliers, considered_lambdas=considered_lambdas,
        compute_difference=compute_difference
    )
    
    qats_predictions = {
        key:hartree_to_ev(value) for (key,value) in qats_predictions.items()
    }  # Converts to eV
    if use_ts or compute_difference:
        qats_predictions = pd.DataFrame(
            qats_predictions,
            index=[f'QATS-{i}' for i in range(max_qats_order+1)]
        )
    else:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=['QATS']
        )
    if compute_difference:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors

def qats_error_excitation_energy(
    df_qc, df_qats, target_label, target_charge=0, excitation_level=1,
    basis_set='aug-cc-pVQZ', use_ts=True,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, compute_difference=False
):
    """Computes QATS errors in system excitation energies.

    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    qc_prediction = hartree_to_ev(
        get_qc_excitation(
            df_qc, target_label, target_charge=target_charge,
            excitation_level=excitation_level, basis_set=basis_set,
            ignore_one_row=ignore_one_row
        )
    )
    qats_predictions = get_qa_excitation(
        df_qc, df_qats, target_label, target_charge=target_charge,
        excitation_level=excitation_level, basis_set=basis_set,
        use_ts=use_ts, ignore_one_row=ignore_one_row,
        considered_lambdas=considered_lambdas,
        compute_difference=compute_difference
    )
    
    qats_predictions = {key:hartree_to_ev(value) for (key,value) in qats_predictions.items()}  # Converts to eV
    if use_ts:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=[f'QATS-{i}' for i in range(max_qats_order+1)]
        )  # Makes dataframe
    else:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=['QATS']
        )  # Makes dataframe

    if compute_difference:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors


