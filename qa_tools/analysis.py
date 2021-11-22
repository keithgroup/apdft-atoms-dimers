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

def qa_pes_errors(
    df_qc, n_electrons, excitation_level=0, basis_set='aug-cc-pV5Z',
    bond_length=None, return_energies=False, energy_type='total'):
    """Computes the error associated with predicting a system's absolute
    electronic energy using quantum alchemy.

    In other words, this quantifies the error when using a quantum alchemy
    reference and nuclear charge perturbation to model a target. For example,
    how accurate is using C- basis set with a lambda of 1 to predict N.
    
    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    n_electrons : :obj:`int`
        Total number of electrons for the quantum alchemical PES.
    excitation_level : :obj:`int`, optional
        Electronic state of the system with respect to the ground state. ``0``
        represents the ground state, ``1`` the first excited state, etc.
        Defaults to ground state.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pV5Z'``.
    bond_length : :obj:`float`, optional
        Desired bond length for dimers; must be specified.
    return_energies : :obj:`bool`, optional
        Return quantum alchemy energies instead of errors. Defaults to ``False``.
    energy_type : :obj:`str`, optional
        Species the energy type/contributions to examine. Can be ``'total'``
        energies, ``'hf'`` for Hartree-Fock contributions, or ``'correlation'``
        energies. Defaults to ``'total'``.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        System and state labels (e.g., `'c.chrg0.mult1'`) in the order of
        increasing atomic number (and charge).
    :obj:`numpy.ndarray`
        Quantum alchemy errors (or energies) with respect to standard quantum
        chemistry.
    """
    if energy_type == 'total':
        df_energy_type = 'electronic_energy'
    elif energy_type == 'hf':
        df_energy_type = 'hf_energy'
    elif energy_type == 'correlation':
        df_energy_type = 'correlation_energy'

    df_qa_pes = df_qc.query(
        'n_electrons == @n_electrons'
        '& basis_set == @basis_set'
    )
    sys_labels = list(set(df_qa_pes['system'].values))

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
        df_sys = df_qa_pes.query('system == @sys_label')
        if is_dimer:
            assert bond_length is not None
            df_sys = df_sys.query('bond_length == @bond_length')

        # Select multiplicity
        df_state = select_state(
            df_sys.query('lambda_value == 0.0'), excitation_level,
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

def qats_pes_errors(
    df_qc, df_qats, n_electrons, qats_order=2, excitation_level=0,
    basis_set='aug-cc-pV5Z', return_energies=False):
    """Computes the error associated with using a Taylor series to approximate
    the quantum alchemical potential energy surface.

    Errors are in reference to quantum alchemy. Only atom dataframes are 
    supported.
    
    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    df_qats : :obj:`pandas.DataFrame`, optional
        QATS dataframe.
    n_electrons : :obj:`int`
        Total number of electrons for the quantum alchemical PES.
    qats_order : :obj:`int`, optional
        Desired Taylor series order to use. Defaults to ``2``.
    excitation_level : :obj:`int`, optional
        Electronic state of the system with respect to the ground state. ``0``
        represents the ground state, ``1`` the first excited state, etc.
        Defaults to ground state.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pV5Z'``.
    return_energies : :obj:`bool`, optional
        Return QATS energies instead of errors. Defaults to ``False``.
    
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
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        raise ValueError('Dimers are not supported.')

    df_qa_pes = df_qc.query(
        'n_electrons == @n_electrons & basis_set == @basis_set'
    )
    mult_sys_test = df_qa_pes.iloc[0]['system']
    state_mult = get_multiplicity(
        df_qa_pes.query('system == @mult_sys_test'), excitation_level,
        ignore_one_row=False
    )
    df_qa_pes = df_qa_pes.query('multiplicity == @state_mult')

    df_sys_info = df_qa_pes.query('lambda_value == 0.0')
    charge_sort = np.argsort(df_sys_info['charge'].values)  # most negative to most positive
    sys_labels = df_sys_info['system'].values[charge_sort]
    sys_atomic_numbers = df_sys_info['atomic_numbers'].values[charge_sort]
    sys_charges = df_sys_info['charge'].values[charge_sort]


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
            df_selection='qats', excitation_level=excitation_level,
             considered_lambdas=None
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
                ref_atomic_numbers, target_atomic_numbers
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

def error_change_charge_qats_atoms(
    df_qc, df_qats, target_label, delta_charge, change_signs=False,
    basis_set='aug-cc-pV5Z', target_initial_charge=0, use_ts=True,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, return_qats_vs_qa=False):
    """Automates the procedure of calculating errors for changing charges on
    atoms.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    df_qats : :obj:`pandas.DataFrame`, optional
        QATS dataframe.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    delta_charge : :obj:`str`
        Overall change in the initial target system.
    change_signs : :obj:`bool`, optional
        Multiply all predictions by -1. Used to correct the sign for computing
        electron affinities. Defaults to ``False``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pV5Z'``.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.
    use_ts : :obj:`bool`, optional
        Use a Taylor series approximation (with finite differences) to make
        QATS-n predictions (where n is the order). Defaults to ``True``.
    max_qats_order : :obj:`int`, optional
        Maximum order to use for the Taylor series. Defaults to ``4``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
        Defaults to ``None``.
    return_qats_vs_qa : :obj:`bool`, optional
        Return the difference of QATS-n - QA predictions; i.e., the error of
        using a Taylor series with repsect to quantum alchemy.
        Defaults to ``False``.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        raise ValueError('Dimers are not supported.')
    
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
        return_qats_vs_qa=return_qats_vs_qa
    )
    
    qats_predictions = {
        key:hartree_to_ev(value) for (key,value) in qats_predictions.items()
    }  # Converts to eV
    if use_ts or return_qats_vs_qa:
        qats_predictions = pd.DataFrame(
            qats_predictions,
            index=[f'QATS-{i}' for i in range(max_qats_order+1)]
        )
    else:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=['QATS']
        )
    if return_qats_vs_qa:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors

def error_change_charge_qats_dimer(
    df_qc, df_qats, target_label, delta_charge, change_signs=False,
    basis_set='cc-pV5Z', target_initial_charge=0, use_ts=True,
    lambda_specific_atom=0, lambda_direction=None,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, return_qats_vs_qa=False,
    n_points=2, poly_order=4, remove_outliers=False,
    zscore_cutoff=3.0):
    """Computes QATS errors in change the charge of a system.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    df_qats : :obj:`pandas.DataFrame`, optional
        QATS dataframe.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    delta_charge : :obj:`str`
        Overall change in the initial target system.
    change_signs : :obj:`bool`, optional
        Multiply all predictions by -1. Used to correct the sign for computing
        electron affinities. Defaults to ``False``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pV5Z'``.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.
    use_ts : :obj:`bool`, optional
        Use a Taylor series approximation (with finite differences) to make
        QATS-n predictions (where n is the order). Defaults to ``True``.
    lambda_specific_atom : :obj:`int`, optional
        Applies the entire lambda change to a single atom in dimers. For
        example, OH -> FH+ would be a lambda change of +1 only on the first
        atom. Defaults to ``0``.
    lambda_direction : :obj:`str`, optional
        Defines the direction of lambda changes for dimers. ``'counter'`` is
        is where one atom increases and the other decreases their nuclear
        charge (e.g., CO -> BF).
        If the atomic numbers of the reference are the same, the first atom's
        nuclear charge is decreased and the second is increased. IF they are
        different, the atom with the largest atomic number increases by lambda.
        Defaults to ``None``.
    max_qats_order : :obj:`int`, optional
        Maximum order to use for the Taylor series. Defaults to ``4``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
        Defaults to ``None``.
    return_qats_vs_qa : :obj:`bool`, optional
        Return the difference of QATS-n - QATS predictions; i.e., the error of
        using a Taylor series approximation with repsect to the alchemical
        potential energy surface. Defaults to ``False``.
    n_points : :obj:`int`, optional
        The number of surrounding points on either side of the minimum bond
        length. Defaults to ``2``.
    poly_order : :obj:`int`, optional
        Maximum order of the fitted polynomial. Defaults to ``2``.
    remove_outliers : :obj:`bool`, optional
        Do not include bond lengths that are marked as outliers by their z
        score. Defaults to ``False``.
    zscore_cutoff : :obj:`float`, optional
        Bond length energies that have a z score higher than this are
        considered outliers. Defaults to ``3.0``.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    qc_prediction = hartree_to_ev(
        energy_change_charge_qc_dimer(
            df_qc, target_label, delta_charge,
            target_initial_charge=target_initial_charge,
            change_signs=change_signs, basis_set=basis_set,
            ignore_one_row=ignore_one_row, n_points=n_points,
            poly_order=poly_order, remove_outliers=remove_outliers,
            zscore_cutoff=zscore_cutoff
        )
    )
    qats_predictions = energy_change_charge_qa_dimer(
        df_qc, df_qats, target_label, delta_charge,
        target_initial_charge=target_initial_charge, change_signs=change_signs,
        basis_set=basis_set, use_ts=use_ts,
        lambda_specific_atom=lambda_specific_atom, lambda_direction=lambda_direction,
        ignore_one_row=ignore_one_row, poly_order=poly_order, n_points=n_points,
        remove_outliers=remove_outliers, considered_lambdas=considered_lambdas,
        return_qats_vs_qa=return_qats_vs_qa
    )
    
    qats_predictions = {
        key:hartree_to_ev(value) for (key,value) in qats_predictions.items()
    }  # Converts to eV
    if use_ts or return_qats_vs_qa:
        qats_predictions = pd.DataFrame(
            qats_predictions,
            index=[f'QATS-{i}' for i in range(max_qats_order+1)]
        )
    else:
        qats_predictions = pd.DataFrame(
            qats_predictions, index=['QATS']
        )
    if return_qats_vs_qa:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors

def error_mult_gap_qa_atom(
    df_qc, df_qats, target_label, target_charge=0,
    basis_set='aug-cc-pV5Z', use_ts=True,
    max_qats_order=4, ignore_one_row=False,
    considered_lambdas=None, return_qats_vs_qa=False):
    """Computes QATS errors in system multiplicity gaps.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    df_qats : :obj:`pandas.DataFrame`, optional
        QATS dataframe.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    target_charge : :obj:`int`, optional
        The system charge. Defaults to ``0``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pV5Z'``.
    use_ts : :obj:`bool`, optional
        Use a Taylor series approximation to make QATS-n predictions
        (where n is the order). Defaults to ``True``.
    max_qats_order : :obj:`int`, optional
        Maximum order to use for the Taylor series. Defaults to ``4``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``False``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    return_qats_vs_qa : :obj:`bool`, optional
        Return the difference of QATS-n - QATS predictions; i.e., the error of
        using a Taylor series approximation with repsect to the alchemical
        potential energy surface. Defaults to ``False``.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
    """
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        raise ValueError('Dimers are not supported.')
    
    qc_prediction = hartree_to_ev(
        mult_gap_qc_atom(
            df_qc, target_label, target_charge=target_charge,
            basis_set=basis_set, ignore_one_row=ignore_one_row
        )
    )
    qats_predictions = mult_gap_qa_atom(
        df_qc, df_qats, target_label, target_charge=target_charge,
        basis_set=basis_set, use_ts=use_ts, ignore_one_row=ignore_one_row,
        considered_lambdas=considered_lambdas,
        return_qats_vs_qa=return_qats_vs_qa
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

    if return_qats_vs_qa:
        return qats_predictions
    else:
        qats_errors = qats_predictions.transform(lambda x: x - qc_prediction)
        return qats_errors


