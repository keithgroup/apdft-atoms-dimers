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
import pandas as pd

from qa_tools.utils import *

def get_qc_pred(
    df_qc, system_label, charge, excitation_level=0, lambda_values=[-1, 0, 1],
    basis_set='aug-cc-pVQZ', ignore_one_row=True):
    """Calculate QC prediction at specified lambdas

    Parameters
    ----------
    df_qc : :obj:`pandas.dataframe`
        A dataframe with quantum chemistry data.
    system_label : :obj:`str`
        The system label of the desired APDFT prediction target. For example,
        `'c'`, `'h'`, etc.
    charge : :obj:`int`
        Total system charge.
    excitation_level : :obj:`int`, optional
        Electronic state of the system with respect to the ground state. ``0``
        represents the ground state, ``1`` the first excited state, etc.
        Defaults to ground state.
    lambda_values : :obj:`float`, :obj:`list`, optional
        Lambda values to make QC predictions at. Defaults to ``[-1, 0, 1]``.
    basis_set : :obj:`str`, optional
        Desired basis sets the predictions are from. Defaults to
        ``aug-cc-pVQZ``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Energies of specified lambda values predicted using the quantum chemical
        method with the same shape.
    """
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        is_dimer = True
    else:
        is_dimer = False
    if not isinstance(lambda_values, np.ndarray) and not isinstance(lambda_values, list):
        lambda_values = np.array([lambda_values])
    
    ref_qc = df_qc.query(
        'system == @system_label'
        '& charge == @charge'
        '& basis_set == @basis_set'
        '& lambda_value in @lambda_values'
    )
    
    # Selects state if required.
    if len(set(df_qc['multiplicity'].values)) > 1:
        assert excitation_level is not None
        sys_mult = get_multiplicity(
            ref_qc, excitation_level, ignore_one_row=ignore_one_row
        )
        ref_qc = ref_qc.query('multiplicity == @sys_mult')

    energies = np.zeros(len(lambda_values))
    for i in range(len(lambda_values)):
        lambda_value = lambda_values[i]
        energies[i] = ref_qc.query(
            'lambda_value == @lambda_value'
        )['electronic_energy'].values[0]

    return energies

def calc_qats_pred(poly_coeffs, order, lambda_values):
    """APDFTn predictions using a Taylor series.

    Parameters
    ----------
    poly_coeffs : :obj:`list`
        Polynomial coefficients in decreasing order (e.g., zeroth, first,
        second, etc.).
    order :obj:`int`
        Highest APDFT order to calculate.
    lambda_values : :obj:`float`, :obj:`list`
        Lambda values to make QC predictions at.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        APDFTn predictions using a nth order Taylor series.
    """
    if not isinstance(lambda_values, np.ndarray) \
    or not isinstance(lambda_values, list):
        lambda_values = np.array([lambda_values])
    return np.polyval(poly_coeffs[:order+1][::-1], lambda_values)

def get_qc_change_charge(
    df_qc, target_label, delta_charge, target_initial_charge=0, bond_length=None,
    change_signs=False, basis_set='aug-cc-pVQZ', force_same_method=False,
    ignore_one_row=True):
    """Calculate electron affinity using quantum chemistry.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A pandas dataframe with quantum chemistry data. It should have the
        following columns (from `get_qc_dframe`): system, atomic_numbers,
        charge, multiplicity, n_electrons, qc_method, basis_set, lambda_range,
        finite_diff_delta, finite_diff_acc, poly_coeff.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'c'``, ``'si'``, or ``'f.h'``.
    delta_charge : :obj:`int`
        Overall change in the initial target system.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.

    Returns
    -------
    :obj:`float`
        Quantum chemistry predicted energy change due to changing the charge
        of the system.
    """
    # Checks that a bond_length is provided for dimers.
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        assert bond_length is not None
    
    # Selects initial target ground state QC data.
    target_initial_qc = df_qc[
        (df_qc.system == target_label) & (df_qc.charge == target_initial_charge)
        & (df_qc.lambda_value == 0.0) & (df_qc.basis_set == basis_set)
    ]
    
    if bond_length is not None:
        target_initial_qc = target_initial_qc.query(
            'bond_length == @bond_length'
        )
    
    if len(target_initial_qc) == 0:
        # Often we do not have the data to make the prediction.
        # So we return NaN.
        return np.nan
    else:
        target_initial_qc = select_state(
            target_initial_qc, 0, ignore_one_row=ignore_one_row
        )
        assert len(target_initial_qc) == 1  # Should only have one row.
    target_initial_n_electrons = target_initial_qc.n_electrons.values[0]

    # Selects final target ground state QC data.
    target_final_n_electrons = target_initial_n_electrons - delta_charge
    
    target_final_qc = df_qc[
        (df_qc.system == target_label)
        & (df_qc.lambda_value == 0.0)
        & (df_qc.n_electrons == target_final_n_electrons)
        & (df_qc.basis_set == basis_set)
    ]
    if bond_length is not None:
        target_final_qc = target_final_qc.query(
            'bond_length == @bond_length'
        )
    if len(target_final_qc) == 0:
        # Often we do not have the data to make the prediction.
        # So we return NaN.
        return np.nan
    else:
        target_final_qc = select_state(
            target_final_qc, 0, ignore_one_row=ignore_one_row
        )
        assert len(target_final_qc) == 1  # Should only have one row.
    
    # Checks that methods are the same (HF with HF, CCSD with CCSD, etc.)
    if force_same_method:
        energies = unify_qc_energies(target_initial_qc, target_final_qc)
    else:
        energies = [
            target_initial_qc.iloc[0]['electronic_energy'],
            target_final_qc.iloc[0]['electronic_energy']
        ]

    e_diff = energies[1] - energies[0]
    if change_signs:
        e_diff *= -1
    return e_diff

def get_qats_change_charge(
    df_qc, df_qats, target_label, delta_charge, bond_length=None,
    target_initial_charge=0, change_signs=False, basis_set='aug-cc-pVQZ',
    use_qats=True, lambda_specific_atom=None, lambda_direction=None,
    ignore_one_row=True, considered_lambdas=None, compute_difference=False):
    """Use an APDFT reference to predict the energy change due to adding or
    removing an electron.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A pandas dataframe with quantum chemistry data. It should have the
        following columns (from `get_qc_dframe`): system, atomic_numbers,
        charge, multiplicity, n_electrons, qc_method, basis_set, lambda_range,
        finite_diff_delta, finite_diff_acc, poly_coeff.
    df_qats : :obj:`pandas.DataFrame`
        A pandas dataframe with APDFT data. It should have the
        following columns (from `get_qats_dframe`): system, atomic_numbers,
        charge, multiplicity, n_electrons, qc_method, basis_set, lambda,
        electronic_energy, hf_energy, and correlation_energy.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'c'``, ``'si'``, or ``'f.h'``.
    delta_charge : :obj:`str`
        Overall change in the initial target system.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.
    change_signs : :obj:`bool`, optional
        Multiply all predictions by -1. Used to correct the sign for computing
        electron affinities. Defaults to ``False``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'aug-cc-pVQZ'``.
    use_qats : :obj:`bool`, optional
        Use a Taylor series approximation (with finite differences) to make
        APDFTn predictions (where n is the order). Defaults to ``True``.
    lambda_specific_atom : :obj:`int`, optional
        Applies the entire lambda change to a single atom in dimers. For
        example, OH -> FH+ would be a lambda change of +1 only on the first
        atom.
    lambda_direction : :obj:`str`, optional
        Defines the direction of lambda changes for dimers. ``'counter'`` is
        is where one atom increases and the other decreases their nuclear
        charge (e.g., CO -> BF).
        If the atomic numbers of the reference are the same, the first atom's
        nuclear charge is decreased and the second is increased. IF they are
        different, the atom with the largest atomic number increases by lambda.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    compute_difference : :obj:`bool`, optional
        Return the difference of APDFTn - APDFT predictions; i.e., the error of
        using a Taylor series approximation with repsect to the alchemical
        potential energy surface. Defaults to ``False``.
    
    Returns
    -------
    :obj:`dict`
    """
    if compute_difference: assert use_qats == True
    if len(df_qc.iloc[0]['atomic_numbers']) == 2: assert bond_length is not None
    assert delta_charge != 0
    if delta_charge < 0: assert change_signs == True
    # Selects initial target ground state QC data.
    target_initial_qc = df_qc[
        (df_qc.system == target_label) & (df_qc.charge == target_initial_charge)
        & (df_qc.lambda_value == 0.0) & (df_qc.basis_set == basis_set)
    ]
    # Checks for dimers
    if bond_length is not None:
        target_initial_qc = target_initial_qc.query(
            'bond_length == @bond_length'
        )
    target_initial_qc = select_state(
        target_initial_qc, 0, ignore_one_row=ignore_one_row
    )
    assert len(target_initial_qc) == 1  # Should only have one row.
    target_initial_n_electrons = target_initial_qc.n_electrons.values[0]
    target_atomic_numbers = target_initial_qc.iloc[0]['atomic_numbers']

    # Performs checks on lambda selections.
    if len(target_atomic_numbers) == 2:
        assert lambda_specific_atom is not None or lambda_direction is not None

    # Selects final target ground state QC data.
    target_final_n_electrons = target_initial_n_electrons - delta_charge

    # Get all available references for the initial target based on ground state
    # energies.
    avail_ref_final_sys = set(
        df_qats[
            (df_qats.system != target_label)
            & (df_qats.n_electrons == target_final_n_electrons)
            & (df_qats.basis_set == basis_set)
        ].system.values
    )
    
    ref_initial_qats = get_qa_refs(
        df_qc, df_qats, target_label, target_initial_n_electrons,
        basis_set=basis_set
    )
    if bond_length is not None:
        ref_initial_qats = ref_initial_qats.query(
            'bond_length == @bond_length'
        )
    ref_initial_qats = ref_initial_qats[
        ref_initial_qats['system'].isin(avail_ref_final_sys)
    ]
    ref_initial_qats = select_state(
        ref_initial_qats, 0, ignore_one_row=ignore_one_row
    )

    # Get all available references for the final target based on ground state
    # energies.
    ref_final_qats = get_qa_refs(
        df_qc, df_qats, target_label, target_final_n_electrons,
        basis_set=basis_set
    )
    if bond_length is not None:
        ref_final_qats = ref_final_qats.query(
            'bond_length == @bond_length'
        )
    ref_final_qats = ref_final_qats[
        ref_final_qats['system'].isin(ref_initial_qats.system)
    ]
    ref_final_qats = select_state(
        ref_final_qats, 0, ignore_one_row=ignore_one_row
    )

    # Checks that the size of initial and final dataframe is the same
    assert len(ref_initial_qats) == len(ref_final_qats)


    predictions = {}
    for system in ref_initial_qats.system:
        ref_initial = ref_initial_qats.query('system == @system')
        ref_final = ref_final_qats.query('system == @system')
        lambda_initial = get_lambda_value(
            ref_initial.iloc[0]['atomic_numbers'], target_atomic_numbers,
            specific_atom=lambda_specific_atom, direction=lambda_direction
        )

        lambda_final = get_lambda_value(
            ref_final.iloc[0]['atomic_numbers'], target_atomic_numbers,
            specific_atom=lambda_specific_atom, direction=lambda_direction
        )

        assert lambda_initial == lambda_final

        if considered_lambdas is not None:
            if lambda_initial not in considered_lambdas:
                continue

        if use_qats or compute_difference == True:
            order_preds = []
            for order in range(len(ref_initial.iloc[0]['poly_coeff'])):
                e_target_initial = calc_qats_pred(
                    ref_initial.iloc[0]['poly_coeff'], order, lambda_initial
                )
                e_target_final = calc_qats_pred(
                    ref_final.iloc[0]['poly_coeff'], order, lambda_final
                )
                e_diff = (e_target_final - e_target_initial)[0]
                
                if change_signs:
                    e_diff *= -1
                order_preds.append(e_diff)
            predictions[system] = np.array(order_preds)
        if not use_qats or compute_difference == True:
            chrg_ref_initial = ref_initial.iloc[0]['charge']
            mult_ref_initial = ref_initial.iloc[0]['multiplicity']
            ref_initial_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_initial'
                '& multiplicity == @mult_ref_initial'
                '& basis_set == @basis_set'
            )
            if bond_length is not None:
                ref_initial_qc = ref_initial_qc.query(
                    'bond_length == @bond_length'
                )
            assert len(ref_initial_qc) == 1
            e_target_initial = ref_initial_qc.iloc[0]['electronic_energy']
            
            chrg_ref_final = ref_final.iloc[0]['charge']
            mult_ref_final = ref_final.iloc[0]['multiplicity']
            ref_final_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_final'
                '& multiplicity == @mult_ref_final'
                '& basis_set == @basis_set'
            )
            
            e_target_final = ref_final_qc.iloc[0]['electronic_energy']
            e_diff = e_target_final - e_target_initial
            if change_signs:
                e_diff *= -1
            if compute_difference:
                pred_diff = [i - e_diff for i in predictions[system]]
                predictions[system] = np.array(pred_diff)
            else:
                predictions[system] = np.array([e_diff])

    return predictions

def get_dimer_minimum(bond_lengths, energies, n_points=2, poly_order=4,
    remove_outliers=False, zscore_cutoff=3.0):
    """Interpolate the minimum energy of a dimer with respect to bond length
    using a fitted polynomial to the lowest energies.

    Parameters
    ----------
    bond_lengths : :obj:`numpy.ndarray`
        All bond lengths considered.
    energies : :obj:`numpy.ndarray`
        Corresponding electronic energies.
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
    :obj:`float`
        Equilibrium bond length.
    :obj:`float`
        Electronic energy corresponding to the equilibrium bond length.
    """
    bond_lengths_fit, poly_coeffs = fit_dimer_poly(
        bond_lengths, energies, n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=zscore_cutoff
    )
    eq_bond_length, eq_energy = find_poly_min(
        bond_lengths_fit, poly_coeffs
    )
    return eq_bond_length, eq_energy

def get_dimer_curve(df, lambda_value=None, use_qats=False, qa_order=None):
    """Bond lengths and their respective electronic energies.

    There should only be one system left in the dataframe.

    Parameters
    ----------
    df : :obj:`pandas.dataframe`
        A quantum chemistry or APDFT dataframe.
    lambda_value : :obj:`float`
        Desired lambda value if more than one in the dataframe if finite
        differences are required.
    use_qats : :obj:`bool`, optional
        Make APDFTn predictions using Taylor series approximation with
        derivatives from finite differences. Defaults to ``False``.
    qa_order : :obj:`int`, optional
        Taylor series order to be used in APDFT predictions.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        All considered bond lengths available for a system.
    """
    if use_qats:
        assert 'poly_coeff' in df.columns
        assert qa_order is not None
        bond_length_order = np.argsort(df['bond_length'].values)
        bond_lengths = []
        energies = []
        for idx in bond_length_order:
            bond_lengths.append(
                df.iloc[idx]['bond_length']
            )
            poly_coeffs = df.iloc[idx]['poly_coeff']
            energies.append(
                calc_qats_pred(poly_coeffs, qa_order, lambda_value)[0]
            )
        
        return np.array(bond_lengths), np.array(energies)
    else:
        assert 'electronic_energy' in df.columns
        assert qa_order is None
        if lambda_value is not None:
            df = df.query('lambda_value == @lambda_value')
        bond_length_order = np.argsort(df['bond_length'].values)
        bond_lengths = df['bond_length'].values[bond_length_order]
        energies = df['electronic_energy'].values[bond_length_order]
        return np.array(bond_lengths), np.array(energies)

def dimer_binding_curve(
    df_qc, system_label, system_charge, excitation_level=0, calc_type='qc',
    use_qats=False, df_qats=None, specific_atom=0,
    direction=None, basis_set='cc-pV5Z', n_points=2, poly_order=4,
    remove_outliers=False, zscore_cutoff=3.0, considered_lambdas=None):
    """Compute the equilbirum bond length and energy using a polynomial fit.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    system_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    system_charge : :obj:`str`
        Overall change in the system.
    excitation_level : :obj:`int`, optional
        Specifies the desired electronic state. ``0`` for ground state and
        ``1`` for first excited state.
    calc_type : :obj:`str`, optional
        Specifies the method of the calculation. Can either be ``'qc'`` or
        ``'alchemy'``. Defaults to ``'qc'``.
    df_qats : :obj:`pandas.DataFrame`, optional
        APDFT dataframe. Needs to be specified if ``calc_type == 'alchemy'``.
    qa_order : :obj:`int`, optional
        Taylor series order used for APDFT predictions. Defaults to ``2``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'cc-pV5Z'``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
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
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    
    Returns
    -------
    :obj:`dict`
        The equilibrium bond length from all possible references. For the qc
        method the reference is the same as the system.
    :obj:`dict`
        The equilibrium energy from all possible references.
    """
    assert calc_type in ['qc', 'alchemy']
    df_sys = df_qc.query(
        'system == @system_label'
        '& charge == @system_charge'
    )
    multiplicity_sys = get_multiplicity(
        df_sys.query('lambda_value == 0'), excitation_level
    )
    df_sys = df_sys.query('multiplicity == @multiplicity_sys')

    if calc_type == 'qc':
        df_sys = df_sys.query('lambda_value == 0')
        bl_sys, e_sys = get_dimer_curve(df_sys, lambda_value=0)
        bl_sys = np.array(bl_sys)
        e_sys = np.array(e_sys)
        bl_dict = {system_label: bl_sys}
        e_dict = {system_label: e_sys}
        return bl_dict, e_dict
    
    elif calc_type == 'alchemy':
        sys_n_electron = df_sys.iloc[0]['n_electrons']
        sys_atomic_numbers = df_sys.iloc[0]['atomic_numbers']
        if use_qats:
            assert df_qats is not None
            df_selection = 'qats'
        else:
            df_selection = 'qc'
        df_refs = get_qa_refs(
            df_qc, df_qats, system_label, sys_n_electron,
            basis_set=basis_set, df_selection=df_selection,
            excitation_level=excitation_level,
            specific_atom=specific_atom, direction=direction
        )

        ref_system_labels = tuple(set(df_refs['system']))
        bl_dict = {}
        e_dict = {}
        for i in range(len(ref_system_labels)):
            ref_label = ref_system_labels[i]
            df_ref = df_refs.query('system == @ref_label')
            ref_atomic_numbers = df_ref.iloc[0]['atomic_numbers']
            ref_lambda_value = get_lambda_value(
                ref_atomic_numbers, sys_atomic_numbers,
                specific_atom=specific_atom
            )

            if considered_lambdas is not None:
                if ref_lambda_value not in considered_lambdas:
                    continue
            
            if not use_qats:
                bl_ref, e_ref = get_dimer_curve(
                    df_ref, lambda_value=ref_lambda_value,
                    use_qats=use_qats, qa_order=None
                )
            else:
                bl_ref = []
                e_ref = []

                max_qa_order = len(df_ref.iloc[0]['poly_coeff'])
                for qa_order in range(max_qa_order):
                    bl_ref_order, e_ref_order = get_dimer_curve(
                        df_ref, lambda_value=ref_lambda_value,
                        use_qats=use_qats, qa_order=qa_order
                    )
                    bl_ref.append(bl_ref_order)
                    e_ref.append(e_ref_order)
                bl_ref = np.array(bl_ref)
                e_ref = np.array(e_ref)
                
            bl_dict[ref_label] = bl_ref
            e_dict[ref_label] = e_ref
        
        return bl_dict, e_dict

def dimer_eq(
    df_qc, system_label, system_charge, excitation_level=0, calc_type='qc',
    use_qats=False, df_qats=None, specific_atom=0,
    direction=None, basis_set='cc-pV5Z', n_points=2, poly_order=4,
    remove_outliers=False, zscore_cutoff=3.0, considered_lambdas=None):
    """Compute the equilbirum bond length and energy using a polynomial fit.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    system_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    system_charge : :obj:`str`
        Overall change in the system.
    excitation_level : :obj:`int`, optional
        Specifies the desired electronic state. ``0`` for ground state and
        ``1`` for first excited state.
    calc_type : :obj:`str`, optional
        Specifies the method of the calculation. Can either be ``'qc'`` or
        ``'alchemy'``. Defaults to ``'qc'``.
    df_qats : :obj:`pandas.DataFrame`, optional
        APDFT dataframe. Needs to be specified if ``calc_type == 'alchemy'``.
    qa_order : :obj:`int`, optional
        Taylor series order used for APDFT predictions. Defaults to ``2``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'cc-pV5Z'``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
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
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    
    Returns
    -------
    :obj:`dict`
        The equilibrium bond length from all possible references. For the qc
        method the reference is the same as the system.
    :obj:`dict`
        The equilibrium energy from all possible references.
    """
    assert calc_type in ['qc', 'alchemy']
    
    bl_dict, e_dict = dimer_binding_curve(
        df_qc, system_label, system_charge, excitation_level=excitation_level,
        calc_type=calc_type, use_qats=use_qats, df_qats=df_qats,
        specific_atom=specific_atom, direction=direction, basis_set=basis_set,
        n_points=n_points, poly_order=poly_order,
        remove_outliers=remove_outliers, zscore_cutoff=zscore_cutoff,
        considered_lambdas=considered_lambdas
    )

    bl_eq_dict = {}
    e_eq_dict ={}
    for sys_label in bl_dict.keys():
        bl_sys = bl_dict[sys_label]
        e_sys = e_dict[sys_label]

        if len(bl_sys.shape) == 1:
            bl_sys = np.array([bl_sys])
            e_sys = np.array([e_sys])

        bl_eq = []
        e_eq = []
        for i in range(len(bl_sys)):
            bl_eq_i, e_eq_i = get_dimer_minimum(
                bl_sys[i], e_sys[i], n_points=n_points, poly_order=poly_order,
                remove_outliers=remove_outliers, zscore_cutoff=zscore_cutoff
            )
            bl_eq.append(bl_eq_i)
            e_eq.append(e_eq_i)
        bl_eq_dict[sys_label] = np.array(bl_eq)
        e_eq_dict[sys_label] = np.array(e_eq)
    
    return bl_eq_dict, e_eq_dict

def get_qc_change_charge_dimer(
    df_qc, target_label, delta_charge, target_initial_charge=0,
    change_signs=False, basis_set='cc-pV5Z',
    ignore_one_row=True, n_points=2, poly_order=4, remove_outliers=False,
    zscore_cutoff=3.0):
    """

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A pandas dataframe with quantum chemistry data.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'f.h'``.
    delta_charge : :obj:`str`
        Overall change in the initial target system.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.
    change_signs : :obj:`bool`, optional
        Multiply all predictions by -1. Used to correct the sign for computing
        electron affinities. Defaults to ``False``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'cc-pV5Z'``.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
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
    :obj:`float`
        Quantum chemistry predicted energy change due to changing the charge
        of the system.
    """
    # Checks that a bond_length is provided for dimers.
    assert len(df_qc.iloc[0]['atomic_numbers']) == 2
    
    # Selects initial target ground state QC data.
    target_initial_qc = df_qc[
        (df_qc.system == target_label) & (df_qc.charge == target_initial_charge)
        & (df_qc.lambda_value == 0.0) & (df_qc.basis_set == basis_set)
    ]
    ground_multiplicity_initial = get_multiplicity(target_initial_qc, 0)
    target_initial_qc = target_initial_qc.query(
        'multiplicity == @ground_multiplicity_initial'
    )
    target_initial_n_electrons = target_initial_qc.n_electrons.values[0]
    target_initial_bond_lengths, target_initial_energies = get_dimer_curve(
        target_initial_qc, lambda_value=None, use_qats=False,
        qa_order=None
    )
    _, target_initial_energy = get_dimer_minimum(
        target_initial_bond_lengths, target_initial_energies, n_points=n_points,
        poly_order=poly_order, remove_outliers=remove_outliers,
        zscore_cutoff=zscore_cutoff
    )
    
    # Selects final target ground state QC data.
    target_final_n_electrons = target_initial_n_electrons - delta_charge
    
    target_final_qc = df_qc[
        (df_qc.system == target_label)
        & (df_qc.lambda_value == 0.0)
        & (df_qc.n_electrons == target_final_n_electrons)
        & (df_qc.basis_set == basis_set)
    ]
    ground_multiplicity_final = get_multiplicity(target_final_qc, 0)
    target_final_qc = target_final_qc.query(
        'multiplicity == @ground_multiplicity_final'
    )
    target_final_bond_lengths, target_final_energies = get_dimer_curve(
        target_final_qc, lambda_value=None, use_qats=False, qa_order=None
    )
    _, target_final_energy = get_dimer_minimum(
        target_final_bond_lengths, target_final_energies, n_points=n_points,
        poly_order=poly_order, remove_outliers=remove_outliers,
        zscore_cutoff=zscore_cutoff
    )

    e_diff = target_final_energy - target_initial_energy
    if change_signs:
        e_diff *= -1
    return e_diff

def get_qats_change_charge_dimer(
    df_qc, df_qats, target_label, delta_charge,
    target_initial_charge=0, change_signs=False, basis_set='cc-pV5Z',
    use_qats=True, lambda_specific_atom=None, lambda_direction=None,
    ignore_one_row=True, poly_order=4, n_points=2, remove_outliers=False,
    considered_lambdas=None, compute_difference=False):
    """Use an APDFT reference to predict the energy change due to adding or
    removing an electron to dimers.

    The minimum energy from a fitted parabola is used for each state.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A pandas dataframe with quantum chemistry data. It should have the
        following columns (from `get_qc_dframe`): system, atomic_numbers,
        charge, multiplicity, n_electrons, qc_method, basis_set, lambda_range,
        finite_diff_delta, finite_diff_acc, poly_coeff.
    df_qats : :obj:`pandas.DataFrame`
        A pandas dataframe with APDFT data. It should have the
        following columns (from `get_qats_dframe`): system, atomic_numbers,
        charge, multiplicity, n_electrons, qc_method, basis_set, lambda,
        electronic_energy, hf_energy, and correlation_energy.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'c'``, ``'si'``, or ``'f.h'``.
    delta_charge : :obj:`str`
        Overall change in the initial target system.
    target_initial_charge : :obj:`int`
        Specifies the initial charge state of the target system. For example,
        the first ionization energy is the energy difference going from
        charge ``0 -> 1``, so ``target_initial_charge`` must equal ``0``.
    change_signs : :obj:`bool`, optional
        Multiply all predictions by -1. Used to correct the sign for computing
        electron affinities. Defaults to ``False``.
    basis_set : :obj:`str`, optional
        Specifies the basis set to use for predictions. Defaults to
        ``'cc-pVQZ'``.
    use_qats : :obj:`bool`, optional
        Use a Taylor series approximation (with finite differences) to make
        APDFTn predictions (where n is the order). Defaults to ``True``.
    lambda_specific_atom : :obj:`int`, optional
        Applies the entire lambda change to a single atom in dimers. For
        example, OH -> FH+ would be a lambda change of +1 only on the first
        atom.
    lambda_direction : :obj:`str`, optional
        Defines the direction of lambda changes for dimers. ``'counter'`` is
        is where one atom increases and the other decreases their nuclear
        charge (e.g., CO -> BF).
        If the atomic numbers of the reference are the same, the first atom's
        nuclear charge is decreased and the second is increased. IF they are
        different, the atom with the largest atomic number increases by lambda.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    poly_order : :obj:`int`, optional
        Maximum order of the fitted polynomial. Defaults to ``2``.
    n_points : :obj:`int`, optional
        The number of surrounding points on either side of the minimum bond
        length. Defaults to ``2``.
    remove_outliers : :obj:`bool`, optional
        Do not include bond lengths that are marked as outliers by their z
        score. Defaults to ``False``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    compute_difference : :obj:`bool`, optional
        Return the difference of APDFTn - APDFT predictions; i.e., the error of
        using a Taylor series approximation with repsect to the alchemical
        potential energy surface. Defaults to ``False``.
    """    
    assert delta_charge != 0
    assert len(df_qc.iloc[0]['atomic_numbers']) == 2
    if compute_difference: assert use_qats == True

    # Selects initial target ground state QC data.
    target_initial_qc = df_qc[
        (df_qc.system == target_label) & (df_qc.charge == target_initial_charge)
        & (df_qc.lambda_value == 0.0) & (df_qc.basis_set == basis_set)
    ]
    
    ground_multiplicity_initial = get_multiplicity(target_initial_qc, 0)
    target_initial_qc = target_initial_qc.query(
        'multiplicity == @ground_multiplicity_initial'
    )
    assert len(target_initial_qc) > 1
    target_initial_n_electrons = target_initial_qc.iloc[0]['n_electrons']
    target_atomic_numbers = target_initial_qc.iloc[0]['atomic_numbers']

    # Performs checks on lambda selections.
    assert lambda_specific_atom is not None or lambda_direction is not None

    # Selects final target ground state QC data.
    target_final_n_electrons = target_initial_n_electrons - delta_charge
    target_final_qc = df_qc[
        (df_qc.system == target_label)
        & (df_qc.charge == target_initial_charge + delta_charge)
        & (df_qc.lambda_value == 0.0) & (df_qc.basis_set == basis_set)
    ]
    ground_multiplicity_final = get_multiplicity(target_final_qc, 0)
    target_final_qc = target_final_qc.query(
        'multiplicity == @ground_multiplicity_final'
    )

    # Get all available references for the initial target based on ground state
    # energies.
    avail_ref_final_sys = set(
        df_qats[
            (df_qats.system != target_label)
            & (df_qats.n_electrons == target_final_n_electrons)
            & (df_qats.basis_set == basis_set)
        ].system.values
    )
    
    ref_initial_qats = df_qats.query(
        'n_electrons == @target_initial_n_electrons'
        '& basis_set == @basis_set'
        '& multiplicity == @ground_multiplicity_initial'
    )
    ref_initial_qats = ref_initial_qats[
        ref_initial_qats['system'].isin(avail_ref_final_sys)
    ]

    # Get all available references for the final target based on ground state
    # energies.
    ref_final_qats = df_qats.query(
        'n_electrons == @target_final_n_electrons'
        '& basis_set == @basis_set'
        '& multiplicity == @ground_multiplicity_final'
    )
    ref_final_qats = ref_final_qats[
        ref_final_qats['system'].isin(ref_initial_qats.system)
    ]

    # Checks that the size of initial and final dataframe is the same
    assert len(ref_initial_qats) == len(ref_final_qats)

    predictions = {}
    for system in set(ref_initial_qats.system):
        ref_initial = ref_initial_qats.query('system == @system')
        ref_final = ref_final_qats.query('system == @system')

        lambda_initial = get_lambda_value(
            ref_initial.iloc[0]['atomic_numbers'], target_atomic_numbers,
            specific_atom=lambda_specific_atom, direction=lambda_direction
        )
        lambda_final = get_lambda_value(
            ref_final.iloc[0]['atomic_numbers'], target_atomic_numbers,
            specific_atom=lambda_specific_atom, direction=lambda_direction
        )
        assert lambda_initial == lambda_final
        if considered_lambdas is not None:
            if lambda_initial not in considered_lambdas:
                continue

        bond_length_order_initial = np.argsort(ref_initial['bond_length'].values)
        bond_length_order_final = np.argsort(ref_final['bond_length'].values)

        if use_qats or compute_difference == True:
            order_preds = []
            for order in range(len(ref_initial.iloc[0]['poly_coeff'])):
                bond_lengths_initial, energies_initial = get_dimer_curve(
                    ref_initial, lambda_value=lambda_initial, use_qats=True,
                    qa_order=order
                )
                _, e_target_initial = get_dimer_minimum(
                    bond_lengths_initial, energies_initial, n_points=n_points,
                    remove_outliers=remove_outliers
                )

                bond_lengths_final, energies_final = get_dimer_curve(
                    ref_final, lambda_value=lambda_final, use_qats=True,
                    qa_order=order
                )
                _, e_target_final = get_dimer_minimum(
                    bond_lengths_final, energies_final, n_points=n_points,
                    remove_outliers=remove_outliers
                )
                
                e_diff = e_target_final - e_target_initial
                if change_signs:
                    e_diff *= -1
                order_preds.append(e_diff)
            predictions[system] = np.array(order_preds)
        if not use_qats or compute_difference == True:
            chrg_ref_initial = ref_initial.iloc[0]['charge']
            mult_ref_initial = ref_initial.iloc[0]['multiplicity']
            ref_initial_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_initial'
                '& multiplicity == @mult_ref_initial'
                '& basis_set == @basis_set'
            )
            bond_lengths_initial, energies_initial = get_dimer_curve(
                ref_initial_qc, lambda_value=lambda_initial, use_qats=False,
                qa_order=None
            )
            _, e_target_initial = get_dimer_minimum(
                bond_lengths_initial, energies_initial, n_points=n_points,
                remove_outliers=remove_outliers
            )
            
            chrg_ref_final = ref_final.iloc[0]['charge']
            mult_ref_final = ref_final.iloc[0]['multiplicity']
            ref_final_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_final'
                '& multiplicity == @mult_ref_final'
                '& basis_set == @basis_set'
            )
            bond_lengths_final, energies_final = get_dimer_curve(
                ref_final_qc, lambda_value=lambda_final, use_qats=False, qa_order=None
            )
            _, e_target_final = get_dimer_minimum(
                bond_lengths_final, energies_final, n_points=n_points,
                remove_outliers=remove_outliers
            )
            
            e_diff = e_target_final - e_target_initial
            if change_signs:
                e_diff *= -1
            
            if compute_difference:
                pred_diff = [i - e_diff for i in predictions[system]]
                predictions[system] = np.array(pred_diff)
            else:
                predictions[system] = np.array([e_diff])

    return predictions

def get_qc_excitation(
    df_qc, target_label, target_charge=0, excitation_level=1,
    basis_set='aug-cc-pVQZ', ignore_one_row=True
):
    """Calculate excitation energies using a quantum-chemistry dataframe.

    Parameters
    ----------
    df_qc : :obj:`pandas.dataframe`
        Quantum chemistry dataframe.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'c'``, ``'si'``, or ``'f.h'``.

    Returns
    -------
    :obj:`numpy.float64`
        The excitation energy in Hartrees.
    """
    # Selects initial target ground state QC data.
    target_qc = df_qc[
        (df_qc.system == target_label)
        & (df_qc.charge == target_charge)
        & (df_qc.lambda_value == 0.0)
        & (df_qc.basis_set == basis_set)
    ]
    if len(target_qc) == 0 or len(target_qc) == 1:
        # Often we do not have the data to make the prediction.
        # So we return NaN.
        return np.nan
    elif len(target_qc) > 1:
        target_initial_qc = select_state(
            target_qc, 0, ignore_one_row=ignore_one_row
        )
        assert len(target_initial_qc) == 1  # Should only have one row.
        target_final_qc = select_state(
            target_qc, excitation_level, ignore_one_row=ignore_one_row
        )
        assert len(target_final_qc) == 1  # Should only have one row.

    e_diff = target_final_qc.iloc[0]['electronic_energy'] \
             - target_initial_qc.iloc[0]['electronic_energy']
    return e_diff

def get_qats_excitation(
    df_qc, df_qats, target_label, target_charge=0, excitation_level=1,
    basis_set='aug-cc-pVQZ', use_qats=True, ignore_one_row=True,
    considered_lambdas=None, compute_difference=False
):
    """Calculate excitation energies using a quantum-chemistry dataframe.

    Parameters
    ----------
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``True``.
    considered_lambdas : :obj:`list`, optional
        Allows specification of lambda values that will be considered. ``None``
        will allow all lambdas to be valid, ``[1, -1]`` would only report
        predictions using references using a lambda of ``1`` or ``-1``.
    compute_difference : :obj:`bool`, optional
        Return the difference of APDFTn - APDFT predictions; i.e., the error of
        using a Taylor series approximation with repsect to the alchemical
        potential energy surface.

    Returns
    -------
    :obj:`numpy.float64`
        The excitation energy in Hartrees.
    """
    if compute_difference:
        assert use_qats == True

    # Selects initial target ground state QC data.
    target_qc = df_qc[
        (df_qc.system == target_label)
        & (df_qc.charge == target_charge)
        & (df_qc.lambda_value == 0.0)
        & (df_qc.basis_set == basis_set)
    ]
    if len(target_qc) == 0 or len(target_qc) == 1:
        # Often we do not have the data to make the prediction.
        # So we return nothing.
        return {}
    elif len(target_qc) > 1:
        n_electrons = list(set(target_qc.n_electrons.values))
        assert len(n_electrons) == 1
        n_electrons = n_electrons[0]
        target_initial_qc = select_state(
            target_qc, 0, ignore_one_row=ignore_one_row
        )
        assert len(target_initial_qc) == 1  # Should only have one row.
        target_final_qc = select_state(
            target_qc, excitation_level, ignore_one_row=ignore_one_row
        )
        assert len(target_final_qc) == 1  # Should only have one row.
    target_atomic_numbers = target_initial_qc.iloc[0]['atomic_numbers']
    

    ref_qats = get_qa_refs(
        df_qc, df_qats, target_label, n_electrons,
        basis_set=basis_set
    )
    if len(ref_qats) == 0 or len(ref_qats) == 1:
        # Often we do not have the data to make the prediction.
        # So we return nothing.
        return {}
    elif len(ref_qats) > 1:
        ref_initial_qats = select_state(
            ref_qats, 0, ignore_one_row=ignore_one_row
        )
        ref_final_qats = select_state(
            ref_qats, excitation_level, ignore_one_row=ignore_one_row
        )

    # Checks that the size of initial and final dataframe is the same
    assert len(ref_initial_qats) == len(ref_final_qats)

    predictions = {}
    for system in ref_initial_qats.system:
        ref_initial = ref_initial_qats.query('system == @system')
        lambda_initial = get_lambda_value(
            ref_initial.iloc[0]['atomic_numbers'], target_atomic_numbers
        )
        
        ref_final = ref_final_qats.query('system == @system')
        lambda_final = get_lambda_value(
            ref_final.iloc[0]['atomic_numbers'], target_atomic_numbers
        )

        assert lambda_initial == lambda_final
        if considered_lambdas is not None:
            if lambda_initial not in considered_lambdas:
                continue

        if use_qats or compute_difference == True:
            order_preds = []
            for order in range(len(ref_initial.iloc[0]['poly_coeff'])):
                e_target_initial = calc_qats_pred(ref_initial.iloc[0]['poly_coeff'], order, lambda_initial)
                e_target_final = calc_qats_pred(ref_final.iloc[0]['poly_coeff'], order, lambda_final)
                e_diff = (e_target_final - e_target_initial)[0]
                order_preds.append(e_diff)
            predictions[system] = np.array(order_preds)
        if not use_qats or compute_difference == True:
            chrg_ref_initial = ref_initial.iloc[0]['charge']
            mult_ref_initial = ref_initial.iloc[0]['multiplicity']
            
            ref_initial_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_initial'
                '& multiplicity == @mult_ref_initial'
                '& basis_set == @basis_set'
            )
            assert len(ref_initial_qc) == 1
            e_target_initial = ref_initial_qc.iloc[0]['electronic_energy']
            
            chrg_ref_final = ref_final.iloc[0]['charge']
            mult_ref_final = ref_final.iloc[0]['multiplicity']
            ref_final_qc = df_qc.query(
                'system == @system & lambda_value == @lambda_initial'
                '& charge == @chrg_ref_final'
                '& multiplicity == @mult_ref_final'
                '& basis_set == @basis_set'
            )
            e_target_final = ref_final_qc.iloc[0]['electronic_energy']
            e_diff = e_target_final - e_target_initial
            
            if compute_difference:
                pred_diff = [i - e_diff for i in predictions[system]]
                predictions[system] = np.array(pred_diff)
            else:
                predictions[system] = np.array([e_diff])

    return predictions
