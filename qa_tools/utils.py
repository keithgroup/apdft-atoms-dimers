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

import json
import numpy as np
import pandas as pd
from scipy import optimize, stats

####    GLOBAL INFORMATION    ####
hartree_to_ev = lambda x: x * 27.21138505

all_atom_systems = ('h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne', 'na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar')
all_dimer_systems = ('li.h', 'be.h', 'b.h', 'c.h', 'n.h', 'o.h', 'f.h', 'ne.h')
post_hf_methods = ('hf', 'uhf', 'ccsd', 'uccsd', 'ccsd(t)', 'uccsd(t)')

all_atom_systems_by_row = (('h', 'he'), ('li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne'), ('na', 'mg', 'al', 'si', 'p', 's', 'cl', 'ar'))

# These values are from DOI: 10.1021/ct100396y
cbs_extrap_alphas = {
    'ano': {'2/3': 5.41, '3/4': 4.48},
    'aug': {'2/3': 4.30, '3/4': 5.79}
}
cbs_extrap_betas = {
    'ano': {'2/3': 2.43, '3/4': 2.97},
    'aug': {'2/3': 2.51, '3/4': 3.05}
}
basis_cardinals = {
    'aug-cc-pVTZ': 3, 'aug-cc-pVQZ': 4
}

# Sources: Haynes, W. M., Lide, D. R., & Bruno, T. J. (2017). CRC Handbook of Chemistry and Physics (Vol. 2016–2017, 97th edition Editor–in–Chief: W.M. Haynes). CRC Press.
# First IP reported in CRC from: Sansonetti, J . E ., Martin, W . C ., and Young, S . L ., Handbook of Basic Atomic Spectroscopic Data (version 1.1), NIST Physical Data web site (October 2004); J. Phys. Chem. Ref. Data, 34, 1559, 2005.
ionization_energy_CRC = (
    np.nan, 24.587387,
    5.391719, 9.32270, 8.29802, 11.26030, 14.5341, 13.61805, 17.4228, 21.56454,
    5.139076, 7.646235, 5.985768, 8.15168, 10.48669, 10.36001, 12.96763, 15.759610
)  # eV
# Sources: Haynes, W. M., Lide, D. R., & Bruno, T. J. (2017). CRC Handbook of Chemistry and Physics (Vol. 2016–2017, 97th edition Editor–in–Chief: W.M. Haynes). CRC Press.
ionization_energy_CRC_second = (
    np.nan, np.nan, 75.6400, 18.21114, 25.1548, 24.3833, 29.6013, 35.1211, 34.9708, 40.96296,
    47.2864, 15.03527, 18.82855, 16.34584, 19.7695, 23.33788, 23.8136, 27.62966
)  # eV # second ionization energy

# Sources: Haynes, W. M., Lide, D. R., & Bruno, T. J. (2017). CRC Handbook of Chemistry and Physics (Vol. 2016–2017, 97th edition Editor–in–Chief: W.M. Haynes). CRC Press.
e_affinity_CRC = (
    0.754195, np.nan,
    0.618049, np.nan, 0.279723, 1.262119, np.nan, 1.4611135, 3.4011897, np.nan,
    0.547926, np.nan, 0.43283, 1.3895211, 0.746607, 2.077104, 3.612725, np.nan
)  # eV
e_affinity_type = (
    'LPT', 'calc',
    'LPT', 'calc', 'LPES', 'e-scat', 'DA', 'LPES', 'LPT', 'calc',
    'LPT', 'calc', 'LPES', 'LPT', 'LPT', 'LPT', 'LPT', 'LPT'
)
# Sources: NIST Atomic Spectra Database Levels Data; Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2020). NIST Atomic Spectra Database (ver. 5.8), [Online]. Available: https://physics.nist.gov/asd [2021, October 1]. National Institute of Standards and Technology, Gaithersburg, MD. DOI: https://doi.org/10.18434/T4W30F    
excitation_energy_NIST = (
    np.nan, 19.81961484203, 57.469, 2.724963, 3.55183, 1.2637284, 2.3835298, 1.9673642, 14.683178,
    18.72638145, 32.7002, 2.7091049, 3.598072, 0.7809579, 1.408587, 1.1454415, 10.6297965, 13.28263902
) # eV # excitation energies for neutral atoms

#{charge1: (mult_ground, mult_excited1, mult_excited2, etc.), charge2: (mult_ground, mult_excited1, mult_excited2, etc.)}
atom_states = {
    'h': {0: (2,), -1: (1, 3), -2: (2, 4)},
    'he': {0: (1, 3), 1: (2,), -1: (2, 4), -2: (1, 3)},
    'li': {0: (2, 4), 1: (1, 3), 2: (2,), -1: (1, 3), -2: (2, 4)},
    'be': {0: (1, 3), 1: (2, 4), 2: (1, 3), -1: (2, 4), -2: (3, 1)},
    'b': {0: (2, 4), 1: (1, 3), 2: (2, 4), -1: (3, 1), -2: (4, 2)},
    'c': {0: (3, 1), 1: (2, 4), 2: (1, 3), -1: (4, 2), -2: (3, 1)},
    'n': {0: (4, 2), 1: (3, 1), 2: (2, 4), -1: (3, 1), -2: (2, 4)},
    'o': {0: (3, 1), 1: (4, 2), 2: (3, 1), -1: (2, 4), -2: (1, 3)},
    'f': {0: (2, 4), 1: (3, 1), 2: (4, 2), -1: (1, 3), -2: (2, 4)},
    'ne': {0: (1, 3), 1: (2, 4), 2: (3, 1), -1: (2, 4), -2: (1, 3)},
    'na': {0: (2, 4), 1: (1, 3), 2: (2, 4), -1: (1, 3), -2: (2, 4)},
    'mg': {0: (1, 3), 1: (2, 4), 2: (1, 3), -1: (2, 4), -2: (3, 1)},
    'al': {0: (2, 4), 1: (1, 3), 2: (2, 4), -1: (3, 1), -2: (4, 2)},
    'si': {0: (3, 1), 1: (2, 4), 2: (1, 3), -1: (4, 2), -2: (3, 1)},
    'p': {0: (4, 2), 1: (3, 1), 2: (2, 4), -1: (3, 1), -2: (2, 4)},
    's': {0: (3, 1), 1: (4, 2), 2: (3, 1), -1: (2, 4), -2: (1, 3)},
    'cl': {0: (2, 4), 1: (3, 1), 2: (4, 2), -1: (1, 3)},
    'ar': {0: (1, 3), 1: (2, 4), 2: (3, 1)},
}

element_to_z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23,
    'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31,'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
    'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
    'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
    'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79,
    'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93,
    'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106,
    'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
    'Uuq': 114, 'Uuh': 116,
}
element_to_z = {key.lower():value for key,value in element_to_z.items()}

def read_json(json_path):
    """Read JSON file.
    
    Parameters
    ----------
    json_path : :obj:`str`
        Path to json file.
    
    Returns
    -------
    :obj:`dict`
        Contents of JSON file.
    """
    with open(json_path, 'r') as reader:
        json_dict = json.load(reader)
    
    return json_dict

def system_to_atomic_numbers(system_label):
    """Converts the standard system label into atomic numbers.

    Parameters
    ----------
    system_label : :obj:`str`
        Specifies the system by stringing together element symbols with a
        ``'.'`` joining them. For example, `be` and `n.h`.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Atomic numbers in the system in the same order as the system_label.
    """
    atomic_numbers = np.array(
        [element_to_z[i] for i in system_label.split('.')]
    )
    return atomic_numbers

def get_lambda_value(
    ref_atomic_numbers, target_atomic_numbers, specific_atom=None,
    direction=None):
    """The overall alchemical lambda value between two atomic systems.
    
    Parameters
    ----------
    ref_atomic_numbers : :obj:`numpy.ndarray`, :obj:`list`
        Atomic numbers of all atoms in the reference system.
    target_atomic_numbers : :obj:`numpy.ndarray`, :obj:`list`
        Atomic numbers of all atoms in the target system.
    specific_atom : :obj:`int`, optional
        Applies the entire lambda change to a single atom in dimers. For
        example, OH -> FH+ would be a lambda change of +1 only on the first
        atom. Defaults to ``None``.
    direction : :obj:`str`, optional
        Defines the direction of lambda changes for dimers. ``'counter'`` is
        is where one atom increases and the other decreases their nuclear
        charge (e.g., CO -> BF).
        
        If the atomic numbers of the reference are the same, the first atom's
        nuclear charge is decreased and the second is increased. If they are
        different, the atom with the largest atomic number increases by lambda.
        Defaults to ``None``.

    Returns
    -------
    :obj:`float`
        Overall lambda value to get from reference to target system.
    """
    assert len(ref_atomic_numbers) == len(target_atomic_numbers)

    # Handles atom systems.
    if len(ref_atomic_numbers) == 1:
        # Determines the number of decimal points to include in lambda value.
        ref_atomic_numbers_deci = str(ref_atomic_numbers[0]).split('.')
        target_atomic_numbers_deci = str(target_atomic_numbers[0]).split('.')
        if len(ref_atomic_numbers_deci) > 1 or len(target_atomic_numbers_deci):
            num_deci = max(
                len(ref_atomic_numbers_deci[-1]),
                len(target_atomic_numbers_deci[-1])
            )
        else:
            num_deci = 0
        
        # Computes lambda value.
        lambda_value = target_atomic_numbers - ref_atomic_numbers
        if type(lambda_value) is np.ndarray:
            lambda_value = lambda_value[0]
        return round(lambda_value, num_deci)
    
    # Handles dimer systems.
    elif len(ref_atomic_numbers) == 2:
        if specific_atom is not None:
            assert direction is None
            return float(
                target_atomic_numbers[specific_atom] - ref_atomic_numbers[specific_atom]
            )
        elif direction is not None:
            assert specific_atom is None
            if direction.lower() == 'counter':
                if ref_atomic_numbers[0] == ref_atomic_numbers[1]:
                    idx = 1
                else:
                    idx = np.argmax(ref_atomic_numbers)
                return float(
                    target_atomic_numbers[idx] - ref_atomic_numbers[idx]
                )
            else:
                raise ValueError(f'{direction} direction is not supported.')
        else:
            raise ValueError(
                f'specific_atom and direction cannot both be None or specified'
            )
    else:
        raise ValueError(
            f'Only one or two atoms are supported. There are {len(ref_atomic_numbers)} in this system.'
        )

def get_qa_refs(
    df_qc, df_qats, target_label, target_n_electrons, basis_set='aug-cc-pV5Z',
    df_selection='qats', excitation_level=None, specific_atom=None,
    direction=None, considered_lambdas=None):
    """Returns dataframe with all possible QATS references for a given target system.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A dataframe with quantum chemistry data.
    df_qats : :obj:`pandas.DataFrame`
        A dataframe with QATS data.
    target_label : :obj:`str`
        Atoms in the system. For example, ``'c'``, ``'si'``, or ``'f.h'``.
    target_n_electrons : :obj:`int`
        The number of electrons in the desired target system. All quantum
        alchemy predictions are isoelectronic (same number of electrons).
    basis_set : :obj:`str`, optional
        Desired basis sets the predictions are from. Defaults to
        ``aug-cc-pV5Z``.
    df_selection : :obj:`str`, optional
        Which dataframe is desired. ``'qc'`` for quantum alchemy predictions
        at specific lambda values or ``'qats'`` to make Taylor series
        predictions (with the polynomial coefficients).
    excitation_level : :obj:`int`, optional
        Selects the excitation levels of the references. ``0`` for ground and
        ``1`` for first excited state. Defaults to ``None`` which means no
        selection is made. Defaults to ``None``.
    specific_atom : :obj:`int`, optional
        Applies the entire lambda change to a single atom in dimers. For
        example, OH -> FH+ would be a lambda change of +1 only on the first
        atom. Defaults to ``None``.
    direction : :obj:`str`, optional
        Defines the direction of lambda changes for dimers. ``'counter'`` is
        is where one atom increases and the other decreases their nuclear
        charge (e.g., CO -> BF).
        
        If the atomic numbers of the reference are the same, the first atom's
        nuclear charge is decreased and the second is increased. If they are
        different, the atom with the largest atomic number increases by lambda.
        Defaults to ``None``.
    considered_lambdas : :obj:`list`
        Specify the only lambda values that will be considered. ``None``
        will allow all lambdas to be valid. ``[1, -1]`` would only return
        references that predict the target with lambdas of ``1`` or ``-1``.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
        Selected qc or qats dataframe with quantum alchemy references able
        to predict the desired target.
    """
    if df_selection == 'qats':
        if 'electronic_energy' not in df_qats.columns.values:
            df_qats = add_energies_to_df_qats(df_qc, df_qats)
        df_ref = df_qats.query(
            'system != @target_label'
            '& n_electrons == @target_n_electrons'
            '& basis_set == @basis_set'
        )
    elif df_selection == 'qc':
        df_ref = df_qc.query(
            'system != @target_label'
            '& n_electrons == @target_n_electrons'
            '& basis_set == @basis_set'
        )
        # Selects lambda values
        refs_sys = tuple(set(df_ref['system']))
        for i in range(len(refs_sys)):
            sys_label = refs_sys[i]
            df_refs_sys = df_ref.query('system == @sys_label')
            if len(set(df_refs_sys['lambda_value'])) != 1:
                target_rows = df_qc.query(
                    'system == @target_label'
                    '& n_electrons == @target_n_electrons'
                    '& basis_set == @basis_set'
                )
                target_atomic_numbers = target_rows.iloc[0]['atomic_numbers']
                ref_atomic_numbers = df_refs_sys.iloc[0]['atomic_numbers']
                sys_lambda_value = get_lambda_value(
                    ref_atomic_numbers, target_atomic_numbers,
                    specific_atom=specific_atom, direction=direction
                )
                drop_filter = df_ref.query(
                    'system == @sys_label & lambda_value != @sys_lambda_value'
                ).index
                df_ref = df_ref.drop(drop_filter)
    
    # Filters all quantum alchemy references by specified lambda values.
    if considered_lambdas is not None:
        refs_sys = tuple(set(df_ref['system']))
        target_atomic_numbers = df_qc.query('system == @target_label').iloc[0]['atomic_numbers']
        for i in range(len(refs_sys)):
            # Gets lambda value to get from reference to target.
            sys_label = refs_sys[i]
            ref_sys = df_ref.query('system == @sys_label')
            ref_atomic_numbers = ref_sys.iloc[0]['atomic_numbers']
            lambda_value = get_lambda_value(
                ref_atomic_numbers, target_atomic_numbers,
                specific_atom=specific_atom, direction=direction
            )

            # Removes undesired lambda values.
            if lambda_value not in considered_lambdas:
                drop_filter = (df_ref['system'] == sys_label)
                df_ref = df_ref[~drop_filter]

    # Selects the desired excitation level if specified.
    if excitation_level is not None:
        assert excitation_level in [0, 1]
        refs_sys = tuple(set(df_ref['system']))
        for i in range(len(refs_sys)):
            sys_label = refs_sys[i]

            # Gets multiplicity
            df_mult = df_ref.query(
                'system == @sys_label'
                '& n_electrons == @target_n_electrons'
                '& basis_set == @basis_set'
            )
            assert 'electronic_energy' in df_mult.columns
            ref_sys_multiplicity = get_multiplicity(
                df_mult, excitation_level
            )
            
            # Removes undesired states.
            drop_filter = df_ref.query(
                'system == @sys_label & multiplicity != @ref_sys_multiplicity'
            ).index
            df_ref = df_ref.drop(drop_filter)

    return df_ref

def add_energies_to_df_qats(df_qc, df_qats):
    """Adds electronic energy data to the QATS dataframe.

    Information is used to select states out of the QATS dataframe.

    Parameters
    ----------
    df_qc : :obj:`pandas.dataframe`
        The quantum chemistry dataframe.
    df_qats : :obj:`pandas.dataframe`
        The QATS dataframe.
    
    Returns
    -------
    :obj:`pandas.dataframe`
        The QATS dataframe with the added electronic energy data.
    """
    df_qc = df_qc.query('lambda_value == 0')

    qc_columns = [
        'system', 'charge', 'multiplicity', 'n_electrons', 'qc_method',
        'basis_set', 'electronic_energy'
    ]
    shared_columns = [
        'system', 'charge', 'multiplicity', 'n_electrons', 'qc_method',
        'basis_set'
    ]
    if 'bond_length' in df_qc.columns.values:
        qc_columns.insert(5, 'bond_length')
        shared_columns.append('bond_length')
    df_qats = pd.merge(
        df_qats,
        df_qc[qc_columns],
        how='inner',
        on=shared_columns,
    )
    return df_qats

def select_state(df, excitation_level, ignore_one_row=False):
    """Filters dataframes to only contain the desired excitation level.

    Only should be one basis set in this dataframe.

    Parameters
    ----------
    df : :obj:`pandas.dataframe`
        A pandas dataframe. Must have `'electronic_energy'` as a column.
    excitation_level : :obj:`int`
        Desired excited state. `0` for ground, `1` for first excited state, etc.
    ignore_one_row : :obj:`bool`, optional
        Do not care if there is only one row. Having only one row could be
        indicative of good filtering. Or, missing data. If you know there is
        no missing data, then you can change this to `True`. Defaults to
        `False`.
    
    Returns
    -------
    :obj:`pandas.dataframe`
        The dataframe with only the selected electronic state remaining.
    """
    assert 'electronic_energy' in df.columns

    try:
        assert len(set(df.basis_set.values)) == 1
    except AssertionError:
        if len(df) == 0:
            raise ValueError('There are no rows in the dataframe to select')
        else:
            raise
    if not ignore_one_row:
        try:
            assert len(df) > 1
        except AssertionError:
            raise ValueError('There is only one row in the dataframe')
    
    # Select excited state by energy.
    grouped = df.groupby('system')
    excitation_slice = []
    for _, indices in grouped.indices.items():
        system_rows = df.iloc[indices]
        excitation_indices = np.where(np.argsort(system_rows.electronic_energy.values) == excitation_level)[0][0]
        excitation_slice.append(indices[excitation_indices])
    df = df.iloc[excitation_slice]
    return df

def get_multiplicity(df, excitation_level, ignore_one_row=False):
    """Multiplicity of the specified ground or excited state.

    Used for dimer dataframes where we have multiple states (usually two) with
    several rows for multiple bond lengths. Dataframe can contain multiple
    systems and lambda values.

    Parameters
    ----------
    df : :obj:`pandas.dataframe`
        QC or QATS dataframe with at least ``atomic_numbers``, ``multiplicity``,
        and ``electronic_energy``.
    excitation_level : :obj:`int`
        Electronic state of the system with respect to the ground state. ``0``
        represents the ground state, ``1`` the first excited state, etc.
    ignore_one_row : :obj:`bool`, optional
        Used to control errors in ``state_selection`` when there is missing
        data (i.e., just one state). If ``True``, no errors are raised. Defaults
        to ``False``.

    Returns
    -------
    :obj:`int`
        System multiplicity of the desired excitation_level.
    """
    if len(df.iloc[0]['atomic_numbers']) == 2:
        is_dimer = True
        # More careful selection of a row. Will be done by finding the row with
        # the lowest energy.
        row = df[df.electronic_energy == df.electronic_energy.min()].iloc[0]
    else:
        is_dimer = False
        row = df.iloc[0]
    
    if 'lambda_value' in df.columns.values:
        lambda_selection = row['lambda_value']
    
    # Prepares the selection dataframe.
    if is_dimer:
        bond_length_selection = row['bond_length']
        df_selection = df.query(
            'bond_length == @bond_length_selection'
        )
        if 'lambda_value' in df.columns.values:
            df_selection = df_selection.query(
                'lambda_value == @lambda_selection'
            )
    else:
        if 'lambda_value' in df.columns.values:
            df_selection = df.query('lambda_value == @lambda_selection')
        else:
            df_selection = df
    assert len(df_selection) == 2

    df_mult = select_state(
        df_selection, excitation_level, ignore_one_row=ignore_one_row
    )
    multiplicity = df_mult.iloc[0]['multiplicity']
    return int(multiplicity)

def _get_ptable_row(atom_label):
    """Number of the periodic table an element is in.

    Parameters
    ----------
    atom_label : :obj:`str`
        The element symbol.
    
    Returns
    -------
    :obj:`int`
        Row of the periodic table the element is from.
    """
    for p_row in range(len(all_atom_systems_by_row)):
        if atom_label in all_atom_systems_by_row[p_row]:
            return p_row

def _remove_dimer_outliers(bond_lengths, energies, zscore_cutoff=3.0):
    """Removes outliers 
    """
    z_score = stats.zscore(energies)
    idx_keep = np.where(z_score < zscore_cutoff)[0]
    return bond_lengths[idx_keep], energies[idx_keep]

def fit_dimer_poly(bond_lengths, energies, n_points=2, poly_order=4,
    remove_outliers=False, zscore_cutoff=3.0):
    """Fits a nth-order polynomial to the binding curve minimum.

    Parameters
    ----------
    bond_lengths : :obj:`numpy.ndarray`
        All bond lengths considered.
    energies : :obj:`numpy.ndarray`
        Corresponding electronic energies.
    n_points : :obj:`int`, optional
        The number of surrounding points on either side (forward or backward)
        of the minimum bond length. Defaults to ``2`` which would fit to a total
        of five points.
    poly_order : :obj:`int`, optional
        Maximum order of the fitted polynomial. Defaults to ``4``.
    remove_outliers : :obj:`bool`, optional
        Do not include bond lengths that are marked as outliers by their z
        score. Useful if there are cases where one quantum alchemy prediction
        is significantly off (i.e., errors on the order of hundreds of eV).
        Defaults to ``False``.
    zscore_cutoff : :obj:`float`, optional
        Bond length energies that have a z score higher than this are
        considered outliers. Defaults to ``3.0``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        The polynomial coefficients representing the minimum of the bond length
        curve. Coefficients need to be in decreasing order (i.e., fourth, 
        third, second, etc.).
    """
    idx_sort = np.argsort(bond_lengths)
    bond_lengths = bond_lengths[idx_sort]
    energies = energies[idx_sort]

    # Removes outliers using z score.
    if remove_outliers:
        bond_lengths, energies = _remove_dimer_outliers(
            bond_lengths, energies, zscore_cutoff=zscore_cutoff
        )

    e_min_idx = np.argmin(energies)
    slice_start = e_min_idx - n_points
    slice_end = e_min_idx + 1 + n_points
    if slice_start < 0: slice_start = 0
    bond_lengths_for_fit = bond_lengths[slice_start:slice_end]
    energies_for_fit = energies[slice_start:slice_end]

    poly_coeff = np.polyfit(bond_lengths_for_fit, energies_for_fit, poly_order)
    
    return bond_lengths_for_fit, poly_coeff

def _dimer_poly_pred(bond_length, poly_coeffs):
    """Energy prediction using a polynomial fitted to the minima of dimer
    bonding curves.

    Used to identifying the minimum energy of the polynomial fit.

    Parameters
    ----------
    bond_length : :obj:`float`
        The bond_length of the dimer (x value).
    poly_coeffs : :obj:`numpy.ndarray`
        The polynomial coefficients representing the minimum of the bond length
        curve. Coefficients need to be in decreasing order (i.e., fourth, 
        third, second, etc.).
    """
    return np.polyval(poly_coeffs, bond_length)

def find_poly_min(fit_bond_lengths, poly_coeffs):
    """Finds the minimum bond length and respective energy of a polynomial fit
    to a dimer bonding curve.

    Used to find dimer equilibrium properties.

    Parameters
    ----------
    fit_bond_lengths : :obj:`numpy.ndarray`
        The bond lengths used to fit the polynomial. Used as bounds for the
        minimizer.
    poly_coeffs : :obj:`numpy.ndarray`
        The polynomial coefficients representing the minimum of the bonding
        curve. Coefficients need to be in decreasing order (i.e., fourth, 
        third, second, etc.).
    
    Returns
    -------
    :obj:`float`
        Equilibrium bond length with respect to the fitted polynomial.
    :obj:`float`
        Electronic energy of the equilibrium bond length.
    """
    bond_length_bounds = (min(fit_bond_lengths), max(fit_bond_lengths))

    opt_data = optimize.minimize(
        _dimer_poly_pred, bond_length_bounds[0], args=(poly_coeffs),
        bounds=(bond_length_bounds,)
    )
    bond_length = opt_data.x[0]
    energy = opt_data.fun

    if type(energy) == np.ndarray:
        assert len(energy) == 1
        energy = energy[0]
    
    return bond_length, energy

def calc_spin(spin_squared):
    """Calculate the spin of the system given <S^2>.

    Solves the expression <S^2> = S(S + 1) for S.

    Parameters
    ----------
    spin_squared : :obj:`float`
        The observed value of the spin squared operator.
    
    Returns
    -------
    :obj:`float`
        System spin (can be used to calculate multiplicity).
    """
    r = np.roots(np.array([1, 1, -spin_squared]))
    r = r[r>0]
    assert len(r) == 1
    return r[0]

def alchemical_pes(
    df_qc, system_label, charge, excitation_level=0, basis_set='aug-cc-pV5Z',
    bond_length=None, energy_type='total', lambdas_center_neutral=False):
    """Lambda values and energies of the quantum alchemical PES.

    Can only be used for atoms or single bond lengths of dimers.
    
    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        Quantum chemistry dataframe.
    system_label : :obj:`str`
        Specifies one of the system in the quantum alchemical PES.
    charge : :obj:`int`
        Corresponding system charge to the `system_label`.
    excitation_level : :obj:`int`, optional
        Electronic state of the system with respect to the ground state. ``0``
        represents the ground state, ``1`` the first excited state, etc.
        Defaults to ``0``.
    basis_set : :obj:`str`, optional
        Desired basis sets the predictions are from. Defaults to
        ``aug-cc-pV5Z``.
    bond_length : :obj:`float`, optional
        Desired bond length for dimers; must be specified.
    energy_type : :obj:`str`, optional
        Species the energy type/contributions to examine. Can be ``'total'``
        energies, ``'hf'`` for Hartree-Fock contributions, or ``'correlation'``
        energies.
    lambdas_center_neutral : :obj:`bool`, optional
        Center the lambda values about the neutral species of the same number
        of electrons. For example, if the desired system is C- the lambda value
        of ``0`` and ``1`` would be C- and N if set to ``False``, respectively.
        If ``True``, then the lambda values for C- and N would be ``-1`` and
        ``0``, respectively. Defaults to ``False``.
    
    Returns
    -------
    :obj:`numpy.ndarray`
        Lambda values representing the nuclear charge perturbation.
    :obj:`numpy.ndarray`
        Energies of the system with respect to lambda values.
    """
    if energy_type == 'total':
        df_energy_type = 'electronic_energy'
    elif energy_type == 'hf':
        df_energy_type = 'hf_energy'
    elif energy_type == 'correlation':
        df_energy_type = 'correlation_energy'

    df_alch_pes = df_qc.query(
        'system == @system_label'
        '& charge == @charge'
        '& basis_set == @basis_set'
    )
    
    if len(df_qc.iloc[0]['atomic_numbers']) == 2:
        assert bond_length is not None
        df_alch_pes = df_alch_pes.query('bond_length == @bond_length')
    if len(set(df_alch_pes['multiplicity'].values)) > 1:
        sys_multiplicity = get_multiplicity(df_alch_pes, excitation_level)
        df_alch_pes = df_alch_pes.query('multiplicity == @sys_multiplicity')

    lambda_sort = np.argsort(df_alch_pes['lambda_value'].values)
    lambda_values = df_alch_pes['lambda_value'].values[lambda_sort]
    energies = df_alch_pes[df_energy_type].values[lambda_sort]
    
    if lambdas_center_neutral and charge != 0:
        lambda_values += charge

    return lambda_values, energies

def clean_state_labels(system_labels, system_charges):
    """Convert system labels and charges into clean labels like C$^{+}$.

    Typically used for figure axes and labels.

    Parameters
    ----------
    system_labels : :obj:`list` [:obj:`str`]
        Specifies the atoms in the system.
    system_charges : :obj:`list` [:obj:`int`]
        Total system charges.
    
    Returns
    -------
    :obj:`list`
        Clean state labels.
    """
    clean_labels = []
    for i in range(len(system_labels)):
        sys_label = ''.join([atom.capitalize() for atom in system_labels[i].split('.')])
        charge = system_charges[i]
        if charge > 0:
            if charge == 1:
                charge = '+'
            else:
                charge = str(charge) + '+'
        elif charge < 0:
            if charge == -1:
                charge = '-'
            else:
                charge = str(charge)[1] + '-'
        else:
            charge = ''
        clean_labels.append(sys_label + '$\,^{' + charge + '}$')
    return clean_labels