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

from math import exp, sqrt
import numpy as np
import pandas as pd
from scipy import optimize
import findiff

from apdft_tools.utils import *

def _calc_distance(r1, r2):
    """Calculates the Euclidean distance between two points.

    Parameters
    ----------
    r1 : :obj:`numpy.ndarray`
        Cartesian coordinates of a point with shape ``(3,)``.
    r2 : :obj:`numpy.ndarray`
        Cartesian coordinates of a point with shape ``(3,)``.
    """
    return np.linalg.norm(r1 - r2)

def _json_parse_qc(system_label, json_calc, only_converged=False):
    """

    Parameters
    ----------
    system_label : :obj:`str`
        The string that defines the atoms and or molecules.
    json_calc : :obj:`dict`
        All data from a single calculation in a single dictionary.
    only_converged : :obj:`bool`, optional
        Only include data where calculations converged. Defaults to ``True``.
    
    Returns
    -------
    :obj:`list`
        A list of dictionaries for new pandas dataframe rows.
    """
    df_rows = []
    # If possible, check all calculations have converged
    if only_converged and 'cc_converged' in json_calc.keys():
        if not np.all(np.array(json_calc['cc_converged'])): return []
    # Getting row-dependent data.
    apdft_lambdas = json_calc['apdft_lambdas']
    electronic_energies = json_calc['electronic_energies']
    scf_converged = json_calc['scf_converged']

    # Keys that are not always included.
    if 'cc_converged' in json_calc.keys():
        cc_converged = json_calc['cc_converged']
    else:
        cc_converged = [None for _ in scf_converged]
    if 'hf_energies' in json_calc.keys():
        hf_energies = json_calc['hf_energies']
    else:
        hf_energies = [None for _ in electronic_energies]
    if 'triples_corrections' in json_calc.keys():
        triples_corrections = json_calc['triples_corrections']
    else:
        triples_corrections = [None for _ in electronic_energies]
    if 'scf_spin_squared' in json_calc.keys():
        scf_spin_squared = json_calc['scf_spin_squared']
    else:
        scf_spin_squared = [None for _ in electronic_energies]
    if 'cc_spin_squared' in json_calc.keys():
        cc_spin_squared = json_calc['cc_spin_squared']
    else:
        cc_spin_squared = [None for _ in electronic_energies]
    if 'broken_symmetry' in json_calc.keys():
        broken_sym = json_calc['broken_symmetry']
    else:
        broken_sym = None

    # Adds df row for every lambda.
    for i in range(len(apdft_lambdas)):
        df_dict = {'system': system_label}

        # Checks convergence stuff.
        if scf_converged[i] and (cc_converged[i] is None or cc_converged[i]):
            converged = True
        else:
            converged = False
        if only_converged and not converged:
            continue
        
        # Adds common information for df rows
        df_dict['atomic_numbers'] = np.array(json_calc['atomic_numbers'])
        df_dict['charge'] = json_calc['molecular_charge']
        df_dict['multiplicity'] = json_calc['molecular_multiplicity']
        df_dict['n_electrons'] = json_calc['n_electrons']
        df_dict['qc_method'] = json_calc['model']['method']
        df_dict['basis_set'] = json_calc['model']['basis']
        df_dict['converged'] = converged

        # Handles energy components for post-HF and DFT methods.
        if hf_energies[i] is not None:
            df_dict['hf_energy'] = hf_energies[i]
            try:
                df_dict['correlation_energy'] = electronic_energies[i] - hf_energies[i]
            except TypeError:
                df_dict['correlation_energy'] = np.nan
        else:
            df_dict['hf_energy'] = electronic_energies[i]
            df_dict['correlation_energy'] = None
        df_dict['cc_spin_squared'] = cc_spin_squared[i]
        df_dict['scf_spin_squared'] = scf_spin_squared[i]
        df_dict['triples_correction'] = triples_corrections[i]
        df_dict['broken_sym'] = broken_sym

        # Important ones go in front and back.
        df_dict['lambda_value'] = float(apdft_lambdas[i])
        df_dict['electronic_energy'] = electronic_energies[i]
        if len(df_dict['atomic_numbers']) == 2:
            geo = np.array(json_calc['molecule']['geometry'])
            df_dict['bond_length'] = _calc_distance(geo[0], geo[1])
        
        df_rows.append(df_dict)

    return df_rows

def _json_parse_apdft(system_label, json_calc):
    """

    Currently there are no checks that the calculations pertinent to finite
    differences have converged.

    Parameters
    ----------
    system_label : :obj:`str`
        The string that defines the atoms and or molecules.
    json_calc : :obj:`dict`
        All data from a single calculation in a single dictionary.
    
    Returns
    -------
    :obj:`list`
        A list of dictionaries for new pandas dataframe rows.
    """
    df_dict = {'system': system_label}
    df_dict['atomic_numbers'] = np.array(json_calc['atomic_numbers'])
    df_dict['charge'] = json_calc['molecular_charge']
    df_dict['multiplicity'] = json_calc['molecular_multiplicity']
    df_dict['n_electrons'] = json_calc['n_electrons']
    df_dict['qc_method'] = json_calc['model']['method']
    df_dict['basis_set'] = json_calc['model']['basis']
    apdft_lambdas = json_calc['apdft_lambdas']
    df_dict['lambda_range'] = (
        int(min(apdft_lambdas)), int(max(apdft_lambdas))
    )
    df_dict['finite_diff_delta'] = json_calc['finite_diff_delta']
    df_dict['finite_diff_acc'] = json_calc['finite_diff_acc']
    df_dict['poly_coeff'] = np.array(json_calc['apdft_poly_coeff'])
    if len(df_dict['atomic_numbers']) == 2:
        geo = np.array(json_calc['molecule']['geometry'])
        df_dict['bond_length'] = _calc_distance(geo[0], geo[1])

    return [df_dict]

def get_qc_dframe(json_dict, only_converged=False):
    """Prepares a Pandas dataframe of quantum chemistry data from a JSON file.

    Parameters
    ----------
    json_dict : :obj:`dict`
        A loaded JSON file containing data organized by system label
        (e.g., `'h'`, `'mg'`, etc.). Under each system label is the individual
        JSON dictionary of that state's calculation with the standard format of
        `atoms.chrg.mult-pyscf-qcmethod.basis`; for example,
        `'h.chrg-1.mult1-pyscf-ccsd.augccpvqz'`.
    only_converged : :obj:`bool`, optional
        Only include data where calculations converged. Defaults to ``True``.
    
    Returns
    -------
    :obj:`pandas.core.frame.DataFrame`
        A data frame with the following columns: system, atomic_numbers, charge,
        multiplicity, n_electrons, qc_method, basis_set, lambda,
        electronic_energy, hf_energy, and correlation_energy.
    """
    prelim_df = []

    # Loops through every system.
    for system_label in json_dict.keys():
        # Loops through every state calculation (of multiple lambdas).
        for state_label in json_dict[system_label].keys():
            for calc_label in json_dict[system_label][state_label].keys():
                if 'electronic_energies' in json_dict[system_label][state_label][calc_label].keys():
                    calc_data = json_dict[system_label][state_label][calc_label]
                    prelim_df.extend(
                        _json_parse_qc(
                            system_label, calc_data, only_converged=only_converged
                        )
                    )
                else:
                    for calc_label2 in json_dict[system_label][state_label][calc_label].keys():
                        calc_data = json_dict[system_label][state_label][calc_label][calc_label2]
                        prelim_df.extend(
                            _json_parse_qc(
                                system_label, calc_data, only_converged=only_converged
                            )
                        )

    return pd.DataFrame(prelim_df)

def hf_error(A, hf_energies, cardinals, alpha):
    """Evaluates the error of the HF extrapolation.

    0 = [A exp(-alpha * sqrt(Y)) - E_scf^(Y)] - [A exp(-alpha * sqrt(X)) - E_scf^(X)]

    Parameters
    ----------
    A : :obj:`float`
        A system dependent parameter (that will be fit).
    hf_energies : :obj:`tuple` (:obj:`float`)
        The small (X) to large (Y) basis set hf energies, respectively.
    alpha : :obj:`float`
        The basis-set dependent constant alpha.
    cardinals : :obj:`tuple`
        The X and Y cardinal numbers of the basis sets, respectively.
    
    Returns
    -------
    :obj:`float`
        Error in the hf extrapolation procedure in the `A` parameter.
    """
    hf_x, hf_y = hf_energies
    cardinal_x, cardinal_y = cardinals
    error_x = ((A * exp(-alpha * sqrt(cardinal_x))) - hf_x)
    error_y = ((A * exp(-alpha * sqrt(cardinal_y))) - hf_y)
    error = error_y - error_x
    return error

def extrapolate_hf(hf_energies, cardinals, alpha):
    """Extrapolates the HF energy from a post-HF energy.

    Based on E_scf^(X) = E_scf^(infinity) + A exp(-alpha * sqrt(X))
    where E_scf^(X) is the SCF energy of cardinal number X, E_scf^(infinity) is
    the CBS extrapolated energy, A and alpha are constants. Typically, alpha is
    a basis-set dependent constant and A needs to be fitted.

    Since we have two energies, we can minimize E_scf^(Y) - E_scf^(X) = 
    A exp(-alpha * sqrt(Y)) - A exp(-alpha * sqrt(X)).

    Parameters
    ----------
    hf_energies : :obj:`tuple` (:obj:`float`)
        The small (X) to large (Y) basis set hf energies, respectively.
    cardinals : :obj:`tuple`
        The X and Y cardinal numbers of the basis sets, respectively.
    alpha : :obj:`float`
        The basis-set dependent constant alpha.
    
    Returns
    -------
    :obj:`float`
        Extrapolated hf energy.
    """
    roots = optimize.fsolve(
        hf_error, 1, args=(hf_energies, cardinals, alpha)
    )
    A = roots[0]
    cbs_hf = hf_energies[0] - A*exp(-alpha * sqrt(cardinals[0]))
    return cbs_hf

def extrapolate_correlation(correlation_energies, cardinals, beta):
    """Extrapolates the correlation energy.

    For more information, see Equation 2 in DOI: 10.1021/ct100396y.

    Parameters
    ----------
    correlation_energies : :obj:`tuple` (:obj:`float`)
        The small (X) to large (Y) basis set correlation energies, respectively.
    cardinals : :obj:`tuple`
        The X and Y cardinal numbers of the basis sets, respectively.
    beta : :obj:`float`
        The basis-set dependent constant beta.
    
    Returns
    -------
    :obj:`float`
        Extrapolated correlation energy.
    """
    correlation_x, correlation_y = correlation_energies
    cardinal_x, cardinal_y = cardinals
    numerator = (cardinal_x**beta * correlation_x) - (cardinal_y**beta * correlation_y)
    denominator = cardinal_x**beta - cardinal_y**beta
    cbs_correlation = numerator / denominator
    return cbs_correlation

def get_qc_df_cbs(
    df_qc, cbs_basis_key='aug', basis_set_lower='aug-cc-pVTZ',
    basis_set_higher='aug-cc-pVQZ'
):
    """Extrapolates post-HF energies and adds CBS rows.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A dataframe with quantum chemistry calculation data.
    cbs_basis_key : :obj:`str`, optional
        Which basis set family to extrapolate. Must be keys of the extrapolation
        dictionaries.
    basis_set_lower : :obj:`str`, optional
        The smaller basis set (with a lower cardinal number).
    basis_set_higher : :obj:`str`, optional
        The larger basis set (with a higher cardinal number).

    Returns
    -------
    :obj:`pandas.DataFrame`
        QC dataframe with CBS extrapolated data added.
    """
    lower_cardinal = basis_cardinals[basis_set_lower]
    upper_cardinal = basis_cardinals[basis_set_higher]
    cbs_cardinal_key = f'{lower_cardinal}/{upper_cardinal}'

    # Only extrapolate post-HF methods.
    lower_df = df_qc[
        (df_qc.basis_set == basis_set_lower)
        & (df_qc.qc_method.transform(lambda x: x.lower()).isin(post_hf_methods))
    ]
    df_cbs_prelim = []
    for row_info in zip(
        lower_df['system'], lower_df['charge'], lower_df['multiplicity'],
        lower_df['qc_method'], lower_df['lambda_value'],
        lower_df['hf_energy'], lower_df['correlation_energy'], 
        lower_df['triples_correction']
    ):
        # Calculation with lower basis set.
        system, charge, multiplicity, qc_method, lambda_value, \
        hf_lower, correlation_lower, triples_lower = row_info

        # Calculation with higher basis set.
        calc_upper = df_qc.query(
            'system == @system' \
            '& charge == @charge' \
            '& multiplicity == @multiplicity' \
            '& qc_method == @qc_method' \
            '& basis_set == @basis_set_higher' \
            '& lambda_value == @lambda_value' \
        )
        if len(calc_upper) == 0:
            # Assumes that there is a missing, possibly unconverged, calculation.
            continue
        else:
            assert len(calc_upper) == 1
        hf_upper = calc_upper.iloc[0]['hf_energy']
        correlation_upper = calc_upper.iloc[0]['correlation_energy']
        triples_upper = calc_upper.iloc[0]['triples_correction']

        # CBS extrapolation
        cbs_hf = extrapolate_hf(
            (hf_lower, hf_upper),
            (lower_cardinal, upper_cardinal),
            cbs_extrap_alphas[cbs_basis_key][cbs_cardinal_key]
        )
        if np.isnan(correlation_lower) or np.isnan(correlation_upper):
            cbs_correlation = np.nan
            cbs_total = cbs_hf
        else:
            cbs_correlation = extrapolate_correlation(
                (correlation_lower, correlation_upper),
                (lower_cardinal, upper_cardinal),
                cbs_extrap_betas[cbs_basis_key][cbs_cardinal_key]
            )
            cbs_total = cbs_hf + cbs_correlation
        
        # Building CBS row.
        cbs_calc = calc_upper.iloc[0].to_dict()
        cbs_calc['basis_set'] = f'CBS-{cbs_basis_key}'
        cbs_calc['electronic_energy'] = cbs_total
        cbs_calc['hf_energy'] = cbs_hf
        cbs_calc['correlation_energy'] = cbs_correlation

        # Triples CBS extrapolation.
        if not np.isnan(triples_lower) or not np.isnan(triples_upper):
            cbs_triples = extrapolate_correlation(
                (triples_lower, triples_upper),
                (lower_cardinal, upper_cardinal),
                cbs_extrap_betas[cbs_basis_key][cbs_cardinal_key]
            )
            cbs_calc['triples_correction'] = cbs_triples
        df_cbs_prelim.append(cbs_calc)
    
    df_cbs_prelim = pd.DataFrame(df_cbs_prelim)
    df_cbs = df_qc.append(df_cbs_prelim)
    return df_cbs

def get_apdft_dframe(json_dict):
    """Prepares a Pandas dataframe of APDFT-relevant data.

    Parameters
    ----------
    json_dict : :obj:`dict`
        A loaded JSON file containing data organized by system label
        (e.g., `'h'`, `'mg'`, etc.). Under each system label is the state_label
        that specifies charge and multiplicity (e.g., `'c.charg0.mult3'`).
        Nested under that are individual JSON dictionaries of that state's
        calculation with the standard format of
        `atoms.chrg.mult-pyscf-qcmethod.basis`; for example,
        `'h.chrg-1.mult1-pyscf-ccsd.augccpvqz'`.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
        A dataframe with the following columns: system, atomic_numbers, charge,
        multiplicity, n_electrons, qc_method, basis_set, lambda_range,
        finite_diff_delta, finite_diff_acc, poly_coeff.
    """
    prelim_df = []

    # Loops through every system.
    for system_label in json_dict.keys():
        # Loops through every state.
        for state_label in json_dict[system_label].keys():
            # Loops through every calculation.
            for calc_label in json_dict[system_label][state_label].keys():
                if 'apdft_poly_coeff' in json_dict[system_label][state_label][calc_label].keys():
                    calc_data = json_dict[system_label][state_label][calc_label]
                    prelim_df.extend(_json_parse_apdft(system_label, calc_data))
                else:
                    for calc_label2 in json_dict[system_label][state_label][calc_label].keys():
                        calc_data = json_dict[system_label][state_label][calc_label][calc_label2]
                        prelim_df.extend(_json_parse_apdft(system_label, calc_data))
                
    return pd.DataFrame(prelim_df)

def get_apdft_df_cbs(
    df_qc, df_apdft, cbs_basis_key='aug', basis_set_higher='aug-cc-pVQZ',
    max_apdft_order=4, finite_diff_delta=0.01,
    finite_diff_acc=2,
):
    """Adds APDFT rows from CBS extrapolated data to a dataframe.

    Parameters
    ----------
    df_qc : :obj:`pandas.DataFrame`
        A dataframe with quantum chemistry data.
    df_apdft : :obj:`pandas.DataFrame`
        A dataframe with APDFT data.
    cbs_basis_key : :obj:`str`, optional
        Which basis set family to extrapolate. Must be keys of the extrapolation
        dictionaries.
    basis_set_higher : :obj:`str`, optional
        The larger basis set (with a higher cardinal number).
    max_apdft_order : :obj:`int`, optional
        The maximum APDFT order desired. Defaults to four.
    finite_diff_delta : :obj:`float`
        The deviation from x0 for each point.
    finite_diff_acc : :obj:`int`, optional
        The overall "accuracy" of the finite difference method. Only even
        integers up to six are allowed.

    Returns
    -------
    :obj:`pandas.DataFrame`
        APDFT dataframe with added CBS extrapolated data.
    """
    stencils = [{'coefficients': np.array([1]), "offsets": np.array([0])}]  # 0th order approximation.
    for order in range(1, max_apdft_order+1):
        stencils.append(
            findiff.coefficients(
                deriv=order, acc=finite_diff_acc
            )['center']
        )
    # Gets all the unique offsets for all APDFT orders in `positions`.
    positions = list(set().union(*[set(_["offsets"]) for _ in stencils]))
    required_lambdas = [_*finite_diff_delta for _ in positions]

    df_qc_for_apdft_cbs = df_qc[
        (df_qc.basis_set == f'CBS-{cbs_basis_key}')
        & (df_qc['lambda_value'].isin(required_lambdas))
        & (df_qc.qc_method.transform(lambda x: x.lower()).isin(post_hf_methods))
    ]
    df_apdft_for_cbs = df_apdft[
        (df_apdft.basis_set == basis_set_higher)
        & (df_apdft.qc_method.transform(lambda x: x.lower()).isin(post_hf_methods))
    ]

    df_cbs_prelim = []
    for row_info in zip(
        df_apdft_for_cbs['system'], df_apdft_for_cbs['charge'],
        df_apdft_for_cbs['multiplicity'],
        df_apdft_for_cbs['qc_method'], df_apdft_for_cbs['lambda_range'],
    ):
        system, charge, multiplicity, qc_method, lambda_range = row_info

        # Gets energies for finite difference.
        df_findiff = df_qc_for_apdft_cbs.query(
            'system == @system' \
            '& charge == @charge' \
            '& multiplicity == @multiplicity' \
            '& qc_method == @qc_method' \
        )
        if len(df_findiff) == 0:
            # Assumes that there is a missing, possibly unconverged, calculation.
            continue
        else:
            try:
                assert np.array_equal(
                    np.sort(df_findiff.lambda_value.values),
                    np.sort(required_lambdas)
                )
            except AssertionError:
                print('Missing or incorrect APDFT calculations for:')
                print(df_findiff, '\n')
                continue
        lambdas_fd = df_findiff.lambda_value.values.tolist()
        energies = df_findiff.electronic_energy.values.tolist()

        # Computes polynomial coefficients.
        poly_coeffs = []
        for order, stencil in enumerate(stencils):
            contribution = 0
            for o, c in zip(stencil["offsets"], stencil["coefficients"]):
                contribution += energies[lambdas_fd.index(o*finite_diff_delta)] * c
            contribution /= finite_diff_delta ** order
            contribution /= np.math.factorial(order)
            poly_coeffs.append(contribution)
        
        # Gets df_apdft row template.
        df_apdft_template = df_apdft_for_cbs.query(
            'system == @system' \
            '& charge == @charge' \
            '& multiplicity == @multiplicity' \
            '& qc_method == @qc_method' \
        ).iloc[0].to_dict()
        
        # Makes changes to CBS.
        df_apdft_template['basis_set'] = f'CBS-{cbs_basis_key}'
        df_apdft_template['finite_diff_delta'] = finite_diff_delta
        df_apdft_template['finite_diff_acc'] = finite_diff_acc
        df_apdft_template['poly_coeff'] = np.array(poly_coeffs)
        df_cbs_prelim.append(df_apdft_template)
    
    df_cbs_prelim = pd.DataFrame(df_cbs_prelim)
    df_cbs = df_apdft.append(df_cbs_prelim)
    return df_cbs

def prepare_dfs(json_path, get_CBS=False, only_converged=False):
    """Driver function for getting all relevant dataframes.

    Parameters
    ----------
    json_path : :obj:`str`
        Path to JSON file.
    get_CBS : :obj:`bool`
        Perform complete basis set extrapolations on both dataframes.
    
    Returns
    -------
    :obj:`pandas.DataFrame`
        The quantum chemistry dataframe.
    :obj:`pandas.DataFrame`
        The APDFT dataframe.
    """
    data_dict = read_json(json_path)
    df_qc = get_qc_dframe(data_dict, only_converged=only_converged)
    df_apdft = get_apdft_dframe(data_dict)
    if get_CBS:
        df_qc = get_qc_df_cbs(
            df_qc, cbs_basis_key='aug', basis_set_lower='aug-cc-pVTZ',
            basis_set_higher='aug-cc-pVQZ'
        )
        df_apdft = get_apdft_df_cbs(
            df_qc, df_apdft, cbs_basis_key='aug',
            basis_set_higher='aug-cc-pVQZ'
        )
    return df_qc, df_apdft
