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
import pandas as pd

from qa_tools.utils import *
from qa_tools.data import *
from qa_tools.data import _json_parse_qc

json_path_atoms = './json-data/atom-pyscf.qa-data.posthf.json'
json_path_converged_atom = './tests/tests_data/n.chrg0.mult4-pyscf-uccsdt.augccpv5z.json'
json_path_not_converged_atom = './tests/tests_data/c.chrg-2.mult3-pyscf-uccsdt.augccpv5z.json'
json_path_converged_dimer = './tests/tests_data/c.h.chrg0.mult2.sep1.10-pyscf-uccsdt.ccpv5z.json'
json_path_not_converged_dimer = './tests/tests_data/b.h.chrg-1.mult2.sep1.60-pyscf-uccsdt.ccpv5z.json'


df_qc = qc_dframe(read_json(json_path_atoms), only_converged=False)

@pytest.mark.cbs
def test_prepare_atom_cbs_reduced():
    global df_qc_atom_cbs
    global df_qats_atom_cbs
    df_qc_nco = df_qc[df_qc.system.isin(['c', 'n', 'o'])]
    num_5z = len(df_qc_nco.query('basis_set == "aug-cc-pV5Z"'))
    num_tz = len(df_qc_nco.query('basis_set == "aug-cc-pVTZ"'))
    correct_total_size = 3*num_tz + num_5z
    df_qc_atom_cbs = add_cbs_extrap_qc_df(
        df_qc_nco, cbs_basis_key='aug', basis_set_lower='aug-cc-pVTZ',
        basis_set_higher='aug-cc-pVQZ'
    )
    assert len(df_qc_atom_cbs) == correct_total_size

def test_json_parse_qc_atom_converged():
    json_dict = read_json(json_path_converged_atom)

    system_label = 'n'
    df_rows_all = _json_parse_qc(
        system_label, json_dict, only_converged=False
    )
    df_rows_only_converged = _json_parse_qc(
        system_label, json_dict, only_converged=True
    )

    assert len(df_rows_all) == 21
    assert len(df_rows_all) == len(df_rows_only_converged)

    df = pd.DataFrame(df_rows_all)

    df_index_check = 3
    df_check_row = df.iloc[df_index_check]
    
    column_keys = (
        'system', 'atomic_numbers', 'charge', 'multiplicity', 'n_electrons',
        'qc_method', 'basis_set', 'converged', 'hf_energy', 'correlation_energy',
        'cc_spin_squared', 'scf_spin_squared', 'triples_correction', 
        'broken_sym', 'lambda_value', 'electronic_energy'
    )
    column_values = (
        'n', np.array([7]), 0, 4, 7,
        'UCCSD(T)', 'aug-cc-pV5Z', True, -34.0532826246006, -0.16331737526824242,
        3.750650031579879, 3.7535805578898604, -0.006736862917495266, 
        False, -1.25, -34.21659999986884
    )
    for key,value in zip(column_keys, column_values):
        try:
            assert np.array_equal(np.array(df_check_row[key]), value)
        except AssertionError:
            print(f'{key}; expected: {value}    parsed: {df_check_row[key]}')
            raise

def test_json_parse_qc_dimer_converged():
    json_dict = read_json(json_path_converged_dimer)

    system_label = 'c.h'
    df_rows_all = _json_parse_qc(
        system_label, json_dict, only_converged=False
    )
    df_rows_only_converged = _json_parse_qc(
        system_label, json_dict, only_converged=True
    )

    assert len(df_rows_all) == 17
    assert len(df_rows_all) == len(df_rows_only_converged)

    df = pd.DataFrame(df_rows_all)

    df_index_check = 3
    df_check_row = df.iloc[df_index_check]
    
    column_keys = (
        'system', 'atomic_numbers', 'charge', 'multiplicity', 'n_electrons',
        'qc_method', 'basis_set', 'converged', 'hf_energy', 'correlation_energy',
        'cc_spin_squared', 'scf_spin_squared', 'triples_correction', 
        'broken_sym', 'lambda_value', 'electronic_energy', 'bond_length'
    )
    column_values = (
        'c.h', np.array([6, 1]), 0, 2, 7,
        'UCCSD(T)', 'cc-pV5Z', True, -34.70631190364489, -0.16619257122525255,
        0.7502306466934399, 0.7604261277920434, -0.004267144753913249, 
        False, -0.25, -34.87250447487014, 1.1
    )
    for key,value in zip(column_keys, column_values):
        try:
            assert np.array_equal(np.array(df_check_row[key]), value)
        except AssertionError:
            print(f'{key}; expected: {value}    parsed: {df_check_row[key]}')
            raise

def test_json_parse_qc_dimer_not_converged():
    json_dict = read_json(json_path_converged_dimer)

    system_label = 'c.h'
    df_rows_all = _json_parse_qc(
        system_label, json_dict, only_converged=False
    )
    df_rows_only_converged = _json_parse_qc(
        system_label, json_dict, only_converged=True
    )

    assert len(df_rows_all) == 17
    assert len(df_rows_all) == len(df_rows_only_converged)

    df = pd.DataFrame(df_rows_all)

    df_index_check = 3
    df_check_row = df.iloc[df_index_check]
    
    column_keys = (
        'system', 'atomic_numbers', 'charge', 'multiplicity', 'n_electrons',
        'qc_method', 'basis_set', 'converged', 'hf_energy', 'correlation_energy',
        'cc_spin_squared', 'scf_spin_squared', 'triples_correction', 
        'broken_sym', 'lambda_value', 'electronic_energy', 'bond_length'
    )
    column_values = (
        'c.h', np.array([6, 1]), 0, 2, 7,
        'UCCSD(T)', 'cc-pV5Z', True, -34.70631190364489, -0.16619257122525255,
        0.7502306466934399, 0.7604261277920434, -0.004267144753913249, 
        False, -0.25, -34.87250447487014, 1.1
    )
    for key,value in zip(column_keys, column_values):
        try:
            assert np.array_equal(np.array(df_check_row[key]), value)
        except AssertionError:
            print(f'{key}; expected: {value}    parsed: {df_check_row[key]}')
            raise

def test_json_parse_qc_atom_not_converged():
    json_dict = read_json(json_path_not_converged_atom)

    system_label = 'c'
    df_rows_all = _json_parse_qc(
        system_label, json_dict, only_converged=False
    )
    df_rows_only_converged = _json_parse_qc(
        system_label, json_dict, only_converged=True
    )

    n_not_converged = 5
    assert len(df_rows_all) == 21
    assert len(df_rows_only_converged) == 21 - n_not_converged

    df = pd.DataFrame(df_rows_all)

    df_index_check = 3
    df_check_row = df.iloc[df_index_check]
    column_keys = (
        'system', 'atomic_numbers', 'charge', 'multiplicity', 'n_electrons',
        'qc_method', 'basis_set', 'converged', 'hf_energy', 'correlation_energy',
        'cc_spin_squared', 'scf_spin_squared', 'triples_correction', 
        'broken_sym', 'lambda_value', 'electronic_energy'
    )
    column_values = (
        'c', np.array([6]), -2, 3, 8,
        'UCCSD(T)', 'aug-cc-pV5Z', False, -37.57186094870421, -0.26504558574033155,
        2.4662682375578484, 2.0027221230670826, -0.028085487022331383, 
        False, 0.01, -37.836906534444545
    )
    for key,value in zip(column_keys, column_values):
        try:
            assert np.array_equal(np.array(df_check_row[key]), value)
        except AssertionError:
            print(f'{key}; expected: {value}    parsed: {df_check_row[key]}')
            raise

@pytest.mark.cbs
def test_df_cbs_n():
    df_qc_n = df_qc_atom_cbs.query(
        'system == "n" & basis_set != "aug-cc-pV5Z" & lambda_value == 0 & charge == 0 & multiplicity == 4'
    )
    assert len(df_qc_n) == 3

    e_elec_tz = -54.52753479376989
    e_hf_tz = -54.40116224025206
    e_corr_tz = e_elec_tz - e_hf_tz
    e_elec_qz = -54.55384812081688
    e_hf_qz = -54.40381952064283
    e_corr_qz = e_elec_qz - e_hf_qz

    assert df_qc_n.query('basis_set == "aug-cc-pVTZ"').iloc[0]['correlation_energy'] == e_corr_tz
    assert df_qc_n.query('basis_set == "aug-cc-pVQZ"').iloc[0]['correlation_energy'] == e_corr_qz
    
    e_corr_cbs_manual = extrapolate_correlation(
        (e_corr_tz, e_corr_qz),
        (basis_cardinals['aug-cc-pVTZ'], basis_cardinals['aug-cc-pVQZ']),
        cbs_extrap_betas['aug']['3/4']
    )
    assert df_qc_n.query('basis_set == "CBS-aug"').iloc[0]['correlation_energy'] == e_corr_cbs_manual

