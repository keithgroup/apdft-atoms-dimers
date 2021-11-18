#!/usr/bin/env python3

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

import os
import re
import numpy as np

###   Variables to Change  ###

overwrite = True

# Dictionary of all calculations desired for the QATS study.
# Organized by system then state where we then define all information needed for
# the PySCF and QATS script.
basis_set = 'cc-pV5Z'
finite_diff_accuracy = '2'
finite_diff_delta = '0.01'

scf_conv_tol = 1e-09  # Default: 1e-9
scf_conv_tol_grad = 1e-06  # Default: 3.162e-6
cc_conv_tol = 1e-07  # Default: 1e-7
cc_conv_tol_normt = 1e-05  # Default: 1e-5

scf_diis_space = 16  # Default: 8
cc_diis_space = 12  # Default: 6
try_diff_guess_if_fail = '1e'  # Tries a different initial guess algorithm. Options are: 1e, atom, huckel. None if not desired.
try_soscf_if_fail = True  # Tries to converge using the SOSCF method.
try_small_basis_if_fail = True  # Preliminary calculation with a smaller basis set.
smaller_basis = 'cc-pVDZ'
scf_conv_tol_prelim = 1e-06  # Default: 1e-9
scf_conv_tol_grad_prelim = 1e-03  # Default: 3.162e-6
diis_damp_prelim = 0.5

# Job properties
nodes = 1
days = 3
hours = 0
cluster = 'smp'
cores = 6

calc_dir = f'./prepared-calcs'

qa_calcs_all = {
    'Li.H': (
        {'state': 'chrg-1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'Be.H': (
        {'state': 'chrg-1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'B.H': (
        {'state': 'chrg-1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'C.H': (
        {'state': 'chrg2.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'N.H': (
        {'state': 'chrg2.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 3)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'O.H': (
        {'state': 'chrg2.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg-1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(0, 2)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'F.H': (
        {'state': 'chrg2.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg0.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-1, 1)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
    'Ne.H': (
        {'state': 'chrg2.mult2', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg2.mult4', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-3, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult1', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
        {'state': 'chrg1.mult3', 'qc_method': 'CCSD(T)', 'try_easier_state_if_fail': 'None', 'basis_set': basis_set, 'lambda_limits': '(-2, 0)', 'dimer_sep_range': '(0.6, 1.9)', 'dimer_sep_step': '0.1', 'specific_atom_lambda': '0', 'broken_symmetry': 'False', 'force_unrestrict_spin': 'False', 'max_qats_order': '4', 'lambda_step': '0.25', 'finite_diff_accuracy': finite_diff_accuracy, 'finite_diff_delta': finite_diff_delta},
    ),
}



###   Global Variables    ###

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





###   Functions   ###

# Ensures calc dir ends in /.
if calc_dir[-1] != '/':
    calc_dir += '/'


def prepare_pyscf_qa_calc(
    system_name, job_name, atoms, charge, multiplicity, calc_dict, save_dir=calc_dir,
    dimer_sep=None
):
    """Creates alchemy calculation for a single lambda.
    """
    if save_dir[-1] != '/':
        save_dir += '/'
    job_dir = save_dir + system_name + '/' + job_name + '/'
    os.makedirs(job_dir, exist_ok=overwrite)

    atoms_string = "['" + "', '".join(atoms) + "']"


    # Write scripts.
    pitt_crc_orca_submit = (
f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cores}
#SBATCH --time={days}-{hours}:00:00
#SBATCH --cluster={cluster}

module purge

cd $SLURM_SUBMIT_DIR
/ihome/crc/install/python/miniconda3-3.7/bin/python qa-pyscf-calc.py

crc-job-stats.py
"""
)
    with open(job_dir + 'submit-pyscf-qa.slurm', 'w') as f:
        f.write(pitt_crc_orca_submit)
    with open(job_dir + 'submit-pyscf-qa.slurm', 'rb') as open_file:
        content = open_file.read()
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
    with open(job_dir + 'submit-pyscf-qa.slurm', 'wb') as open_file:
        open_file.write(content)
    
    # Just need to have {{}} for cases were we are not using f-strings here.
    pyscf_qa_script = (
f"""
import os
import pathlib
import re
import numpy as np
import json
import pyscf
import pyscf.gto
import pyscf.qmmm
from pyscf import scf
import pyscf.dft
import pyscf.lib
import pyscf.mp
from pyscf import dftd3
from pyscf import __version__ as pyscf_version
import findiff
import matplotlib.pyplot as plt

try:
    import cclib
    has_cclib = True
except:
    has_cclib = False

try:
    from mpi4pyscf import scf
except:
    pass

# Specifying system.
atoms = {atoms_string}
dimer_sep = {dimer_sep}  # `None` if a single atom, float if dimer (in Angstroms).
charge = {charge}
multiplicity = {multiplicity}
scf_conv_tol = {scf_conv_tol}  # Default: 1e-9
scf_conv_tol_grad = {scf_conv_tol_grad}  # Default: 3.162e-6
cc_conv_tol = {cc_conv_tol}  # Default: 1e-7
cc_conv_tol_normt = {cc_conv_tol_normt}  # Default: 1e-5
force_unrestrict_spin = {calc_dict['force_unrestrict_spin']}
broken_symmetry = {calc_dict['broken_symmetry']}

scf_diis_space = {scf_diis_space}  # Default: 8
cc_diis_space = {cc_diis_space}  # Default: 6
try_soscf_if_fail = {try_soscf_if_fail}  # Tries to converge using the SOSCF method.
try_diff_guess_if_fail = '{try_diff_guess_if_fail}'  # Tries a different initial guess algorithm. Options are: 1e, atom, huckel. None if not desired.
try_easier_state_if_fail = {calc_dict['try_easier_state_if_fail']}  # Preliminary calc in ground state. (chrg, mult) or None. Same basis set
try_small_basis_if_fail = {try_small_basis_if_fail}  # Preliminary calculation with a smaller basis set.
smaller_basis = '{smaller_basis}'
scf_conv_tol_prelim = {scf_conv_tol_prelim}  # Default: 1e-9
scf_conv_tol_grad_prelim = {scf_conv_tol_grad_prelim}  # Default: 3.162e-6
diis_damp_prelim = {diis_damp_prelim}

# Specifying alchemical parts.
max_qats_order = {calc_dict['max_qats_order']}
lambda_step = {calc_dict['lambda_step']}
lambda_limits = {calc_dict['lambda_limits']}
specific_atom_lambda = {calc_dict['specific_atom_lambda']}

finite_diff_accuracy = {calc_dict['finite_diff_accuracy']}  # 2, 4, 6
finite_diff_delta = {calc_dict['finite_diff_delta']}

# Specifying calculation.
qc_method = '{calc_dict['qc_method']}'
basis_set = '{calc_dict['basis_set']}'
n_cores = {cores}  # Number of cores used on SMP (used for max memory).





#####     NO CHANGES NEEDED BELOW     #####


# Setting up paths.
work_dir = str(pathlib.Path(__file__).parent.resolve())
if work_dir[-1] == '/': work_dir = work_dir[:-1]  # Ensures path does NOT end in '/'

_element_to_z = {{
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
}}

def atoms_by_number(atom_list):
    return [_element_to_z[i] for i in atom_list]

qc_method_lower = qc_method.lower()
atomic_numbers = atoms_by_number(atoms)
if multiplicity > 1 or broken_symmetry or force_unrestrict_spin:
    qc_method_label = 'U' + qc_method
else:
    qc_method_label = qc_method

# Plot colors
ref_color = 'dimgrey'

def get_sys_label(atoms, charge, multiplicity, dimer_sep=None):
    atoms_label = '.'.join([i.lower() for i in atoms])
    if len(atoms) == 2:
        assert dimer_sep is not None
        return f'{{atoms_label}}.chrg{{charge}}.mult{{multiplicity}}.sep{{dimer_sep:.2f}}'
    elif len(atoms) == 1:
        return f'{{atoms_label}}.chrg{{charge}}.mult{{multiplicity}}'

def get_calc_label(sys_label, qc_method, basis_set):
    pattern = re.compile('[\W_]+')
    qc_method_label = pattern.sub('', qc_method.lower())
    basis_set_label = pattern.sub('', basis_set.lower())
    return f'{{sys_label}}-pyscf-{{qc_method_label}}.{{basis_set_label}}'

# Prepares molecular structures.
mol = pyscf.gto.Mole()

## Atomic coordinates in units of Angstrom.
mol.unit = 'angstrom'  # Insures Angstroms are enforced.
coords = np.array([np.zeros(3)])
mol.atom = [[atoms[0], coords[0]]]
if len(atoms) == 2:
    assert dimer_sep is not None
    coords = np.vstack((coords, np.array([dimer_sep, 0, 0])))
    mol.atom.extend([[atoms[1], coords[1]]])
elif len(atoms) == 1:
    assert dimer_sep is None

mol.charge = charge
mol.multiplicity = multiplicity
if basis_set == 'aug-cc-pV5Z':
    # Was not available for all elements of interest.
    import urllib
    from bs4 import BeautifulSoup
    bsse_url = 'https://www.basissetexchange.org/api/basis'
    augcc_url = f'{{bsse_url}}/aug-cc-pv5z/format/nwchem/?version=1&elements='
    augcc_url += ','.join([str(z) for z in atomic_numbers])
    soup = BeautifulSoup(urllib.request.urlopen(augcc_url), 'html.parser')
    basis_set_string = soup.get_text()
    mol.basis = pyscf.gto.basis.parse(basis_set_string)
else:
    mol.basis = basis_set

# Handles cores and memory.
os.environ['OMP_NUM_THREADS'] = f'{{n_cores}}'  # For CRC jobs only.
mol.max_memory = 8000 * n_cores

# Handles calculation logging.
mol.verbose = 4  # Controls print level; 4 is recommended.
system_label = get_sys_label(atoms, charge, multiplicity, dimer_sep=dimer_sep)
calc_label = get_calc_label(system_label, qc_method_label, basis_set)
mol.output = f'{{work_dir}}/{{calc_label}}.log'  # Sets the output file.

# Builds the molecular structure. (Must go after everything else).
mol.build()

def get_lambda_atom_values(atomic_numbers, lambda_value, specific_atom=specific_atom_lambda):
    if len(atomic_numbers) == 1:
        return np.array([lambda_value])
    elif len(atomic_numbers) == 2:
        if specific_atom is not None:
            lambda_values = np.zeros(len(atomic_numbers))
            lambda_values[specific_atom] = lambda_value
        else:
            if atomic_numbers[0] == atomic_numbers[1]:
                lambda_values = np.array([-lambda_value, lambda_value])
            else:
                i_max = np.argmax(atomic_numbers)
                i_min = np.argmin(atomic_numbers)
                lambda_values = np.zeros((2,))
                lambda_values[i_max] = lambda_value
                lambda_values[i_min] = -lambda_value
    else:
        raise ValueError(f'Only one or two atoms are supported. There are {{len(atomic_numbers)}} in this system.')
    return lambda_values


def set_Z(calc, mol, deltaZ):
    mf = pyscf.qmmm.add_mm_charges(calc, mol.atom_coords(unit='angstrom'), deltaZ)

    # AM: Not quite sure what this does?
    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q += deltaZ
        return mol.energy_nuc(q)
    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf

def _get_hf_calc(l_val):
    if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
        calc = scf.UHF(mol)
    else:
        calc = scf.RHF(mol)
    
    calc.chkfile = None
    calc = set_Z(calc, mol, l_val)

    calc.max_cycle = 200  # Default: 50
    calc.conv_tol = scf_conv_tol
    calc.conv_tol_grad = scf_conv_tol_grad
    calc.diis_space = scf_diis_space

def get_energy(
    qc_method, l_val, broken_symmetry=broken_symmetry,
    force_unrestrict_spin=force_unrestrict_spin
):
    calc_results = dict()

    global multiplicity

    if isinstance(l_val, float) or isinstance(l_val, int):
        l_val = np.array(l_val)
    
    if qc_method in ['hf', 'ccsd', 'ccsd(t)']:
        if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
            calc = scf.UHF(mol)
        else:
            calc = scf.RHF(mol)
        
        calc.chkfile = None
        calc = set_Z(calc, mol, l_val)

        calc.max_cycle = 200  # Default: 50
        calc.conv_tol = scf_conv_tol
        calc.conv_tol_grad = scf_conv_tol_grad
        calc.diis_space = scf_diis_space
        
        if broken_symmetry:
            calc.kernel(init_guess_breaksym=broken_symmetry)
        else:
            calc.kernel()
        
        # Attempts to converge SCF.
        if not calc.converged and try_soscf_if_fail:
            calc.newton()
            if broken_symmetry:
                calc.kernel(init_guess_breaksym=broken_symmetry)
            else:
                calc.kernel()
        
        if not calc.converged and try_diff_guess_if_fail is not None:
            # Ensures we are using DIIS.
            calc.diis = scf.CDIIS
            calc.diis_space = scf_diis_space
            calc.init_guess = try_diff_guess_if_fail
            if broken_symmetry:
                calc.kernel(init_guess_breaksym=broken_symmetry)
            else:
                calc.kernel()
            calc.init_guess = 'minao'  # Set back to default.
        
        if not calc.converged and try_small_basis_if_fail:
            import copy
            mol_smaller = copy.copy(mol)
            mol_smaller.basis = smaller_basis
            mol_smaller.build()

            if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
                calc_prelim = scf.UHF(mol_smaller)
            else:
                calc_prelim = scf.RHF(mol_smaller)
            calc_prelim = set_Z(calc_prelim, mol_smaller, l_val)
            calc_prelim.max_cycle = 200  # Default: 50
            calc_prelim.conv_tol = scf_conv_tol_prelim
            calc_prelim.conv_tol_grad = scf_conv_tol_grad_prelim
            calc_prelim.diis_space = scf_diis_space
            calc_prelim.chkfile = './prelim-calc.chk'
            calc_prelim.damp = diis_damp_prelim
            if broken_symmetry:
                calc_prelim.kernel(init_guess_breaksym=broken_symmetry)
            else:
                calc_prelim.kernel()
            if calc_prelim.converged:
                get_dm = scf.hf.SCF(mol)
                get_dm = set_Z(get_dm, mol, l_val)
                get_dm.conv_tol = scf_conv_tol
                get_dm.conv_tol_grad = scf_conv_tol_grad
                get_dm.diis_space = scf_diis_space
                get_dm.max_cycle = 200  # Default: 50
                dm = get_dm.from_chk('./prelim-calc.chk')

                # Creates new calculation.
                if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
                    calc = scf.UHF(mol)
                else:
                    calc = scf.RHF(mol)
                calc.chkfile = None
                calc = set_Z(calc, mol, l_val)
                calc.max_cycle = 200  # Default: 50
                calc.conv_tol = scf_conv_tol
                calc.conv_tol_grad = scf_conv_tol_grad
                calc.diis_space = scf_diis_space
                calc.damp = diis_damp_prelim
                calc.kernel(dm)
            
            if not calc.converged and try_easier_state_if_fail is not None:
                import copy
                mol_easier = copy.copy(mol)
                mol_easier.charge = try_easier_state_if_fail[0]
                mol_easier.multiplicity = try_easier_state_if_fail[1]
                mol_easier.build()

                if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
                    calc_prelim = scf.UHF(mol_easier)
                else:
                    calc_prelim = scf.RHF(mol_easier)
                calc_prelim = set_Z(calc_prelim, mol_easier, l_val)
                calc_prelim.max_cycle = 200  # Default: 50
                calc_prelim.conv_tol = scf_conv_tol_prelim
                calc_prelim.conv_tol_grad = scf_conv_tol_grad_prelim
                calc_prelim.diis_space = scf_diis_space
                calc_prelim.chkfile = './prelim-calc.chk'
                calc_prelim.damp = diis_damp_prelim
                if broken_symmetry:
                    calc_prelim.kernel(init_guess_breaksym=broken_symmetry)
                else:
                    calc_prelim.kernel()
                if calc_prelim.converged:
                    # Creates new calculation.
                    if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
                        calc = scf.UHF(mol)
                    else:
                        calc = scf.RHF(mol)
                    calc.chkfile = None
                    calc = set_Z(calc, mol, l_val)
                    calc.max_cycle = 200  # Default: 50
                    calc.conv_tol = scf_conv_tol
                    calc.conv_tol_grad = scf_conv_tol_grad
                    calc.diis_space = scf_diis_space
                    calc.damp = diis_damp_prelim
                    calc.chkfile = './prelim-calc.chk'
                    calc.init_guess = 'chkfile'
                    calc.kernel()
        
        calc_results['hf_energy'] = calc.e_tot
        calc_results['scf_converged'] = calc.converged

        calc.analyze(with_meta_lowdin=True)  # SCF analysis
        
        # Selects the appropriate method and final energies.
        if qc_method == 'hf':
            calc_results['correlation_energy'] = 0.0
        elif qc_method in ['ccsd', 'ccsd(t)']:
            cc_calc = pyscf.cc.CCSD(calc)

            cc_calc.max_cycle = 200
            cc_calc.conv_tol = cc_conv_tol
            cc_calc.conv_tol_normt = cc_conv_tol_normt
            cc_calc.diis_space = cc_diis_space

            # Runs calculation.
            try:
                cc_calc.kernel()
            except np.linalg.LinAlgError:
                calc_results['cc_converged'] = False
            except:
                calc_results['cc_converged'] = False
            else:
                calc_results['cc_converged'] = cc_calc.converged
            
            try:
                calc_results['correlation_energy'] = cc_calc.e_corr
            except:
                pass
            
            if multiplicity == 1 and not broken_symmetry and not force_unrestrict_spin:
                calc_results['t1_diagnostic'] = cc_calc.get_t1_diagnostic()

            if qc_method == 'ccsd(t)':
                if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
                    e_triples = cc_calc.uccsd_t()
                else:
                    e_triples = cc_calc.ccsd_t()
                
                calc_results['triples_correction'] = e_triples
                calc_results['correlation_energy'] += e_triples
        
        calc_results['electronic_energy'] = np.nansum(
            np.array(
                [calc_results['hf_energy'], calc_results['correlation_energy']]
            )
        )
    else:
        # Assume DFT
        if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
            calc = pyscf.dft.UKS(mol)
        else:
            calc = pyscf.dft.RKS(mol)

        calc.grids.level = 5
        calc = set_Z(calc, mol, l_val)
        if qc_method == 'pbe':
            calc.xc = 'pbe,pbe'
        elif qc_method == 'pbe0':
            calc.xc = 'pbe0,pbe0'
        
        calc.max_cycle = 300
        calc.conv_tol = scf_conv_tol
        calc.conv_tol_grad = scf_conv_tol_grad
        
        if broken_symmetry:
            raise ValueError('Broken symmetry is not allowed for DFT.')
        else:
            e_elec = calc.kernel()

        calc_results['scf_converged'] = calc.converged
        calc.analyze(with_meta_lowdin=True)  # SCF analysis

        # Gets dispersion corrections
        d3_calc = dftd3.dftd3(calc)
        calc_results['electronic_energy'] = d3_calc.kernel()
        calc_results['dispersion_energy'] = calc_results['electronic_energy'] - e_elec
    
    # Post calculation analysis.
    if multiplicity != 1 or broken_symmetry or force_unrestrict_spin:
        calc_results['scf_spin_squared'], _ = calc.spin_square()
        if qc_method in ['ccsd', 'ccsd(t)']:
            if calc_results['cc_converged']:
                calc_results['cc_spin_squared'], _ = cc_calc.spin_square()
            else:
                calc_results['cc_spin_squared'] = np.nan

    return calc_results

def get_qc_data(qc_method, lambda_atom_values):
    qc_method = qc_method.lower()
    # Initializes everything.
    data = dict()
    total_energies = []
    if qc_method in ['hf', 'ccsd', 'ccsd(t)']:
        hf_energies = []
    else:
        dispersion_energies = []
    if qc_method == 'ccsd(t)':
        triples_corrections = []
    scf_converged = []
    cc_converged = []
    scf_spin_squared = []
    cc_spin_squared = []
    t1_diagnostic = []
    
    # Calculates everything.
    for val in lambda_atom_values:
        calc_results = get_energy(qc_method, val)
        total_energies.append(calc_results['electronic_energy'])
        if qc_method in ['hf', 'ccsd', 'ccsd(t)']:
            hf_energies.append(calc_results['hf_energy'])
        else:
            dispersion_energies.append(calc_results['dispersion_energy'])
        
        if 'scf_converged' in calc_results.keys():
            scf_converged.append(calc_results['scf_converged'])
        
        if 'cc_converged' in calc_results.keys():
            cc_converged.append(calc_results['cc_converged'])

        if 'triples_correction' in calc_results.keys():
            triples_corrections.append(calc_results['triples_correction'])

        if 'scf_spin_squared' in calc_results.keys():
            scf_spin_squared.append(calc_results['scf_spin_squared'])
        if 'cc_spin_squared' in calc_results.keys():
            cc_spin_squared.append(calc_results['cc_spin_squared'])
        
        if 't1_diagnostic' in calc_results.keys():
            t1_diagnostic.append(calc_results['t1_diagnostic'])
    
    # Stores everything.
    data['total_energies'] = np.array(total_energies)
    if qc_method in ['hf', 'ccsd', 'ccsd(t)']:
        data['hf_energies'] = np.array(hf_energies)
        if qc_method in ['ccsd', 'ccsd(t)']:
            if qc_method == 'ccsd(t)':
                data['triples_corrections'] = np.array(triples_corrections)
    else:
        data['dispersion_energies'] = np.array(dispersion_energies)
    
    if len(scf_converged) != 0:
        data['scf_converged'] = np.array(scf_converged)
    if len(cc_converged) != 0:
        data['cc_converged'] = np.array(cc_converged)
    if len(scf_spin_squared) != 0:
        data['scf_spin_squared'] = np.array(scf_spin_squared)
    if len(cc_spin_squared) != 0:
        data['cc_spin_squared'] = np.array(cc_spin_squared)
    if len(t1_diagnostic) != 0:
        data['t1_diagnostic'] = np.array(t1_diagnostic)
        
    return data

# Calculates electronic energies at all lambdas.
lambda_values = np.arange(min(lambda_limits), max(lambda_limits) + lambda_step, lambda_step)
lambda_atom_values = np.array([get_lambda_atom_values(atomic_numbers, l_val) for l_val in lambda_values])
lambda_qc_data = get_qc_data(qc_method_lower, lambda_atom_values)

# Prepares to do QATS finite differences.
# Gets a stencil (framework) for computing finite differences of different orders.
def get_stencil():
    stencil = [{{'coefficients': np.array([1]), "offsets": np.array([0])}}]  # 0th order approximation.
    for order in range(1, max_qats_order+1):
        stencil.append(findiff.coefficients(deriv=order, acc=finite_diff_accuracy)['center'])
    return stencil
s = get_stencil()

# Gets all the unique offsets for all QATS orders in `positions`.
positions = list(set().union(*[set(_["offsets"]) for _ in s]))
lambdas_fd = [_*finite_diff_delta for _ in positions]
lambdas_fd_atom_values = np.array([get_lambda_atom_values(atomic_numbers, l_val) for l_val in lambdas_fd])

# Calculates energies for finite differences.
fd_qc_data = get_qc_data(qc_method, lambdas_fd_atom_values)

# Computes polynomial coefficients.
poly_coeffs = []
for order, stencil in enumerate(s):
    contribution = 0
    for o, c in zip(stencil["offsets"], stencil["coefficients"]):
        contribution += fd_qc_data['total_energies'][lambdas_fd.index(o*finite_diff_delta)] * c
    contribution /= finite_diff_delta ** order
    contribution /= np.math.factorial(order)
    poly_coeffs.append(contribution)

# Making QATS predictions
def qats_pred(poly_coeffs, order, lambda_values):
    return np.polyval(poly_coeffs[:order+1][::-1], lambda_values)

lambda_qats_energies = []
for order in range(0, max_qats_order+1):
    lambda_qats_energies.append(qats_pred(poly_coeffs, order, lambda_values))

# Plotting predictions.
plt.plot(lambda_values, lambda_qc_data['total_energies'], label=f'{{qc_method_label}}/{{basis_set}}', color=ref_color)
for order in range(0, max_qats_order+1):
    plt.plot(
        lambda_values,
        lambda_qats_energies[order],  # QATS predictions
        label=f"QATS{{order}}",
        marker='o',
        markeredgewidth=0,
        alpha=1.0,
        linestyle=''
    )
plt.legend()
plt.xlabel('$\lambda$')
plt.ylabel('Energy (Eh)')
plt.ylim(min(lambda_qc_data['total_energies'])-1, max(lambda_qc_data['total_energies'])+1)
plt.savefig(f'{{work_dir}}/{{calc_label}}-qc.qa.energies.png', dpi=1000)
plt.close()

# Plotting error.
for order in range(0, max_qats_order+1):
    plt.plot(
        lambda_values,
        lambda_qats_energies[order] - lambda_qc_data['total_energies'],
        label=f"QATS{{order}}",
        marker='o',
    )
plt.legend()
plt.ylabel("Error [Eh]")
plt.xlabel("$\lambda$")
plt.ylim(-1, 1)
plt.savefig(f'{{work_dir}}/{{calc_label}}-qcts.error.png', dpi=1000)
plt.close()


# JSON file.

# Merging QC data

assert lambdas_fd[0] == 0.0
all_lambdas = np.hstack((lambda_values, lambdas_fd[1:]))  # Trims first one (which is zero).
sort_i = np.argsort(all_lambdas)
all_lambdas = np.take_along_axis(all_lambdas, sort_i, axis=0)

all_total_energies = np.take_along_axis(np.hstack((lambda_qc_data['total_energies'], fd_qc_data['total_energies'][1:])), sort_i, axis=0)
if qc_method_lower in ['hf', 'ccsd', 'ccsd(t)']:
    all_hf_energies = np.take_along_axis(np.hstack((lambda_qc_data['hf_energies'], fd_qc_data['hf_energies'][1:])), sort_i, axis=0)
else:
    all_dispersion_energies = np.take_along_axis(np.hstack((lambda_qc_data['dispersion_energies'], fd_qc_data['dispersion_energies'][1:])), sort_i, axis=0)

qats_energies = dict()
for order in range(0, max_qats_order+1):
    qats_energies[str(order)] = qats_pred(poly_coeffs, order, all_lambdas)


json_dict = {{
    'schema_name': 'pyscf_qa_output',
    'provenance': {{
        'creator': 'PySCF',
        'version': pyscf_version
    }},
    'molecule': {{
        'geometry': coords,
        'symbols': atoms
    }},
    'molecular_charge': charge,
    'molecular_multiplicity': multiplicity,
    'name': calc_label,
    'atomic_numbers': atomic_numbers,
    'n_electrons': mol.nelectron,
    'model': {{
        'method': qc_method_label,
        'basis': basis_set,
        'scf_conv_tol': scf_conv_tol,
        'scf_conv_tol_grad': scf_conv_tol_grad,
    }},
    'broken_symmetry': broken_symmetry,
    'finite_diff_delta': finite_diff_delta,
    'finite_diff_acc': finite_diff_accuracy,
    'qa_lambdas': all_lambdas,
    'electronic_energies': all_total_energies,
    'qats_energies': qats_energies,
    'qats_poly_coeffs': poly_coeffs
}}

if qc_method_lower in ['ccsd', 'ccsd(t)']:
    json_dict['hf_energies'] = all_hf_energies
    json_dict['model']['cc_conv_tol'] = cc_conv_tol_normt
    json_dict['model']['cc_conv_tol_normt'] = cc_conv_tol
elif qc_method_lower not in ['hf', 'ccsd', 'ccsd(t)']:
    json_dict['dispersion_energies'] = all_dispersion_energies

if qc_method_lower == 'ccsd(t)':
    try:
        json_dict['triples_corrections'] = np.take_along_axis(
            np.hstack((lambda_qc_data['triples_corrections'], fd_qc_data['triples_corrections'][1:])),
            sort_i, axis=0
        )
    except:
        pass

if 'scf_converged' in lambda_qc_data.keys() and 'scf_converged' in fd_qc_data.keys():
    json_dict['scf_converged'] = np.take_along_axis(np.hstack((lambda_qc_data['scf_converged'], fd_qc_data['scf_converged'][1:])), sort_i, axis=0)
if 'cc_converged' in lambda_qc_data.keys() and 'cc_converged' in fd_qc_data.keys():
    json_dict['cc_converged'] = np.take_along_axis(np.hstack((lambda_qc_data['cc_converged'], fd_qc_data['cc_converged'][1:])), sort_i, axis=0)
if 'scf_spin_squared' in lambda_qc_data.keys() and 'scf_spin_squared' in fd_qc_data.keys():
    json_dict['scf_spin_squared'] = np.take_along_axis(np.hstack((lambda_qc_data['scf_spin_squared'], fd_qc_data['scf_spin_squared'][1:])), sort_i, axis=0)
if 'cc_spin_squared' in lambda_qc_data.keys() and 'cc_spin_squared' in fd_qc_data.keys():
    json_dict['cc_spin_squared'] = np.take_along_axis(np.hstack((lambda_qc_data['cc_spin_squared'], fd_qc_data['cc_spin_squared'][1:])), sort_i, axis=0)
try:
    if 't1_diagnostic' in lambda_qc_data.keys():
        json_dict['t1_diagnostic'] = np.take_along_axis(np.hstack((lambda_qc_data['t1_diagnostic'], fd_qc_data['t1_diagnostic'][1:])), sort_i, axis=0)
except:
    pass

if has_cclib:
    json_string = json.dumps(
        json_dict, cls=cclib.io.cjsonwriter.JSONIndentEncoder,
        sort_keys=True, indent=4
    )
else:
    json_string = json.dumps(
        json_dict,
        sort_keys=True
    )

with open(f'{{work_dir}}/{{calc_label}}.json', 'w') as f:
    f.write(json_string)
"""
)
    with open(job_dir + 'qa-pyscf-calc.py', 'w') as f:
        f.write(pyscf_qa_script)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(calc_dir, exist_ok=True)

    # Loop through all systems.
    for atoms_label in qa_calcs_all.keys():
        atoms = atoms_label.split('.')
        save_dir = calc_dir + atoms_label.lower()
        os.makedirs(save_dir, exist_ok=True)
        # Loop through every state.
        for calc_dict in qa_calcs_all[atoms_label]:
            state_label = calc_dict['state']
            chrg_str, mult_str = state_label.split('.')
            chrg = int(chrg_str[4:])
            mult = int(mult_str[4:])

            basis_set = calc_dict['basis_set']
            qc_method = calc_dict['qc_method']
            pattern = re.compile('[\W_]+')
            method_label = pattern.sub('', qc_method.lower())
            basis_set_label = pattern.sub('', basis_set.lower())
            dimer_sep_range = calc_dict['dimer_sep_range']
            dimer_sep_step = calc_dict['dimer_sep_step']
            broken_symmetry = calc_dict['broken_symmetry']
            force_unrestrict_spin = calc_dict['force_unrestrict_spin']
            if dimer_sep_range == 'None' and dimer_sep_step == 'None':
                system_name = f'{atoms_label.lower()}.{state_label}'
                job_name = f'{system_name}-pyscf.qa-{method_label}.{basis_set_label}'
                if broken_symmetry == 'True':
                    job_name += '.brokensym'
                elif force_unrestrict_spin == 'True':
                    job_name += '.spinunrestrict'
                prepare_pyscf_qa_calc(
                    system_name, job_name,
                    atoms, chrg, mult, calc_dict,
                    save_dir=save_dir
                )
            else:
                save_dir_dimer = save_dir + f'/{atoms_label.lower()}.{state_label}'
                os.makedirs(save_dir_dimer, exist_ok=True)
                dimer_sep_range = tuple(map(float, dimer_sep_range[1:-1].split(', ')))
                for dimer_sep in np.arange(
                    min(dimer_sep_range),
                    max(dimer_sep_range) + float(dimer_sep_step)-0.0001, float(dimer_sep_step)
                ):
                    dimer_sep = round(dimer_sep, 2)
                    system_name = f'{atoms_label.lower()}.{state_label}.sep{dimer_sep}'
                    job_name = f'{system_name}-pyscf.qa-{method_label}.{basis_set_label}'
                    if broken_symmetry == 'True':
                        job_name += '.brokensym'
                    elif force_unrestrict_spin == 'True':
                        job_name += '.spinunrestrict'
                    prepare_pyscf_qa_calc(
                        system_name, job_name,
                        atoms, chrg, mult, calc_dict,
                        save_dir=save_dir_dimer, dimer_sep=dimer_sep
                    )


if __name__ == "__main__":
    main()
