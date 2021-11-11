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
from apdft_tools.analysis import *

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

def test_alchemical_errors():
    n_electrons = 15
    excitation_level = 0
    basis_set = 'aug-cc-pV5Z'
    return_energies = False

    state_labels, qa_errors = get_alchemical_errors(
        df_qc_atom, n_electrons, excitation_level=excitation_level,
        basis_set=basis_set, return_energies=return_energies
    )

    state_lables_manual = ['al.chrg-2.mult4', 'si.chrg-1.mult4', 'p.chrg0.mult4', 's.chrg1.mult4', 'cl.chrg2.mult4']
    qa_errors_manual = np.array(
        [[0.0, 0.7828883447321857, 2.835326525172661, 5.658270797490502, 8.674296487280657],
        [1.0055729091639591, 0.0, 0.7599298992914783, 2.808954243598066, 5.631777796490667],
        [4.11854036214288, 0.9619192528856502, 0.0, 0.7597458717308427, 2.8079653948236682],
        [9.631753252025021, 3.9649925208588, 0.9300921688667358, 0.0, 0.7619958524857111],
        [17.835355149857264, 9.267295696851704, 3.8360369384392357, 0.9087141331775115, 0.0]]
    )
    assert state_labels == state_lables_manual
    assert np.allclose(qa_errors, qa_errors_manual)
