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

import math
import numpy as np

from apdft_tools.utils import read_json

json_path_atoms = '../json-data/atom-pyscf.apdft-data.posthf.json'
json_path_converged = './tests/tests_data/n.chrg0.mult4-pyscf-uccsdt.augccpv5z.json'
json_path_not_converged = './tests/tests_data/c.chrg-2.mult3-pyscf-uccsdt.augccpv5z.json'

def test_check_converged_json():
    json_dict = read_json(json_path_converged)

    n_electrons = 7
    charge = 0
    mult = 4
    apdft_lambdas = np.array([-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, -0.02, -0.01, 0.0, 0.01, 0.02, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    electronic_energies = np.array([-24.47433630149931, -27.522589225510888, -30.768271713179974, -34.21659999986884, -37.871699238480694, -41.734888915704204, -45.80518909287975, -50.08139595282322, -54.196634161725555, -54.37948622174288, -54.56266526988731, -54.74617131741751, -54.93000428965259, -59.248284490717715, -64.13747331852693, -69.22928129533933, -74.52251817636171, -80.0156953557857, -85.70697986074306, -91.5941643935081, -97.67465334157724])
    
    assert json_dict['n_electrons'] == n_electrons
    assert json_dict['molecular_charge'] == charge
    assert json_dict['molecular_multiplicity'] == mult
    assert np.array_equal(np.array(json_dict['apdft_lambdas']), apdft_lambdas)
    assert np.array_equal(np.array(json_dict['electronic_energies']), electronic_energies)
