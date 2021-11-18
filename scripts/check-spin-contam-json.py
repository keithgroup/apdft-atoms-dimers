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
import json
import numpy as np
from qa_tools.utils import calc_spin

data_dir = '../../qa-atoms-data/data'

only_filename = True  # Instead of printing the absolute path, we print just the filename.
spin_deviation = 0.2   # Minimum spin deviation to consider "contaminated".

# Lambda selection. Allows you to only check certain lambdas for convergence.
# Note that both options can be True at the same time.
only_fin_diff_lambdas = True  # Only check calculations for lambdas used in finite differences.
only_int_lambdas = True  # Only check calculations for integer lambdas.

max_fin_diff = 0.02  # The maximal lambda value used for finite differences.


###   SCRIPT   ###

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_files(path, expression, recursive=True):
    """Returns paths to all files in a given directory that matches a provided
    expression in the file name. Commonly used to find all files of a certain
    type, e.g. output or xyz files.
    
    Parameters
    ----------
    path : :obj:`str`
        Specifies the directory to search.
    expression : :obj:`str`
        Expression to be tested against all file names in 'path'.
    recursive :obj:`bool`, optional
        Recursively find all files in all subdirectories.
    
    Returns
    -------
    :obj:`list` [:obj:`str`]
        All absolute paths to files matching the provided expression.
    """
    if path[-1] != '/':
        path += '/'
    if recursive:
        all_files = []
        for (dirpath, _, filenames) in os.walk(path):
            index = 0
            while index < len(filenames):
                if dirpath[-1] != '/':
                    dirpath += '/'
                filenames[index] = dirpath + filenames[index]
                index += 1
            all_files.extend(filenames)
        files = []
        for f in all_files:
            if expression in f:
                files.append(f)
    else:
        files = []
        for f in os.listdir(path):
            filename = os.path.basename(f)
            if expression in filename:
                files.append(path + f)
    return files

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




def main():
    
    # Finds all QCJSON files in data directory.
    all_output_paths = get_files(data_dir, '.json', recursive=True)
    high_spin_contam_labels = []
    high_spin_errors = []
    lambda_high_spin_errors = []
    n_unrestricted = 0

    # Loops through all QCJSON files and adds QATS information.
    for json_path in all_output_paths:
        json_dict = read_json(json_path)
        multiplicity = json_dict['molecular_multiplicity']
        spin_expected = ((multiplicity - 1)/2)

        # Selecting which lambda values to check
        l_values = json_dict['qa_lambdas']
        bool_idx = [True for _ in l_values]  # Initial values
        if only_fin_diff_lambdas:
            for i in range(len(l_values)):
                if abs(l_values[i]) <= max_fin_diff:
                    bool_idx[i] = True
                else:
                    bool_idx[i] = False
        if only_int_lambdas:
            for i in range(len(l_values)):
                if l_values[i].is_integer():
                    bool_idx[i] = True
                else:
                    if not abs(l_values[i]) <= max_fin_diff and only_fin_diff_lambdas:
                        bool_idx[i] = False
        
        if 'scf_spin_squared' in json_dict.keys():
            n_unrestricted += 1
            if 'cc_spin_squared' in json_dict.keys():
                spin_squared_observed = np.array(json_dict['cc_spin_squared'])[bool_idx]
            else:
                spin_squared_observed = np.array(json_dict['scf_spin_squared'])[bool_idx]
            spin_error = np.array(
                [calc_spin(i) - spin_expected for i in spin_squared_observed if i is not None]
            )
            if np.any(spin_error[spin_error>spin_deviation]):
                if only_filename:
                    json_name = json_path.split('/')[-1]
                else:
                    json_name = json_path
                high_spin_contam_labels.append(json_name)

                bool_idx_high_spin_error = [True if i > spin_deviation else False for i in spin_error]
                high_spin_errors.append(spin_error[bool_idx_high_spin_error])
                lambda_high_spin_errors.append(np.array(l_values)[bool_idx][bool_idx_high_spin_error])
    

    
    print(f'The following calculations had high spin contamination (above {spin_deviation}):\n')
    for json_name,high_spin_error,high_l_values in zip(high_spin_contam_labels, high_spin_errors, lambda_high_spin_errors):
        print(json_name)
        print(f'Lambda values: {high_l_values}')
        print(f'Spin errors: {high_spin_error} out of {len(spin_error)}')
        print()
    print(f'\n{len(high_spin_errors)} calcs have high spin contamination out of {n_unrestricted} calcs.')
            
        

if __name__ == "__main__":
    main()
