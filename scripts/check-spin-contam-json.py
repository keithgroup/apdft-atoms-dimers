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
from apdft_tools.utils import calc_spin

data_dir = '../../apdft-dimers-data/data'

only_filename = True  # Instead of printing the absolute path, we print just the filename.
spin_deviation = 0.1   # Minimum spin deviation to consider "contaminated".


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
    high_spin_contam = []
    spin_errors = []
    n_unrestricted = 0

    # Loops through all QCJSON files and adds APDFT information.
    for json_path in all_output_paths:
        json_dict = read_json(json_path)
        multiplicity = json_dict['molecular_multiplicity']
        spin_expected = ((multiplicity - 1)/2)
        if 'scf_spin_squared' in json_dict.keys():
            n_unrestricted += 1
            if 'cc_spin_squared' in json_dict.keys():
                spin_squared_observed = json_dict['cc_spin_squared']
            else:
                spin_squared_observed = json_dict['scf_spin_squared']
            spin_error = np.array(
                [calc_spin(i) - spin_expected for i in spin_squared_observed if i is not None]
            )
            if np.any(spin_error[spin_error>spin_deviation]):
                if only_filename:
                    json_name = json_path.split('/')[-1]
                else:
                    json_name = json_path
                high_spin_contam.append(json_name)
                spin_errors.append(spin_error)
    
    print(f'The following calculations had high spin contamination (above {spin_deviation}):\n')
    for json_name,spin_error in zip(high_spin_contam, spin_errors):
        print(json_name)
        print(f'Spin errors: {[i for i in spin_error if i > spin_deviation]} out of {len(spin_error)}')
        print()
    print(f'\n{len(high_spin_contam)} calcs have high spin contamination out of {n_unrestricted} calcs.')
            
        

if __name__ == "__main__":
    main()
