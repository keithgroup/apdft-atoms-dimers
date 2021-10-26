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

only_filename = False
print_converged = False
only_not_fin_diff = True
max_fin_diff = 0.02
data_dir = '/home/alex/repos/apdft-atoms-data'

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
    did_not_converge = []
    did_converge = []

    print(f'There are a total of {len(all_output_paths)} calculations\n')
    # Loops through all QCJSON files and adds APDFT information.
    for json_path in all_output_paths:
        
        json_dict = read_json(json_path)
        l_values = json_dict['apdft_lambdas']
        if only_not_fin_diff:
            bool_idx = [True if abs(i) <= max_fin_diff else False for i in l_values]
        else:
            bool_idx = [True for _ in l_values]
        if 'cc_converged' in json_dict.keys():
            cc_conv = np.array(json_dict['cc_converged'])
            all_converged_cc = np.all(cc_conv[bool_idx])
        if 'scf_converged' in json_dict.keys():
            scf_conv = np.array(json_dict['scf_converged'])
            all_converged_scf = np.all(scf_conv[bool_idx])
        
        if only_filename:
            json_name = json_path.split('/')[-1]
        else:
            json_name = json_path
        if not all_converged_scf or not all_converged_cc:
            did_not_converge.append(json_name)
        else:
            did_converge.append(json_name)
        
    print('The following calculations did not all converge:\n')
    for i in did_not_converge: print(i)
    print(f'\n{len(did_not_converge)} did not converge.')
    if print_converged:
        print(f'\n\nThe following {len(did_converge)} calculations converged:')
        for i in did_converge: print(i)

if __name__ == "__main__":
    main()
