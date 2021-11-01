# APDFT for Atoms and Dimers

## Requirements

Python3 with the following packages:

- NumPy
- Pandas
- SciPy
- findiff

## Installation

In order to use, reproduce, and analyze APDFT atom and dimer data you need to install the `apdft_tools` package from the [GitHub repository](https://github.com/keithgroup/apdft-atoms-dimers).
You just have to clone and install the GitHub repo like so.

```text
git clone https://github.com/keithgroup/apdft-atoms-dimers
cd apdft-atoms-dimers
pip install .
```

## Contents

### json-data

Cumulative JSON files for atoms and dimers containing quantum chemistry and APDFT data for all species of interest.
Individual JSON and log files for [atoms](https://github.com/keithgroup/apdft-atoms-data) and [dimers](https://github.com/keithgroup/apdft-dimers-data) can be found in their respective repositories.

### apdft_tools

A collection of modules that streamline the APDFT analysis workflow.

- `data.py`: Manages parsing and converting JSON files into panda data frames.
- `prediction.py`: Property predictions or querying of data frames.
- `analysis.py`: Automated functions to assist analysis.
- `utils.py`: Useful functions that support other modules.

### scripts

Routines to check and analyze calculation JSON files and generate PySCF/APDFT calculations.

### notebooks

Jupyter notebook examples of how to query and analyze atom and dimer calculations.
Ionization energies, electron affinities, excited states, equilibrium bond lengths, and potential energy surfaces.

## License

Distributed under the MIT License. See `LICENSE` for more information.
