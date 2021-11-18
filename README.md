# Quantum Alchemy for Atoms and Dimers

## Requirements

Python3 with the following packages:

- NumPy
- Pandas
- SciPy
- findiff

## Installation

In order to use, reproduce, and analyze quantum alchemy atom and dimer data you need to install the `qa_tools` package from the [GitHub repository](https://github.com/keithgroup/apdft-atoms-dimers).
You just have to clone and install the GitHub repo like so.

```text
git clone https://github.com/keithgroup/apdft-atoms-dimers
cd apdft-atoms-dimers
pip install .
```

## Contents

### json-data

Cumulative JSON files for atoms and dimers containing quantum chemistry and quantum alchemy data for all species of interest.
Individual JSON, log files, and explanations for [atoms](https://github.com/keithgroup/apdft-atoms-data) and [dimers](https://github.com/keithgroup/apdft-dimers-data) data can be found in their respective repositories.

### qa_tools

A collection of modules that streamline the quantum alchemy analysis workflow.

- `data.py`: Manages parsing and converting JSON files into panda data frames.
- `prediction.py`: Property predictions or querying of data frames.
- `analysis.py`: Automated functions to assist analysis.
- `utils.py`: Useful functions that support other modules.

### scripts

Routines to check and analyze calculation JSON files and generate PySCF/quantum alchemy calculations.

### notebooks

Jupyter notebook examples of how to query and analyze atom and dimer calculations.
Ionization energies, electron affinities, excited states, equilibrium bond lengths, and potential energy surfaces.

## Terminology

We will be using several terms throughout the repository.

- **Quantum chemistry (QC)**: Data using straightforward wave function methods or density functional theory.
- **Quantum alchemy (QA)**: Predictions of target systems through nuclear charge perturbations of reference systems.
- **Target**: A specific system, and its respective property, we are interested in predicting.
For example, the ionization energy, electron affinity, or multiplicity gap of N.
- **Reference**: A system, used with quantum alchemy, to predict target energies and properties.
Must have the same number of electrons as the target system.
For example, to predict the energy of N we could use B<sup>2&ndash;</sup>, C<sup> &ndash;</sup>, O<sup>+</sup>, and F<sup>2+</sup>.
- **Quantum alchemy with Taylor series (QATS)**: Approximating the alchemical potential energy surface (PES) using a Taylor series centered at &#8710;Z = 0.
Central finite difference using a h of 0.01 was used to calculate first through fourth derivatives.

## License

Distributed under the MIT License. See `LICENSE` for more information.
