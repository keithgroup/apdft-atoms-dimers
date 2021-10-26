# APDFT

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [Modules](#modules)
  - [utils](#utils)
  - [data](#data)
  - [pred](#pred)
- [Units](#units)
- [License](#license)

## Requirements

Python 3 with the following packages:

- cclib
- NumPy
- Pandas
- SciPy
- findiff

## Modules

### utils

### data

### pred

## Units

All QCJSONs use Angstrom and Hartree as the units of distance and energy, respectively.
Derived units adhere to this specification as well; for example, gradients will be in Hartree/Angstrom.
If properties use other units (e.g., cm<sup>-1</sup> for vibrational frequencies) these are specified in the key descriptions above.

## License

Distributed under the MIT License. See `LICENSE` for more information.
