# Solver of the electron motion equation in given external field
A python module for solving motion equations with radiation reaction and spin. 
Several algorithms for spin-resolved radiation are implemented.
Numba jit compilation is used for parallel processing of multiple particles.

# Dependencies
- scipy
- numba
- numpy
- xarray (optional)
- numba_progress (optional)

# Examples
see `MagneticField.py` or `MagneticField.ipynb`