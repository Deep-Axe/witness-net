"""Shared contracts without CARLA, CAVISE, numerical, or model SDK imports."""

import os as _os

# NumPy work here is small matrices on many calls; threaded BLAS only spins
# (measured: ~3.5 min of system CPU per test run). Respect explicit settings.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
