# utils.py
"""
This module contains utility functions for the simulation.
Includes array initialization and HDF5 file handling.
"""

import numpy as np
import h5py
from numba import njit

@njit
def init_arrays(Nz, Nn):
    """Initializes and returns all the numpy arrays needed for the simulation."""
    # Create empty arrays for the main physical quantities
    psi = np.zeros((Nz, Nn + 1))  # Stream function
    omg = np.zeros((Nz, Nn + 1))  # Vorticity
    tem = np.zeros((Nz, Nn + 1))  # Temperature

    # These arrays store the time derivatives for the Adams-Bashforth method
    domgdt = np.zeros((Nz, Nn + 1, 2))
    dtemdt = np.zeros((Nz, Nn + 1, 2))

    # --- Set up the initial temperature profile ---
    z_vals = np.linspace(0, 1, Nz)
    # A linear background temperature gradient (hot at bottom, cold at top)
    tem[:, 0] = 1 - z_vals
    
    # Add small sinusoidal perturbations to kickstart convection
    tem[:, 1] = 0.07 * np.sin(np.pi * z_vals)
    tem[:, 8] = 0.07 * np.sin(np.pi * z_vals)

    return psi, omg, tem, domgdt, dtemdt

def initialize_h5_movie_file(h5file, nx, nz, total_frames, ra, pr, aspect):
    """Sets up the HDF5 file with the correct groups and datasets for storing movie frames."""
    # Store simulation parameters as metadata, which is super useful for later analysis
    meta = h5file.create_group("metadata")
    meta.attrs["ra"] = ra
    meta.attrs["pr"] = pr
    meta.attrs["aspect"] = aspect
    meta.attrs["nx"] = nx
    meta.attrs["nz"] = nz
    meta.attrs["total_frames"] = total_frames

    # Pre-allocate space for the data, which is much more efficient than appending
    h5file.create_dataset('temperature', shape=(total_frames, nx, nz), dtype='float32')
    h5file.create_dataset('streamfunction', shape=(total_frames, nx, nz), dtype='float32')
    h5file.create_dataset('timesteps', shape=(total_frames,), dtype='int32')

def save_movie_frame_h5(h5file, temmov, psimov, frame_idx, timestep):
    """Saves a single frame of data to the HDF5 file."""
    h5file['temperature'][frame_idx] = temmov
    h5file['streamfunction'][frame_idx] = psimov
    h5file['timesteps'][frame_idx] = timestep
