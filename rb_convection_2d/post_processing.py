# post_processing.py
"""
This module handles post-simulation tasks.
Mainly for transforming data from spectral space to real space for plotting and saving.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import os 

@njit(parallel=True)
def compute_visualization_data(tem, psi, cosa, sina, nn, nx, nz):
    """Transforms spectral data (modes) into a 2D grid for creating movie frames."""
    temmov = np.zeros((nx, nz))
    psimov = np.zeros((nx, nz))

    # This is essentially a parallelized inverse Fourier transform
    for i in prange(nx):
        for k in range(nz):
            tem_sum = 0.0
            psi_sum = 0.0
            for n in range(nn + 1):
                tem_sum += tem[k, n] * cosa[n, i]
                psi_sum += psi[k, n] * sina[n, i]
            temmov[i, k] = tem_sum
            psimov[i, k] = psi_sum
    return temmov, psimov

@njit(parallel=True)
def compute_tem_omg_psi_values(Nz, Nn, Nx, tem, omg, psi, z, x, inva):
    """Another version to compute real-space values, specifically for the final plot."""
    tem_plot = np.zeros((Nx, Nz))
    psi_plot = np.zeros((Nx, Nz))

    for i in prange(Nx):
        for k in range(Nz):
            for n in range(Nn + 1):
                cosa = np.cos(float(n) * np.pi * x[i] * inva)
                sina = np.sin(float(n) * np.pi * x[i] * inva)
                tem_plot[i, k] += tem[k, n] * cosa
                psi_plot[i, k] += psi[k, n] * sina

    return tem_plot, psi_plot

def plot_flow_field(initial_tem, tem, omg, psi, Nz, Nn, Nx, a, save_dir, tsteps):
    """Generates and saves the final plot with temperature and stream function contours."""
    z = np.linspace(0, 1, Nz)
    x = np.linspace(0.0, a, Nx)
    inva = 1.0 / a
    
    # Get the data ready for plotting
    tem_plot, psi_plot = compute_tem_omg_psi_values(Nz, Nn, Nx, tem, omg, psi, z, x, inva)
    initial_tem_plot, _ = compute_tem_omg_psi_values(Nz, Nn, Nx, initial_tem, omg, psi, z, x, inva)
    
    # --- Create the plots ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)
    fig.suptitle(f'Rayleigh-Bénard Convection Results', fontsize=16)

    # Plot 1: Initial Temperature
    cs1 = axs[0].contourf(x, z, initial_tem_plot.T, cmap="cividis", levels=50)
    fig.colorbar(cs1, ax=axs[0], label="Temperature")
    axs[0].set_title("Initial Temperature State")
    axs[0].set_ylabel("z")

    # Plot 2: Final Temperature
    cs2 = axs[1].contourf(x, z, tem_plot.T, cmap="cividis", levels=50)
    fig.colorbar(cs2, ax=axs[1], label="Temperature")
    axs[1].set_title(f"Final Temperature after {tsteps} steps")
    axs[1].set_ylabel("z")

    # Plot 3: Final Stream Function
    cs3 = axs[2].contourf(x, z, psi_plot.T, cmap="viridis", levels=50)
    fig.colorbar(cs3, ax=axs[2], label="Stream Function")
    axs[2].contour(x, z, psi_plot.T, colors='k', linewidths=0.5) # Add contour lines for clarity
    axs[2].set_title("Final Flow Structure (Stream Function)")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("z")

    # Save the final figure
    save_path = os.path.join(save_dir, "final_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Final plot saved at: {save_path}")
    plt.show()
