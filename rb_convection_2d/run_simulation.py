# run_simulation.py
"""
This is the main script to run the Rayleigh-Bénard convection simulation.
It sets the parameters, orchestrates the simulation loop, and handles the final output.
"""

import numpy as np
import time
import os
import copy
import h5py
from numba import set_num_threads

# --- Import our custom modules ---
# These imports bring in the functions we've organized into other files
from utils import init_arrays, initialize_h5_movie_file, save_movie_frame_h5
from core_solvers import compute_linear_terms, compute_nonlinear_terms, update_variables, compute_psi
from post_processing import compute_visualization_data, plot_flow_field, compute_tem_omg_psi_values

def simulation(Ra, Pr, Nz, Nn, Nx, tsteps, a, dt, save_dir, t_write, frame_save):
    """The main orchestrator for the simulation loop."""
    # --- Pre-computation and Initialization ---
    pi = np.pi
    dz = 1 / (Nz - 1)
    c = pi / a
    c1 = 0.25 * c / dz
    c2 = pi / (2 * a)
    oodz2 = 1 / (dz * dz)

    # Get our initial arrays from the utils module
    psi, omg, tem, domgdt, dtemdt = init_arrays(Nz, Nn)
    initial_tem = copy.deepcopy(tem) # Keep a copy of the start for the final plot
    center_index = int(0.32 * Nz) # For saving data from the middle of the box

    # --- Setup for HDF5 Movie File ---
    total_frames = tsteps // frame_save
    h5_filename = os.path.join(save_dir, "movie_data.h5")
    with h5py.File(h5_filename, 'w') as h5file:
        initialize_h5_movie_file(h5file, Nx, Nz, total_frames, Ra, Pr, a)
        x = np.linspace(0, a, Nx)
        z = np.linspace(0, 1, Nz)
        h5file['metadata'].create_dataset('x', data=x)
        h5file['metadata'].create_dataset('z', data=z)

        # Pre-calculate sine and cosine values for visualization to speed things up
        ooaspect = 1.0 / a
        sina = np.zeros((Nn + 1, Nx))
        cosa = np.zeros((Nn + 1, Nx))
        for n in range(Nn + 1):
            for i in range(Nx):
                sina[n, i] = np.sin(float(n) * pi * x[i] * ooaspect)
                cosa[n, i] = np.cos(float(n) * pi * x[i] * ooaspect)
        
        # Coefficients for the tridiagonal solver
        sub = np.full(Nz, -oodz2)
        sup = np.full(Nz, -oodz2)
        
        frame_idx = 0
        print("--- Starting Simulation Loop ---")

        # --- Main Time Loop ---
        for t in range(tsteps):
            # 1. Calculate derivatives at the current time step
            dtemdt, domgdt = compute_linear_terms(Nz, Nn, oodz2, c, Ra, Pr, tem, omg, dtemdt, domgdt)
            dtemdt, domgdt = compute_nonlinear_terms(Nz, Nn, dz, tem, psi, omg, c1, c2, dtemdt, domgdt)

            # 2. Update the fields to the next time step
            tem, omg = update_variables(Nz, Nn, dt, tem, omg, dtemdt, domgdt)

            # 3. Solve for the new stream function
            psi = compute_psi(omg, psi, sub, sup, c, Nn, Nz, oodz2)

            # 4. Enforce boundary conditions (important!)
            omg[0, :], omg[-1, :] = 0, 0
            tem[0, 1:], tem[-1, 1:] = 0, 0
            psi[0, :], psi[-1, :] = 0, 0

            # 5. Prepare for the next step by shifting the time-derivative arrays
            dtemdt[1:-1, :, 0] = dtemdt[1:-1, :, 1]
            domgdt[1:-1, :, 0] = domgdt[1:-1, :, 1]

            # 6. Save a movie frame if it's time
            if t % frame_save == 0 and frame_idx < total_frames:
                temmov, psimov = compute_visualization_data(tem, psi, cosa, sina, Nn, Nx, Nz)
                save_movie_frame_h5(h5file, temmov, psimov, frame_idx, t)
                frame_idx += 1
            
            # Print progress update
            if t % t_write == 0:
                print(f'Timestep {t}/{tsteps} completed')
        
        print("--- Simulation Loop Finished ---")
        
    return tem, omg, psi, center_index, initial_tem


if __name__ == "__main__":
    # --- Simulation Parameters ---
    # Adjust these to explore different physical regimes!
    Ra_val = 1e6      # Rayleigh number: The driving force of convection
    Pr = 0.7         # Prandtl number: Ratio of momentum to thermal diffusivity
    Nz = 101          # Vertical grid points
    Nn = 50          # Number of Fourier modes in the horizontal
    Nx = 201          # Horizontal grid points (for visualization only)
    tsteps = int(1e6) # Total number of timesteps to run
    
    # --- Performance & I/O Parameters ---
    set_num_threads(6) # Set how many CPU cores Numba should use
    a = np.sqrt(2)            # Aspect ratio of the simulation box
    dt =  0.6*3e-6  # Time step size (CRITICAL for stability!)
    frame_save = 5*100  # How often to save a frame to the movie file
    t_write = tsteps // 10 # How often to print a progress update

    # --- Setup Output Directory ---
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"./simulation_results_Ra{Ra_val:.1e}_Pr{np.round(Pr,2)}_{time_stamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    # --- Run the Simulation ---
    start_time = time.time()
    tem, omg, psi, center_index, initial_tem = simulation(
        Ra_val, Pr, Nz, Nn, Nx, tsteps, a, dt, save_dir, t_write, frame_save
    )
    end_time = time.time()
    print(f"Total simulation time: {(end_time - start_time) / 3600:.2f} hours")

    # --- Final Analysis & Output ---
    # Save the final spectral coefficients to a text file
    file_path = os.path.join(save_dir, "final_spectral_values.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{'n':<10}{'tem_n':<20}{'omg_n':<20}{'psi_n':<20}\n")
        file.write("-" * 70 + "\n")
        for n in range(min(Nn + 1, 21)):  # Save up to the 20th mode
            t_val = tem[center_index, n]
            o_val = omg[center_index, n]
            p_val = psi[center_index, n]
            file.write(f"{n:<10}{t_val:<20.6f}{o_val:<20.6f}{p_val:<20.6f}\n")
    print(f"Final spectral values saved in {file_path}")
    
    # Generate the final plot
    plot_flow_field(initial_tem, tem, omg, psi, Nz, Nn, Nx, a, save_dir, tsteps)
