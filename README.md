# 2D Rayleigh-Bénard Convection Simulation

This repository contains a Python code for simulating 2D Rayleigh-Bénard convection, a phenomenon of fluid dynamics that occurs when a fluid is heated from below and cooled from above. The simulation is heavily optimized using Numba for high performance.

**Author:** Adarsh Ramtel, MSc Physics, IIT Roorkee
**Core Algorithm:** The simulation algorithm is based on the book "Introduction to Modelling Convection in Planets and Stars" by Gary Glatzmeier. All credits to the author for the physical and numerical model.




##  Features

* **High Performance:** Core computational loops are JIT-compiled with Numba and parallelized. 
* **Efficient Solver:** Uses a custom Numba-based Tridiagonal Matrix (Thomas Algorithm) solver to avoid SciPy overhead. 
* **HDF5 Output:** Simulation data is saved efficiently in the HDF5 format, which is ideal for handling large datasets. 
* **Modular Code:** The code is organized into clear functions for initialization, computation, and I/O.
* **Visualization Ready:** Includes scripts for plotting final results and generating animations of the time evolution. 



##  Repository Structure


PythonBased-Rayleigh-Benard-2D-convection-Solver/

- .gitignore               
- README.md                # This file
- requirements.txt         # Project dependencies
-
- rb_convection_2d/        # Main Python source code
   - init.py
   - utils.py             # Setup, array creation, and file I/O
   - core_solvers.py      # The physics and math engine
   - post_processing.py   # Plotting and final data transformation
   - run_simulation.py    # The main script for execution

- visualization/
   - create_animation.py  # Script to make GIFs from output
  
- documentation/
   - project_presentation.pdf 
   - Thesis.pdf
---

##  Setup and Installation

1.  **Clone the repository:**
    
    git clone https://github.com/Adarshramtel28/PythonBased-Rayleigh-Benard-2D-convection-Solver.git
    cd PythonBased-Rayleigh-Benard-2D-convection-Solver

2.  **Create a virtual environment (recommended):**
    
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    

3.  **Install the required libraries:** 
    
    pip install -r requirements.txt
    

---

## How to Run a Simulation


All simulation parameters are set inside the main execution script.

1.Open the main script: rb_convection_2d/run_simulation.py.

2.Set Simulation Parameters: Scroll to the bottom if __name__ == "__main__": block to set the physical and numerical parameters.


# --- Simulation Parameters ---
Ra_val = 1e6      # Rayleigh number
Pr = 10.0         # Prandtl number
Nz = 101          # Vertical grid points
Nn = 50           # Number of Fourier modes
dt = 0.6 * 3e-6   # Time step size (CRITICAL!)


3.Execute the Script: Run the file from your terminal
python rb_convection_2d/run_simulation.py


The simulation will print its progress and save the results, including a `movie_data.h5` file and a final plot (`.png`), inside a new timestamped folder.

### A Note on Stability

**This is extremely important.** The simulation can become unstable and "blow up" if the parameters are not set carefully. 

* As the Rayleigh number **`Ra_val`** increases, the grid resolution (**`Nz`**, **`Nn`**) must also be increased. 
* As **`Nz`** and **`Nn`** increase, the time step **`dt`** must be decreased. 
* A good starting point for the time step is the Courant-Friedrichs-Lewy (CFL) condition: $$ \Delta t < \frac{\Delta z^2}{4} $$. However, due to non-linear terms, you will likely need a **`dt`** much smaller than this, especially at high `Ra`.
* The best approach is trial and error, starting with the value from the stability condition. 

---

##  Generating Animations

The simulation saves a detailed `movie_data.h5` file. You can use the `create_animation.py` script to generate a GIF of the temperature evolution. 

1.  **Open the animation script:** `visualization/create_animation.py`.

2.  **Set the file path:** Update the `h5_file_path` variable to point to your HDF5 file. 

    
    h5_file_path = "../results/simulation_resultsT_.../movie_data.h5"
    

3.  **Run the script from the `visualization` directory:**
    
    cd visualization
    python create_animation.py
    

The script will produce a GIF named `convection_animation.gif` in the same folder. You can tweak the `fps` (frames per second) and `dpi` for a smoother or higher-resolution animation. 

**Important**: Memory Crash Concerns on Laptops 
When creating animations from large datasets (e.g., 1000+ frames), you might run out of memory, causing the program to crash. This is common on laptops which have less RAM than a cluster.

If you experience crashes, the most likely cause is the DPI (Dots Per Inch) setting. A high DPI forces your computer to render a very large image for every single frame, which consumes a lot of memory.
Another cause could be you are saving too frequent frames in the h5 file. Try to estimate the animation requirements and lower the frame save frequency(in simulation) and/or the animation rendering frequency( in animation).

Solution: In create_animation.py, lower the dpi value. A value between 100 and 150 is usually a safe balance between quality and memory usage.
---

## Contact and Citation

If you use this code in your work, please consider providing credit.  If you find any issues, especially regarding stability, and have a fix, feel free to reach out. 

**Email:** adarshramtel28@gmail.com 