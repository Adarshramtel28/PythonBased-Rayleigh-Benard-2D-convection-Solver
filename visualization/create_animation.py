# Single-cell implementation that works with existing HDF5 files

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.colors import Normalize
from datetime import datetime
from IPython.display import Image, display
import glob

def create_animation(h5_file_path, output_dir='output', fps=10, dpi=100,
                     cmap_temp='hot', cmap_psi='viridis', start_frame=None, end_frame=None):


    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {h5_file_path}...")

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5file:
        # Get metadata
        metadata = h5file['metadata']
        ra = metadata.attrs['ra']
        pr = metadata.attrs['pr']
        aspect = metadata.attrs['aspect']
        nx = metadata.attrs['nx']
        nz = metadata.attrs['nz']

        # Get grid coordinates
        x = metadata['x'][:]
        z = metadata['z'][:]

        # Get total number of frames
        total_frames = metadata.attrs['total_frames']
        print(f"Found {total_frames} frames in the dataset")

        # Get timesteps for each frame
        timesteps = h5file['timesteps'][:]

        # Handle frame range selection
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = total_frames

        selected_frames = min(end_frame - start_frame, total_frames - start_frame)

        print(f"Creating animation for frames {start_frame} to {min(end_frame, total_frames-1)} ({selected_frames} frames)")


        # Create figure and axes for animation
        fig, ax = plt.subplots(figsize=(10, 6))

        # Find global min/max values for consistent colormaps
        print("Calculating global min/max values for consistent colormap...")
        temp_min = float('inf')
        temp_max = float('-inf')

        # Read a sample of frames to determine colormap limits
        sample_step = max(1, total_frames // 10)  # Sample ~10 frames
        for i in range(0, total_frames, sample_step):
            temp_data = h5file['temperature'][i]
            temp_min = min(temp_min, np.min(temp_data))
            temp_max = max(temp_max, np.max(temp_data))

        # Create normalizer for consistent color scaling
        temp_norm = Normalize(vmin=temp_min, vmax=temp_max)

        # Initial plot setup
        cmap_temp = 'cividis'  ################### Colour of Temp Contour Map
        temp_plot = ax.contourf(x, z, np.zeros((nx, nz)).T, 50, cmap=cmap_temp, norm=temp_norm)

        # Add colorbar
        #cbar = plt.colorbar(temp_plot, ax=ax, label='Temperature')

        # Set titles and labels
        ax.set_title(f'Temperature Field (Ra={ra:.1e}, Pr={pr:.1f})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')

        # Add timestep counter
        time_text = fig.text(0.5, 0.01, '', ha='center')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for timestep text

        # Function to update the frame for animation
        def update_frame(frame_idx):
            # Clear previous plot
            ax.clear()

            # Get data for current frame
            actual_frame = frame_idx + start_frame
            temperature = h5file['temperature'][actual_frame]
            current_timestep = timesteps[actual_frame]

            # Update plot
            temp_plot = ax.contourf(x, z, temperature.T, 50, cmap=cmap_temp, norm=temp_norm)

            # Update title and labels
            ax.set_title(f'Temperature Field (Ra={ra:.1e}, Pr={pr:.1f})')
            ax.set_xlabel('x')
            ax.set_ylabel('z')

            # Update timestep text
            time_text.set_text(f'Timestep: {current_timestep}')
            print(f"\rCreating frame {frame_idx+1}/{selected_frames}", end="")
            return [ax, time_text]

        # Create animation
        print("\nGenerating animation...")
        anim = animation.FuncAnimation(
            fig, update_frame, frames=selected_frames,
            blit=False, interval=1000/fps
        )

        # Save as GIF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"temperature_animation_{timestamp}.gif")
        print(f"Saving animation to {output_file}...")
        anim.save(
            output_file,
            writer='pillow',
            fps=fps,
            dpi=dpi,
            progress_callback=lambda i, n: print(f"\rSaving frame {i+1}/{n}", end="")
        )
        print(f"\nAnimation saved to {output_file}")

        # Close the matplotlib figure
        plt.close(fig)
        return output_file

# Main execution
# Just specify the path to your HDF5 file here
h5_file_path =   #### CHANGE THIS PATH to your HDF5 file location. **Tip - use file path as raw string - eg., - r"C:Users\name\location_to_h5file" 

# Create animation
output_file = create_animation(
    h5_file_path=h5_file_path,
    output_dir='output',
    fps=15,
    dpi=400,
    cmap_temp='coolwarm',
    cmap_psi='viridis',
    start_frame=None,
    end_frame=None
)

# Display the generated GIF
display(Image(output_file))
