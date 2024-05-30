# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:15:17 2024

@author: Anh
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider  # Import RangeSlider
import cv2
import os

# Define the path to the folder where the dataset is located
dataset_folder = os.path.join('q20', 'Day2')

# Define the filename of the dataset
dataset_filename = 'awe_l1r_2024002_00647_q20_v00.nc'

# Combine the folder path and filename to get the full path to the dataset
dataset_path = os.path.join(dataset_folder, dataset_filename)

# Load the dataset
dataset = nc.Dataset(dataset_path, 'r')
radiance = dataset.variables['Radiance'][:]
iss_latitude = dataset.variables['ISS_Latitude'][:]  # Load ISS latitude data
iss_longitude = dataset.variables['ISS_Longitude'][:]  # Load ISS longitude data

print("=== Global Attributes ===")
for attr in dataset.ncattrs():
    print(f"{attr}: {dataset.getncattr(attr)}")

print("\n=== Dimensions ===")
for dim in dataset.dimensions.keys():
    print(f"{dim}: {len(dataset.dimensions[dim])}")

print("\n=== Variables ===")
for var in dataset.variables.keys():
    print(f"{var}: {dataset.variables[var]}")
    print("Attributes:")
    for attr in dataset.variables[var].ncattrs():
        print(f"    {attr}: {dataset.variables[var].getncattr(attr)}")

# Close the dataset
dataset.close()

# Initial setup
current_time_step = 0
vmin_default = 4
vmax_default = 12

# Create figure and axes for the plot and histogram

fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 row, 2 columns for side-by-side

# Adjust the figure to add space for the slider and range slider
plt.subplots_adjust(bottom=0.3)

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0]-1, valinit=current_time_step, valfmt='%0.0f')

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.55, 0.05, 0.35, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 28, valinit=(vmin_default, vmax_default))

colorbar = None  # To keep track of the colorbar

# import matplotlib.patches as patches

def update_plot(time_step):
    global colorbar
    time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Clear previous content
    axs[0].clear()
    radiance_at_time = radiance[time_step, :, :]
    iss_lat = iss_latitude[time_step]
    iss_lon = iss_longitude[time_step]
    
    # Define circle parameters
    cx, cy, r = 150, 132, 166  # Center and radius of the circle
    
    # Create a grid of x,y coordinates that match the pixel positions
    y, x = np.ogrid[:radiance_at_time.shape[0], :radiance_at_time.shape[1]]
    
    # Calculate the distance of all points from the center of the circle
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create a mask for points outside the circle (distance from center > radius)
    outside_circle_mask = distance_from_center > r
    
    # Apply the mask to set values outside the circle to NaN (or another value that indicates they should not be plotted)
    radiance_inside_circle = np.copy(radiance_at_time)
    radiance_inside_circle[outside_circle_mask] = np.nan  # Use np.nan for missing data
    
    # Plot the masked radiance data
    #img = 
    axs[0].imshow(radiance_inside_circle, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Radiance at Time Step {time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}')
    axs[0].set_xlabel('Spatial Dimension X')
    axs[0].set_ylabel('Spatial Dimension Y')
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    axs[0].set_aspect('equal')
    
    # # Draw the circle
    # circle = patches.Circle((cx, cy), r, edgecolor='red', facecolor='none')
    # axs[0].add_patch(circle)

    # Update histogram - now only for points inside the circle
    radiance_inside_flat = radiance_inside_circle.flatten()
    radiance_inside_flat = radiance_inside_flat[~np.isnan(radiance_inside_flat)]  # Remove NaN values for histogram
    axs[1].clear()
    axs[1].hist(radiance_inside_flat, range=(0, 28), bins=50, color='blue', edgecolor='black')
    axs[1].set_title(f'Histogram of Radiance at Time Step {time_step} (Inside Circle)')
    axs[1].set_xlabel('Radiance Value')
    axs[1].set_ylabel('Frequency')
    axs[1].grid(True)

    plt.draw()


# Connect the slider and range slider to the update_plot function
slider.on_changed(update_plot)
range_slider.on_changed(lambda val: update_plot(slider.val))

# Function to handle key presses for time step navigation
def on_key(event):
    global current_time_step
    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)  # Adjusted for consistency
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)  # Adjusted for consistency
    elif event.key == 'up':
        current_time_step = max(current_time_step - 20, 0)
    elif event.key == 'down':
        current_time_step = max(current_time_step + 20, 0)
    slider.set_val(current_time_step)  # This will automatically update the plot via the slider's on_changed event

# Connect the key press event to the on_key function
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial plot update
update_plot(current_time_step)

plt.show()

