import os
import netCDF4 as nc
import numpy as np
import cv2
import random
import csv

# Define the folder where images will be saved
base_image_folder = 'ImageDataMay30'
metadata_file = os.path.join(base_image_folder, 'image_metadata.csv')  # Path for the metadata CSV file
separated_image_folder = 'SeparatedImageDataMay30'

# Ensure the base image directory exists before creating or opening files within it
os.makedirs(base_image_folder, exist_ok=True)
os.makedirs(separated_image_folder, exist_ok=True)

# Define the paths to the folders where the datasets are located
day1_folder = os.path.join('q20', 'Day1')
day2_folder = os.path.join('q20', 'Day2')

# Define intervals for each file where images are considered "Cloud"
cloud_intervals = {
    'awe_l1r_2024001_00621_q20_v00.nc': [(76, 365)],
    'awe_l1r_2024001_00622_q20_v00.nc': [(300, 565), (1615, 1745)],
    'awe_l1r_2024001_00623_q20_v00.nc': [(870, 965)],
    'awe_l1r_2024001_00624_q20_v00.nc': [(48, 135), (440, 683), (940, 1085)],
    'awe_l1r_2024001_00625_q20_v00.nc': [(75, 350), (436, 620)],
    'awe_l1r_2024001_00626_q20_v00.nc': [(40, 552)],
    'awe_l1r_2024001_00627_q20_v00.nc': [(0, 65), (132, 442), (472, 555), (1162, 1280), (1640, 1742)],
    'awe_l1r_2024001_00628_q20_v00.nc': [(0, 36), (77, 469), (500, 598), (664, 852), (950, 1030), (1865, 1915)],
    'awe_l1r_2024001_00629_q20_v00.nc': [(148, 482)],
    'awe_l1r_2024001_00630_q20_v00.nc': [(0, 35), (160, 520), (1096, 1188)],
    'awe_l1r_2024001_00631_q20_v00.nc': [(0, 108), (165, 654), (890, 1000), (1110, 1256)],
    'awe_l1r_2024001_00632_q20_v00.nc': [(16, 318), (375, 572), (1260, 1342), (1346, 1448), (1478, 1566)],
    'awe_l1r_2024001_00633_q20_v00.nc': [(134, 444), (444, 566), (906, 1085), (1236, 1556), (1612, 1718)],
    'awe_l1r_2024001_00634_q20_v00.nc': [],
    'awe_l1r_2024001_00635_q20_v00.nc': [(0, 95)],
}

# Define circle parameters for image processing
cx, cy, r = 150, 132, 166  # Center and radius of the circle

def create_metadata_file_if_not_exists(file_path):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Latitude', 'Longitude', 'Epoch', 'Category'])

# Function to process and save images and write metadata
def process_and_save_images(filename, radiance, intervals, latitudes, longitudes, times, save_all=False, separated_metadata_file=None):
    cloud_folder = os.path.join(base_image_folder, 'Cloud')
    clear_folder = os.path.join(base_image_folder, 'Clear')
    
    # For Day2 files, create individual folders based on file numbers
    if save_all:
        orbit_number = filename.split('_')[3]
        orbit_folder = os.path.join(separated_image_folder, orbit_number)
        os.makedirs(orbit_folder, exist_ok=True)

    # Ensure the output folders exist
    for folder in [cloud_folder, clear_folder]:
        os.makedirs(folder, exist_ok=True)
    
    metadata_file_to_use = separated_metadata_file if save_all else metadata_file
    with open(metadata_file_to_use, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for time_step in range(radiance.shape[0]):
            is_cloud = any(start <= time_step <= end for start, end in intervals)
            latitude = latitudes[time_step]
            longitude = longitudes[time_step]
            iss_time = times[time_step]
            category = 'Cloud' if is_cloud else 'Clear'
            
            if save_all:
                image_folder = orbit_folder
            else:
                if is_cloud and random.randint(1, 3) == 1:
                    image_folder = cloud_folder
                elif not is_cloud and random.randint(1, 8) == 1:
                    image_folder = clear_folder
                else:
                    continue  # Skip this image
            
            radiance_at_time = radiance[time_step, :, :]
            y, x = np.ogrid[:radiance_at_time.shape[0], :radiance_at_time.shape[1]]
            distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            outside_circle_mask = distance_from_center > r
            radiance_inside_circle = np.copy(radiance_at_time)
            radiance_inside_circle[outside_circle_mask] = np.nan
            radiance_inside_circle[np.isnan(radiance_inside_circle)] = 0
            radiance_norm = cv2.normalize(radiance_inside_circle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            radiance_img = np.uint8(radiance_norm)
            
            image_filename = f'{filename[:-3]}_time_step_{time_step}.png'
            image_path = os.path.join(image_folder, image_filename)
            cv2.imwrite(image_path, radiance_img)  # Save the image
            
            # Write metadata to CSV including new fields
            writer.writerow([image_filename, latitude, longitude, iss_time, category])

# Create the metadata file with headers if it does not exist
create_metadata_file_if_not_exists(metadata_file)

# Process files in 'Day1'
for filename, intervals in cloud_intervals.items():
    dataset_path = os.path.join(day1_folder, filename)
    dataset = nc.Dataset(dataset_path, 'r')
    radiance = dataset.variables['Radiance'][:]
    latitudes = dataset.variables['ISS_Latitude'][:]
    longitudes = dataset.variables['ISS_Longitude'][:]
    times = dataset.variables['Epoch'][:]  # seconds since 1980-01-06 00:00:00 UTC
    dataset.close()
    
    process_and_save_images(filename, radiance, intervals, latitudes, longitudes, times)

# Process separated files in 'Day2'
separated_intervals = {
    'awe_l1r_2024002_00636_q20_v00.nc': [(0, 300)],
    'awe_l1r_2024002_00637_q20_v00.nc': [(182, 336), (35, 436)]
}

for separated_filename, intervals in separated_intervals.items():
    separated_dataset_path = os.path.join(day2_folder, separated_filename)
    separated_metadata_file = os.path.join(separated_image_folder, f'{separated_filename.split("_")[3]}_metadata.csv')
    create_metadata_file_if_not_exists(separated_metadata_file)
    
    separated_dataset = nc.Dataset(separated_dataset_path, 'r')
    separated_radiance = separated_dataset.variables['Radiance'][:]
    separated_latitudes = separated_dataset.variables['ISS_Latitude'][:]
    separated_longitudes = separated_dataset.variables['ISS_Longitude'][:]
    separated_times = separated_dataset.variables['Epoch'][:]  # seconds since 1980-01-06 00:00:00 UTC
    separated_dataset.close()
    
    process_and_save_images(separated_filename, separated_radiance, intervals, separated_latitudes, separated_longitudes, separated_times, save_all=True, separated_metadata_file=separated_metadata_file)
