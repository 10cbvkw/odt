import numpy as np
import mrcfile
from scipy.interpolate import griddata

# Step 1: Load the .mrc volume
def load_mrc_file(filepath):
    with mrcfile.open(filepath, mode='r') as mrc:
        volume = mrc.data.copy()  # Copy to avoid working with memory-mapped data
    return volume

# Step 2: Crop the center 5x5x5 volume and set the values to 0
def crop_center(volume, crop_size=5):
    z, y, x = volume.shape
    z_center, y_center, x_center = z // 2, y // 2, x // 2

    # Calculate the boundaries of the crop
    z_start, z_end = z_center - crop_size // 2, z_center + crop_size // 2 + 1
    y_start, y_end = y_center - crop_size // 2, y_center + crop_size // 2 + 1
    x_start, x_end = x_center - crop_size // 2, x_center + crop_size // 2 + 1

    # Set the center region to 0
    volume[z_start:z_end, y_start:y_end, x_start:x_end] = 0
    return volume, (z_start, z_end, y_start, y_end, x_start, x_end)

# Step 3: Interpolate the cropped region for each slice along the z-axis
def interpolate_2d_each_slice(volume, crop_bounds):
    z_start, z_end, y_start, y_end, x_start, x_end = crop_bounds

    # Loop over each z-slice
    for z in range(volume.shape[0]):
        # Extract the 2D slice in the xy-plane at index z
        slice_2d = volume[z, :, :]

        # Create a mask for the cropped region in this 2D slice
        mask = np.zeros_like(slice_2d)
        if z_start <= z < z_end:
            mask[y_start:y_end, x_start:x_end] = 1

        # Get coordinates of non-zero (valid) surrounding values
        valid_points = np.array(np.nonzero(mask == 0)).T  # Surrounding points (y, x)
        valid_values = slice_2d[mask == 0]  # Values at valid points

        # Coordinates of points to interpolate (inside the cropped region)
        points_to_interpolate = np.array(np.nonzero(mask == 1)).T

        if len(valid_points) > 0 and len(points_to_interpolate) > 0:
            # Perform interpolation
            interpolated_values = griddata(valid_points, valid_values, points_to_interpolate, method='cubic')

            # Assign interpolated values back to the slice
            for i, pt in enumerate(points_to_interpolate):
                slice_2d[pt[0], pt[1]] = interpolated_values[i]

        # Update the volume with the interpolated slice
        volume[z, :, :] = slice_2d

    return volume

# Step 4: Save the modified volume to a new .mrc file
def save_mrc_file(volume, output_filepath):
    with mrcfile.new(output_filepath, overwrite=True) as mrc:
        mrc.set_data(volume.astype(np.float32))

# Main function to load, crop, interpolate, and save the volume
def process_mrc_file(input_filepath, output_filepath, crop_size):
    volume = load_mrc_file(input_filepath)
    volume, crop_bounds = crop_center(volume, crop_size)
    volume = interpolate_2d_each_slice(volume, crop_bounds)
    save_mrc_file(volume, output_filepath)
    print(f"Processed volume saved to {output_filepath}")

# Example usage
f_ = [10, 15, 20, 50, 75, 100]
for frames in f_:
    input_mrc = f'reconstruct_{frames}.mrc'  
    output_mrc = f'reconstruct_{frames}_cropped.mrc'  

    crop_size = 9
    process_mrc_file(input_mrc, output_mrc, crop_size)