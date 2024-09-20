import tifffile as tiff
import mrcfile
import os
import numpy as np

def tif_files_generator(directory):
    # List all TIFF files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    files.sort()  # Sorting to maintain the order

    # Yield each image one at a time
    for file in files:
        image_path = os.path.join(directory, file)
        yield tiff.imread(image_path)

def save_stack_as_mrcs(directory, output_file):
    # Initialize the generator
    generator = tif_files_generator(directory)

    # Use the first image to determine shape and data type
    first_image = next(generator)
    stack_shape = (len(os.listdir(directory)), *first_image.shape)

    # Create an MRC file with the stack shape and dtype of the first image
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(np.zeros(stack_shape, dtype=first_image.dtype))

        # Set the first image
        mrc.data[0, :, :] = first_image

        # Load and set each subsequent image
        for i, image in enumerate(generator, start=1):
            mrc.data[i, :, :] = image

# Path to the directory containing TIFF files
directory = '/home/pc/Desktop/ODT_cryodrgn_data/10/cellphase'

# Path where the .mrcs file will be saved
output_file = 'cellphase_10.mrcs'

# Process the images
save_stack_as_mrcs(directory, output_file)

print("Stack has been saved as .mrcs file.")