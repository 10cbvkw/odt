import numpy as np
import mrcfile
import skimage.measure as measure
import trimesh
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from trimesh.smoothing import filter_laplacian
from trimesh import remesh

# Load the refractive index field (256*256*256 volume)
with mrcfile.open('reconstruct.mrc', permissive=True) as mrc:
    volume = -mrc.data

# Define thresholds
cell_threshold = 0.01

# Step 1: Identify the cell region
cell_mask = volume > cell_threshold

# Fill holes to make sure the cell is treated as a solid object
cell_mask_filled = binary_fill_holes(cell_mask)

# Extract all cell surface meshes using marching cubes
all_cell_meshes = measure.marching_cubes(cell_mask_filled, level=0, allow_degenerate=False)

# Find the largest connected component for the cell
cell_regions = measure.label(cell_mask_filled)
region_props = measure.regionprops(cell_regions)
largest_region = max(region_props, key=lambda r: r.area)
cell_mask_filled_largest = cell_regions == largest_region.label

# Extract the largest cell surface mesh using marching cubes
cell_vertices, cell_faces, _, _ = measure.marching_cubes(cell_mask_filled_largest, level=0, allow_degenerate=False)
cell_mesh = trimesh.Trimesh(vertices=cell_vertices, faces=cell_faces)

# Smooth the cell mesh using Laplacian smoothing
cell_mesh = filter_laplacian(cell_mesh, iterations=10)

# Reduce the number of vertices in the cell mesh
#cell_mesh = cell_mesh.simplify_quadratic_decimation(5000)

# Visualize Smoothed and Reduced Cell Surface Mesh
fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(cell_mesh.vertices[:, 0], cell_mesh.vertices[:, 1], cell_mesh.faces, cell_mesh.vertices[:, 2], linewidth=0.2, antialiased=True)
ax1.set_title('Cell Surface Mesh (Smoothed and Reduced)')
plt.show()

# Step 2: Identify the internal region of the cell
# Use the cell surface to define the internal region
internal_mask = morphology.binary_erosion(cell_mask_filled_largest, morphology.ball(1))
internal_mask &= ~cell_mask_filled_largest  # Ensure only internal voxels are selected

# Step 3: Identify the internal vacuum (vacuole) region inside the cell
internal_vacuum_mask = (volume <= cell_threshold) & internal_mask

# Extract all internal vacuum surface meshes using marching cubes
all_internal_vacuum_meshes = measure.marching_cubes(internal_vacuum_mask, level=0, allow_degenerate=False)

# Find the largest connected component for the internal vacuum
vacuum_regions = measure.label(internal_vacuum_mask)
vacuum_region_props = measure.regionprops(vacuum_regions)
largest_vacuum_region = max(vacuum_region_props, key=lambda r: r.area)
internal_vacuum_mask_largest = vacuum_regions == largest_vacuum_region.label

# Extract the largest internal vacuum surface mesh using marching cubes
internal_vacuum_vertices, internal_vacuum_faces, _, _ = measure.marching_cubes(internal_vacuum_mask_largest, level=0, allow_degenerate=False)
internal_vacuum_mesh = trimesh.Trimesh(vertices=internal_vacuum_vertices, faces=internal_vacuum_faces)

# Smooth the internal vacuum mesh using Laplacian smoothing
internal_vacuum_mesh = filter_laplacian(internal_vacuum_mesh, iterations=10)

# Reduce the number of vertices in the internal vacuum mesh
# internal_vacuum_mesh = internal_vacuum_mesh.simplify_quadratic_decimation(5000)

# Visualize Smoothed and Reduced Internal Vacuum Surface Mesh
fig2 = plt.figure(figsize=(5, 5))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_trisurf(internal_vacuum_mesh.vertices[:, 0], internal_vacuum_mesh.vertices[:, 1], internal_vacuum_mesh.faces, internal_vacuum_mesh.vertices[:, 2], linewidth=0.2, antialiased=True)
ax2.set_title('Internal Vacuum Surface Mesh (Smoothed and Reduced)')
plt.show()

# Export the meshes to .obj files
cell_mesh.export('cell_surface.obj')
internal_vacuum_mesh.export('internal_vacuum_surface.obj')

# Calculate volume and surface area for cell and internal vacuum
cell_volume = cell_mesh.volume
cell_surface_area = cell_mesh.area

internal_vacuum_volume = internal_vacuum_mesh.volume
internal_vacuum_surface_area = internal_vacuum_mesh.area

# Print the results
print(f"Cell Volume: {cell_volume:.3f} cubic units")
print(f"Cell Surface Area: {cell_surface_area:.3f} square units")
print(f"Internal Vacuum Volume: {internal_vacuum_volume:.3f} cubic units")
print(f"Internal Vacuum Surface Area: {internal_vacuum_surface_area:.3f} square units")