import numpy as np
import mrcfile
import skimage.measure as measure
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from trimesh.smoothing import filter_laplacian

# Load the refractive index field (256*256*256 volume)
with mrcfile.open('reconstruct.mrc', permissive=True) as mrc:
    volume = -mrc.data

# Define thresholds
cell_threshold = 0.01
vacuum_threshold = 0.005

# Step 1: Identify the cell region
cell_mask = volume > cell_threshold

# Fill holes to make sure the cell is treated as a solid object
cell_mask_filled = binary_fill_holes(cell_mask)

# Extract the largest connected component for the cell
cell_regions = measure.label(cell_mask_filled)
region_props = measure.regionprops(cell_regions)
largest_region = max(region_props, key=lambda r: r.area)
cell_mask_filled_largest = cell_regions == largest_region.label

# Extract the largest cell surface mesh using marching cubes
cell_vertices, cell_faces, _, _ = measure.marching_cubes(cell_mask_filled_largest, level=0, allow_degenerate=False)
cell_mesh = trimesh.Trimesh(vertices=cell_vertices, faces=cell_faces)

# Smooth the cell mesh using Laplacian smoothing
cell_mesh = filter_laplacian(cell_mesh, iterations=10)

# Convert trimesh to Open3D mesh for further processing
cell_o3d_mesh = o3d.geometry.TriangleMesh()
cell_o3d_mesh.vertices = o3d.utility.Vector3dVector(cell_mesh.vertices)
cell_o3d_mesh.triangles = o3d.utility.Vector3iVector(cell_mesh.faces)
cell_o3d_mesh = cell_o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

# Ensure the cell mesh is watertight for volume calculation
cell_o3d_mesh.remove_duplicated_vertices()
cell_o3d_mesh.remove_duplicated_triangles()
cell_o3d_mesh.remove_degenerate_triangles()
cell_o3d_mesh.remove_unreferenced_vertices()
cell_o3d_mesh = cell_o3d_mesh.compute_convex_hull()[0]

# Visualize the Smoothed and Reduced Cell Surface Mesh
fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_trisurf(np.asarray(cell_o3d_mesh.vertices)[:, 0],
                 np.asarray(cell_o3d_mesh.vertices)[:, 1],
                 np.asarray(cell_o3d_mesh.triangles),
                 np.asarray(cell_o3d_mesh.vertices)[:, 2],
                 linewidth=0.2, antialiased=True)
ax1.set_title('Cell Surface Mesh (Smoothed and Reduced)')
plt.show()

# Export the cell mesh to .obj file
o3d.io.write_triangle_mesh('cell_surface.obj', cell_o3d_mesh)

# Calculate volume and surface area for the cell
cell_volume = cell_o3d_mesh.get_volume()
cell_surface_area = cell_o3d_mesh.get_surface_area()

# Print the cell results
print(f"Cell Volume: {cell_volume:.3f} cubic units")
print(f"Cell Surface Area: {cell_surface_area:.3f} square units")

# Step 2: Identify the internal region of the cell
# Use the cell surface to define the internal region
internal_mask = morphology.binary_erosion(cell_mask_filled_largest, morphology.ball(1))

# Step 3: Identify the internal vacuum (vacuole) region inside the cell
internal_vacuum_mask = (volume <= vacuum_threshold) & internal_mask

# Ensure the internal vacuum mask has sufficient volume for processing
if np.sum(internal_vacuum_mask) > 0:
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

    # Convert trimesh to Open3D mesh for further processing
    internal_vacuum_o3d_mesh = o3d.geometry.TriangleMesh()
    internal_vacuum_o3d_mesh.vertices = o3d.utility.Vector3dVector(internal_vacuum_mesh.vertices)
    internal_vacuum_o3d_mesh.triangles = o3d.utility.Vector3iVector(internal_vacuum_mesh.faces)
    internal_vacuum_o3d_mesh = internal_vacuum_o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

    # Ensure the internal vacuum mesh is watertight for volume calculation
    internal_vacuum_o3d_mesh.remove_duplicated_vertices()
    internal_vacuum_o3d_mesh.remove_duplicated_triangles()
    internal_vacuum_o3d_mesh.remove_degenerate_triangles()
    internal_vacuum_o3d_mesh.remove_unreferenced_vertices()
    internal_vacuum_o3d_mesh = internal_vacuum_o3d_mesh.compute_convex_hull()[0]

    # Visualize Smoothed and Reduced Internal Vacuum Surface Mesh
    fig2 = plt.figure(figsize=(5, 5))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_trisurf(np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 0],
                     np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 1],
                     np.asarray(internal_vacuum_o3d_mesh.triangles),
                     np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 2],
                     linewidth=0.2, antialiased=True)
    ax2.set_title('Internal Vacuum Surface Mesh (Smoothed and Reduced)')
    plt.show()

    # Export the internal vacuum mesh to .obj file
    o3d.io.write_triangle_mesh('internal_vacuum_surface.obj', internal_vacuum_o3d_mesh)

    # Calculate volume and surface area for the internal vacuum
    internal_vacuum_volume = internal_vacuum_o3d_mesh.get_volume()
    internal_vacuum_surface_area = internal_vacuum_o3d_mesh.get_surface_area()

    # Print the internal vacuum results
    print(f"Internal Vacuum Volume: {internal_vacuum_volume:.3f} cubic units")
    print(f"Internal Vacuum Surface Area: {internal_vacuum_surface_area:.3f} square units")
else:
    print("No significant internal vacuum region found.")


fig3 = plt.figure(figsize=(5, 5))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_trisurf(np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 0],
                    np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 1],
                    np.asarray(internal_vacuum_o3d_mesh.triangles),
                    np.asarray(internal_vacuum_o3d_mesh.vertices)[:, 2],
                    linewidth=0.2, antialiased=True, color='blue', alpha=0.5)
ax3.plot_trisurf(np.asarray(cell_o3d_mesh.vertices)[:, 0],
                np.asarray(cell_o3d_mesh.vertices)[:, 1],
                np.asarray(cell_o3d_mesh.triangles),
                np.asarray(cell_o3d_mesh.vertices)[:, 2],
                linewidth=0.2, antialiased=True, color='yellow', alpha=0.2)
ax3.set_title('cell and vacuum (Smoothed and Reduced)')
plt.show()