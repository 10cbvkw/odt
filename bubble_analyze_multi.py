import numpy as np
import mrcfile
import skimage.measure as measure
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from skimage import morphology
from skimage.morphology import ball, binary_dilation, binary_erosion
from trimesh.smoothing import filter_laplacian
from scipy.ndimage import distance_transform_edt, label
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load the refractive index field (256*256*256 volume)
with mrcfile.open('ODT_experment_results/2gy_7.mrc', permissive=True) as mrc:
    volume = -mrc.data

# Define thresholds
cell_threshold = 0.03
vacuum_threshold = 0.025

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
cell_o3d_mesh = cell_o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

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
internal_mask = morphology.binary_erosion(cell_mask_filled_largest, morphology.ball(1))

# Step 3: Identify the internal vacuum (vacuole) region inside the cell
internal_vacuum_mask = (volume <= vacuum_threshold) & internal_mask
labeled_grid, num_features = label(internal_vacuum_mask)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
voxel_points = np.argwhere(internal_vacuum_mask)
ax.scatter(voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2], s=1, color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

total_surface_area = 0
total_volume = 0
all_vacuum_meshes = []

for component_idx in range(1, num_features + 1):
    
    component_voxels = (labeled_grid == component_idx)
    verts, faces, _, _ = measure.marching_cubes(component_voxels, level=0, allow_degenerate=False)
    vacuum_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Smooth the vacuum mesh using Laplacian smoothing
    vacuum_mesh = filter_laplacian(vacuum_mesh, iterations=10)

    # Convert trimesh to Open3D mesh for further processing
    vacuum_o3d_mesh = o3d.geometry.TriangleMesh()
    vacuum_o3d_mesh.vertices = o3d.utility.Vector3dVector(vacuum_mesh.vertices)
    vacuum_o3d_mesh.triangles = o3d.utility.Vector3iVector(vacuum_mesh.faces)
    vacuum_o3d_mesh = vacuum_o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

    # Visualize Smoothed and Reduced Internal Vacuum Surface Mesh
    fig2 = plt.figure(figsize=(5, 5))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_trisurf(np.asarray(vacuum_o3d_mesh.vertices)[:, 0],
                    np.asarray(vacuum_o3d_mesh.vertices)[:, 1],
                    np.asarray(vacuum_o3d_mesh.triangles),
                    np.asarray(vacuum_o3d_mesh.vertices)[:, 2],
                    linewidth=0.2, antialiased=True)
    ax2.set_title('Internal Vacuum Surface Mesh (Smoothed and Reduced)')
    plt.show()

    # Calculate volume and surface area for each vacuum
    try:
        vacuum_volume = vacuum_o3d_mesh.get_volume()
        vacuum_surface_area = vacuum_o3d_mesh.get_surface_area()
    except:
        vacuum_volume = -1
        vacuum_surface_area = -1

    if vacuum_volume == -1:
        vacuum_o3d_mesh.remove_duplicated_vertices()
        vacuum_o3d_mesh.remove_duplicated_triangles()
        vacuum_o3d_mesh.remove_degenerate_triangles()
        vacuum_o3d_mesh.remove_unreferenced_vertices()
        vacuum_o3d_mesh = vacuum_o3d_mesh.compute_convex_hull()[0]
        vacuum_volume = vacuum_o3d_mesh.get_volume()
        vacuum_surface_area = vacuum_o3d_mesh.get_surface_area()
    
    if vacuum_volume > 200:

        total_volume += vacuum_volume
        total_surface_area += vacuum_surface_area

        # Add to the list for visualization
        all_vacuum_meshes.append(vacuum_o3d_mesh)

        # Export each vacuum mesh to .obj file
        o3d.io.write_triangle_mesh(f'internal_vacuum_{component_idx}.obj', vacuum_o3d_mesh)

        # Print the vacuum results
        print(f"Internal Vacuum {component_idx} Volume: {vacuum_volume:.3f} cubic units")
        print(f"Internal Vacuum {component_idx} Surface Area: {vacuum_surface_area:.3f} square units")

    else:
        print(f"Internal Vacuum {component_idx} is too small to analyze")

# Print the total vacuum results
print(f"Total Internal Vacuum Volume: {total_volume:.3f} cubic units")
print(f"Total Internal Vacuum Surface Area: {total_surface_area:.3f} square units")

fig3 = plt.figure(figsize=(5, 5))
ax3 = fig3.add_subplot(111, projection='3d')
for vacuum_mesh in all_vacuum_meshes:
    ax3.plot_trisurf(np.asarray(vacuum_mesh.vertices)[:, 0],
                    np.asarray(vacuum_mesh.vertices)[:, 1],
                    np.asarray(vacuum_mesh.triangles),
                    np.asarray(vacuum_mesh.vertices)[:, 2],
                    linewidth=0.2, antialiased=True, color='blue', alpha=0.5)
ax3.plot_trisurf(np.asarray(cell_o3d_mesh.vertices)[:, 0],
                np.asarray(cell_o3d_mesh.vertices)[:, 1],
                np.asarray(cell_o3d_mesh.triangles),
                np.asarray(cell_o3d_mesh.vertices)[:, 2],
                linewidth=0.2, antialiased=True, color='yellow', alpha=0.2)
ax3.set_title('cell and vacuum (Smoothed and Reduced)')
plt.show()