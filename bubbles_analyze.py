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
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Load the refractive index field (256*256*256 volume)
with mrcfile.open('ODT_experment_results/4gy_6.mrc', permissive=True) as mrc:
    volume = -mrc.data

# Define thresholds
cell_threshold = 0.03
vacuum_threshold = 0.025
n = 3

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
# Use the cell surface to define the internal region
internal_mask = morphology.binary_erosion(cell_mask_filled_largest, morphology.ball(1))

# Step 3: Identify the internal vacuum (vacuole) region inside the cell
internal_vacuum_mask = (volume <= vacuum_threshold) & internal_mask

# Function to split weakly connected regions using PCA and KMeans clustering
def split_weakly_connected_regions(vacuum_mask, n_clusters=2, thickness_threshold=3, small_region_threshold=100):
    
    dilated_mask = binary_dilation(vacuum_mask, selem=ball(2))
    eroded_mask = binary_erosion(dilated_mask, selem=ball(1))
    split_regions = measure.label(eroded_mask)
    final_regions = np.zeros_like(vacuum_mask, dtype=int)
    current_label = 1
    all_coordinates = []
    all_labels = []

    for region_label in range(1, split_regions.max() + 1):
        region_mask = split_regions == region_label

        # Apply distance transform to find narrow regions
        distance_map = distance_transform_edt(region_mask)
        min_thickness = np.min(distance_map[distance_map > 0])

        if min_thickness < thickness_threshold:
            # If thickness is below threshold, attempt splitting using KMeans
            coordinates = np.argwhere(region_mask)
            if len(coordinates) > n_clusters:

                gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(coordinates)
                labels = gmm.predict(coordinates)
                all_coordinates.append(coordinates)
                all_labels.append(labels)

                # Assign new labels to split components
                for cluster_id in range(n_clusters):
                    cluster_mask = np.zeros_like(region_mask, dtype=bool)
                    cluster_mask[tuple(coordinates[labels == cluster_id].T)] = True
                    if np.sum(cluster_mask) > small_region_threshold:
                        final_regions[cluster_mask] = current_label
                        current_label += 1
            else:
                # If not enough points for KMeans, keep the region if it is large enough
                if np.sum(region_mask) > small_region_threshold:
                    final_regions[region_mask] = current_label
                    current_label += 1
        else:
            # If thickness is above threshold, keep as is if it is large enough
            if np.sum(region_mask) > small_region_threshold:
                final_regions[region_mask] = current_label
                current_label += 1

    # Visualize KMeans clustering result
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for coordinates, labels in zip(all_coordinates, all_labels):
        for cluster_id in range(n_clusters):
            cluster_points = coordinates[labels == cluster_id]
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                        color=colors[cluster_id % len(colors)], s=1, label=f'Cluster {cluster_id}')
    ax2.set_title('KMeans Clustering of Vacuum Regions')
    plt.legend()
    plt.show()

    return final_regions

# Split weakly connected vacuum regions
split_vacuum_regions = split_weakly_connected_regions(internal_vacuum_mask, n_clusters=n)

# Ensure the internal vacuum mask has sufficient volume for processing
vacuum_region_props = measure.regionprops(split_vacuum_regions)

# Create an array to store each connected vacuum region
n_vacuums = len(vacuum_region_props)
print(f"Number of Internal Vacuum Regions: {n_vacuums}")
vacuum_masks = np.zeros((n_vacuums, *internal_vacuum_mask.shape), dtype=bool)

for idx, region in enumerate(vacuum_region_props):
    vacuum_masks[idx] = split_vacuum_regions == region.label

if len(vacuum_region_props) > 0:
    total_vacuum_volume = 0
    total_vacuum_surface_area = 0
    all_vacuum_meshes = []

    for idx, region in enumerate(vacuum_region_props):
        # Extract each connected vacuum region
        vacuum_mask = vacuum_masks[idx]

        # Extract the surface mesh using marching cubes
        vacuum_vertices, vacuum_faces, _, _ = measure.marching_cubes(vacuum_mask, level=0, allow_degenerate=False)
        vacuum_mesh = trimesh.Trimesh(vertices=vacuum_vertices, faces=vacuum_faces)

        # Smooth the vacuum mesh using Laplacian smoothing
        vacuum_mesh = filter_laplacian(vacuum_mesh, iterations=10)

        # Convert trimesh to Open3D mesh for further processing
        vacuum_o3d_mesh = o3d.geometry.TriangleMesh()
        vacuum_o3d_mesh.vertices = o3d.utility.Vector3dVector(vacuum_mesh.vertices)
        vacuum_o3d_mesh.triangles = o3d.utility.Vector3iVector(vacuum_mesh.faces)
        vacuum_o3d_mesh = vacuum_o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

        # Ensure the vacuum mesh is watertight for volume calculation
        vacuum_o3d_mesh.remove_duplicated_vertices()
        vacuum_o3d_mesh.remove_duplicated_triangles()
        vacuum_o3d_mesh.remove_degenerate_triangles()
        vacuum_o3d_mesh.remove_unreferenced_vertices()
        vacuum_o3d_mesh = vacuum_o3d_mesh.compute_convex_hull()[0]

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

        # Add to the list for visualization
        all_vacuum_meshes.append(vacuum_o3d_mesh)

        # Export each vacuum mesh to .obj file
        o3d.io.write_triangle_mesh(f'internal_vacuum_{idx}.obj', vacuum_o3d_mesh)

        # Calculate volume and surface area for each vacuum
        vacuum_volume = vacuum_o3d_mesh.get_volume()
        vacuum_surface_area = vacuum_o3d_mesh.get_surface_area()
        total_vacuum_volume += vacuum_volume
        total_vacuum_surface_area += vacuum_surface_area

        # Print the vacuum results
        print(f"Internal Vacuum {idx} Volume: {vacuum_volume:.3f} cubic units")
        print(f"Internal Vacuum {idx} Surface Area: {vacuum_surface_area:.3f} square units")

    # Print the total vacuum results
    print(f"Total Internal Vacuum Volume: {total_vacuum_volume:.3f} cubic units")
    print(f"Total Internal Vacuum Surface Area: {total_vacuum_surface_area:.3f} square units")

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
