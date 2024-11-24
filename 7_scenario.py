#seventh scenario
#references from scenario 1 is used. 

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Clenaing point cloud 
# Noise removing & Dwonsampling 
def clean_point_cloud(file_path):
    
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Original point cloud has {len(pcd.points)} points.")

    # Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f"After Statistical Outlier Removal: {len(pcd.points)} points remain.")

    # Radius Outlier Removal
    pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.5)
    print(f"After Radius Outlier Removal: {len(pcd.points)} points remain.")

    # Downsampling (Voxel Grid)
    voxel_size = 0.05 
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"After Downsampling: {len(pcd.points)} points remain.")

    return pcd

# Extracting Features from clusters
# this will be used to to compare cluster features with the reference 
#reference is taken from the scenario 1 
def extract_features(points):

    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    bbox_dims = max_vals - min_vals  # [width, depth, height]
    width, depth, height = bbox_dims
    width_height_ratio = width / height if height != 0 else 0
    depth_height_ratio = depth / height if height != 0 else 0
    pairwise_distances = distance.pdist(points)
    density = 1 / (np.mean(pairwise_distances) if np.mean(pairwise_distances) != 0 else 1)

    return {
        'bounding_box': bbox_dims,
        'width_height_ratio': width_height_ratio,
        'depth_height_ratio': depth_height_ratio,
        'density': density,
    }

# based on the reference, detecting the similar clusters 
# Additionally cluster that has the most point data 
def detect_largest_similar_cluster(cleaned_pcd, eps=0.3, min_points=10, 
                                   width_height_range=(0.1, 1.0), density_range=(1.0, 4.0), 
                                   height_range=(1.5, 2.0)): #ranges desided based on the reference
    
    labels = np.array(cleaned_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()

    similar_clusters = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        cluster_features = extract_features(points)

        """cluster_features:
          return {
        'bounding_box': bbox_dims,
        'width_height_ratio': width_height_ratio,
        'depth_height_ratio': depth_height_ratio,
        'density': density,
    }"""
        # Matching cluster features to reference features
        width_height_match = width_height_range[0] <= cluster_features['width_height_ratio'] <= width_height_range[1]
        density_match = density_range[0] <= cluster_features['density'] <= density_range[1]
        height_match = height_range[0] <= cluster_features['bounding_box'][2] <= height_range[1] 

        #Printing the mathed points features 
        if width_height_match and density_match and height_match:
            similar_clusters.append((i, cluster_pcd, cluster_features, len(points)))
            print(f"Cluster {i} matched: {cluster_features} with {len(points)} points") 

    # Largest cluster points
    if similar_clusters:
        largest_cluster = max(similar_clusters, key=lambda x: x[3])  
        print(f"Largest matched cluster is {largest_cluster[0]} with {largest_cluster[3]} points.")
        return [largest_cluster]

#Visualizeing
def visualize_with_clusters(cleaned_pcd, clusters):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Matched Largest Cluster")
    vis.add_geometry(cleaned_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 1.0 
    
    for _, cluster_pcd, _, _ in clusters:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Red bounding box
        vis.add_geometry(bbox)

    
    vis.run()
    vis.destroy_window()

    
    """  
    from scene_1_boxed_cluster_points, points are analyzed: 

    reference_features = {'bounding_box': np.array([0.63314719, 0.35928154, 0.8702724]), #
                        'width_height_ratio': 0.7275276068048021, # range 0.1 to 1.0   width_height_range=(0.1, 1.0)
                        'density': 2.739238944946235} # density range 1 to 4  density_range=(1.0, 4.0)

    """

file_path = "07_straight_walk/pcd/pcd_000443.pcd"
cleaned_pcd = clean_point_cloud(file_path)

# detecting the largest cluster that matches the reference features and realistic human dimensions
largest_cluster = detect_largest_similar_cluster(
    cleaned_pcd,
    eps=0.3,
    min_points=10,
    width_height_range=(0.1, 1.0),  
    density_range=(1.0, 4.0),       
    height_range=(0.5, 1.0)        
)

visualize_with_clusters(cleaned_pcd, largest_cluster)
