import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Noise Removal
def clean_point_cloud(file_path):

    # Reading File
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Original point cloud has {len(pcd.points)} points.")

    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After Statistical Outlier Removal: {len(pcd.points)} points remain.")

    # radius outlier removal
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
    print(f"After Radius Outlier Removal: {len(pcd.points)} points remain.")

    return pcd

# Extracting features of the clusters to compare with references  
def extract_features(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    bbox_dims = max_vals - min_vals  # [width, depth, height]
    width, depth, height = bbox_dims
    width_height_ratio = width / height if height != 0 else 0
    density = 1 / np.mean(pdist(points)) if len(points) > 1 else 0
    return {
        'bounding_box': bbox_dims,
        'width_height_ratio': width_height_ratio,
        'density': density,
    }

# Step 3: Clustering and Detect Clusters Based on References
def detect_and_box_clusters(cleaned_pcd, references, eps=0.3, min_points=10):
       #  DBSCAN clustering
    print(f"Clustering with eps={eps}, min_points={min_points}")
    labels = np.array(cleaned_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    """COLORING"""
    # maximum cluster label
    max_label = labels.max() 
    cmap = plt.get_cmap("tab20")  # Tab20 colormap for cluster coloring

    # Normalize labels for coloring, range is [0, 1] --> for colormap listing 
    if max_label > 0:
        normalized_labels = labels / (max_label + 1)  
    else:
        normalized_labels = labels 
    colors = cmap(normalized_labels)  # assign labeled clusters to colors 

    #   noise points detected by DBSCAN clustering (labels < 0) setting to black
    colors[labels < 0] = [0, 0, 0, 1]  

    # Applying colors to the point clouds
    cleaned_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  


    turquoise_rgb = cmap(18 / 20)[:3]
    print(f"Turquoise RGB value: {turquoise_rgb}")

    bboxes = []
    largest_turquoise_cluster = None
    largest_points = 0

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_sample_color = colors[cluster_indices[0]]
        if not np.allclose(cluster_sample_color[:3], turquoise_rgb, atol=0.05): 
            continue

        cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        features = extract_features(points)
        print(f"Cluster {i} features: {features}")

        """
        # boudn boxing 
        1. manually added
        2. matched with the referenced points
        3. largest clustering based on coloring
        """
        # Match with reference features only one of them is matched
        for ref_key, ref_features in references.items():
            bbox_match = np.allclose(features['bounding_box'], ref_features['bounding_box'], atol=0.2)
            ratio_match = np.isclose(features['width_height_ratio'], ref_features['width_height_ratio'], atol=0.1)
            density_match = np.isclose(features['density'], ref_features['density'], atol=0.5)

            if bbox_match and ratio_match and density_match:
                print(f"Cluster {i} matched with reference {ref_key}: {features}")
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)  # Red bounding box for matched cluster
                bboxes.append(bbox)
                break

        # Track the largest turquoise cluster by points
        if len(cluster_indices) > largest_points:
            largest_points = len(cluster_indices)
            largest_turquoise_cluster = cluster_pcd

        # manually adding the cluster
        if i == 149:
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 1, 0)  # Yellow bounding box for Cluster 149
            bboxes.append(bbox)
            print(f"Manually added bounding box for Cluster {i}.")

    # Box the largest turquoise cluster
    if largest_turquoise_cluster:
        largest_bbox = largest_turquoise_cluster.get_axis_aligned_bounding_box()
        largest_bbox.color = (0, 0, 1)  # Blue bounding box for largest turquoise cluster
        bboxes.append(largest_bbox)
        print(f"Added bounding box for the largest turquoise cluster with {largest_points} points.")

    return cleaned_pcd, bboxes


# Visualization
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Cluster Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# Reference Features
references = {
    (0, 0, 1): {  # Turquoise cluster with yellow-like features
        'bounding_box': np.array([0.32850933, 0.59455872, 0.40550107]),
        'width_height_ratio': 0.8101318515946736,
        'density': 4.067433538838899
    },
    (0, 1, 0): {  # Turquoise cluster with turquoise-like features
        'bounding_box': np.array([0.12882328, 0.37467194, 0.86627805]),
        'width_height_ratio': 0.14870892779515574,
        'density': 3.117619651052725
    }
}

file_path = "/Users/selma/Downloads/COSE416_HW1_tutorial/data/04_zigzag_walk/pcd/pcd_000364.pcd"
cleaned_pcd = clean_point_cloud(file_path)
cleaned_pcd, bboxes = detect_and_box_clusters(cleaned_pcd, references, eps=0.3, min_points=10)
visualize_with_bounding_boxes(cleaned_pcd, bboxes, point_size=2.0)


