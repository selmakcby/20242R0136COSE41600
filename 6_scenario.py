import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Noise Removal
def clean_point_cloud(file_path):

    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Original point cloud has {len(pcd.points)} points.")

    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After Statistical Outlier Removal: {len(pcd.points)} points remain.")

    # radius outlier removal
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
    print(f"After Radius Outlier Removal: {len(pcd.points)} points remain.")

    return pcd

# Clustering and Detect Two Largest Yellow Clusters withing the height we set 
def cluster_and_save_yellow_clusters(cleaned_pcd, eps=0.3, min_points=10, height_range=(0.5, 1.0)): 
    # DBSCAN clustering
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

    # Identify the RGB value for yellow
    yellow_rgb = cmap(17 / 20)[:3]  # "tab20" color index for yellow
    print(f"Yellow RGB value: {yellow_rgb}")

    # In this case pedestrians labels are yellow
    # finding all the clusters colored yellow
    yellow_candidates = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_sample_color = colors[cluster_indices[0]]
        if not np.allclose(cluster_sample_color[:3], yellow_rgb, atol=0.05): 
            continue

        # getting the points belong to yellow clusterings 
        cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox_dims = bbox.get_extent()

        """1. Filter by height range, this case 0.5 to 1.0
           2. Find the largest """

        # Filtering clusters by height range
        if height_range[0] <= bbox_dims[2] <= height_range[1]:  # Height range=(0.5, 1.0) --> this is based on the data 
            yellow_candidates.append((i, cluster_pcd, bbox_dims))

    # Select the two largest yellow clusters by number of points
    # The poinst int he clusters that belong to pedestians are most likely the largerst ones 
    yellow_candidates = sorted(yellow_candidates, key=lambda x: len(x[1].points), reverse=True)[:2] # two largest boxes
   
    bboxes = []
    for cluster_id, cluster_pcd, _ in yellow_candidates:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Red bounding box
        bboxes.append(bbox)
    return cleaned_pcd, bboxes

# Visualization
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Clusters After Noise Removal", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# Main Workflow
file_path = "06_straight_crawl/pcd/pcd_000774.pcd"
cleaned_pcd = clean_point_cloud(file_path)
cleaned_pcd, bboxes = cluster_and_save_yellow_clusters(cleaned_pcd, eps=0.3, min_points=10, height_range=(0.5, 1.0))
visualize_with_bounding_boxes(cleaned_pcd, bboxes, point_size=2.0)

