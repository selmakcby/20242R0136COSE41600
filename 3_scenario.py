import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

# Clustering and Detecting Clusters by Color
def cluster_and_save_colored_clusters(cleaned_pcd, eps=0.3, min_points=10):
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

    # this time fromt he scenario pedestrians color is light blue and green
    # 
    light_blue_rgb = cmap(1 / 20)[:3]  # "tab20" color index for light blue
    green_rgb = cmap(5 / 20)[:3]       # "tab20" color index for green

    light_blue_candidates = []
    green_candidates = []

    for i in range(max_label + 1): #same logic from scenario 1 
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_sample_color = colors[cluster_indices[0]]

        if np.allclose(cluster_sample_color[:3], light_blue_rgb, atol=0.05): #for blue
            cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
            light_blue_candidates.append((i, cluster_pcd))

        # green ones 
        elif np.allclose(cluster_sample_color[:3], green_rgb, atol=0.05):
            cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
            green_candidates.append((i, cluster_pcd))

    # largest light blue cluster by number of points
    light_blue_clusters = []
    if light_blue_candidates:
        largest_light_blue = max(light_blue_candidates, key=lambda x: len(x[1].points))
        light_blue_clusters.append(largest_light_blue)

    # largest green cluster by number of points
    green_clusters = []
    if green_candidates:
        largest_green = max(green_candidates, key=lambda x: len(x[1].points))
        green_clusters.append(largest_green)

    # Adifferetn colors for eaxch box 
    # For crawl walking
    bboxes = []
    for _, cluster_pcd in light_blue_clusters:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 1, 1)  # Cyan bounding box for light blue
        bboxes.append(bbox)

    # for straight walking
    for _, cluster_pcd in green_clusters:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 1, 0)  # Green bounding box
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

# Main Workflow 06_straight_crawl/pcd/pcd_000774.pcd
file_path = "03_straight_crawl/pcd/pcd_001251.pcd"
cleaned_pcd = clean_point_cloud(file_path)
cleaned_pcd, bboxes = cluster_and_save_colored_clusters(cleaned_pcd, eps=0.3, min_points=10)
visualize_with_bounding_boxes(cleaned_pcd, bboxes, point_size=2.0)
