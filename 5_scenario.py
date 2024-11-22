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

# Clustering adn color detection 
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

    #  RGB values for yellow and turquoise
    yellow_rgb = cmap(17 / 20)[:3]  # "tab20" color index for yellow
    turquoise_rgb = cmap(18 / 20)[:3]  # "tab20" color index for turquoise

    yellow_candidates = []
    turquoise_candidates = []

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_sample_color = colors[cluster_indices[0]]

        #  yellow clusters -> duck walk
        if np.allclose(cluster_sample_color[:3], yellow_rgb, atol=0.05):
            cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox_height = bbox.get_extent()[2]
            if bbox_height <= 0.5:  # Height condition for yellow
                yellow_candidates.append((i, cluster_pcd, bbox, len(cluster_indices)))

        #  turquoise clusters -> straight walking
        elif np.allclose(cluster_sample_color[:3], turquoise_rgb, atol=0.05):
            cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox_height = bbox.get_extent()[2]
            bbox_width = bbox.get_extent()[0]
            if 0.5 <= bbox_height <= 1.0 and bbox_width <= 0.3:  # Height and width conditions for turquoise
                turquoise_candidates.append((i, cluster_pcd, bbox, len(cluster_indices)))

    """finding largest point clouds for both colors"""
    yellow_clusters = []
    if yellow_candidates:
        largest_yellow = max(yellow_candidates, key=lambda x: x[3])
        yellow_clusters.append(largest_yellow)

    turquoise_clusters = []
    if turquoise_candidates:
        largest_turquoise = max(turquoise_candidates, key=lambda x: x[3])
        turquoise_clusters.append(largest_turquoise)

    # Save points to CSV, will be used for scenario 4 as reference points
    if yellow_clusters:
        yellow_points = np.asarray(cleaned_pcd.points)[
            yellow_clusters[0][2].get_point_indices_within_bounding_box(cleaned_pcd.points)
        ]
        np.savetxt("yellow_cluster.csv", yellow_points, delimiter=",", header="x,y,z", comments="")
        print("Saved yellow cluster points to 'yellow_cluster.csv'.")

    if turquoise_clusters:
        turquoise_points = np.asarray(cleaned_pcd.points)[
            turquoise_clusters[0][2].get_point_indices_within_bounding_box(cleaned_pcd.points)
        ]
        np.savetxt("turquoise_cluster.csv", turquoise_points, delimiter=",", header="x,y,z", comments="")
        print("Saved turquoise cluster points to 'turquoise_cluster.csv'.")

    # Visualization
    bboxes = []
    for _, _, bbox, _ in yellow_clusters:
        bbox.color = (1, 0, 0)  # Red for yellow
        bboxes.append(bbox)

    for _, _, bbox, _ in turquoise_clusters:
        bbox.color = (0, 0, 1)  # Blue for turquoise
        bboxes.append(bbox)

    return cleaned_pcd, bboxes

# Visualization
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Clusters with Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

file_path = "/Users/selma/Downloads/COSE416_HW1_tutorial/data/05_straight_duck_walk/pcd/pcd_000577.pcd"
cleaned_pcd = clean_point_cloud(file_path)
cleaned_pcd, bboxes = cluster_and_save_colored_clusters(cleaned_pcd, eps=0.3, min_points=10)
visualize_with_bounding_boxes(cleaned_pcd, bboxes, point_size=2.0)
