# First scenario 3D PEdestrian detection bound boxing the straight walk 

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

# Clustering and Box the Purple Cluster with the Most Points in the Bounding Box
def cluster_and_save_largest_purple(cleaned_pcd, eps=0.3, min_points=10):

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

    """Find pedestrian based on its color, in this step i checked the color of the pedestrian in the 
    visualization and based on that all the purple clusters targeted."""

    purple_rgb = cmap(8 / 20)[:3]  # "tab20" color index for purple 

    # Find all purple clusters from the labels that DBSCAN set
    purple_clusters = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_sample_color = colors[cluster_indices[0]]
        if not np.allclose(cluster_sample_color[:3], purple_rgb, atol=0.05):  # Compare cluster color
            continue

        # collecting points formt he purple clusters for boxing
        cluster_pcd = cleaned_pcd.select_by_index(cluster_indices)
        purple_clusters.append((i, cluster_pcd))

    # finding purple cluster with the most points inside its bounding box
    # this is because the pedestriand clustering usually contian the most points
    most_points_inside_bbox = 0
    selected_cluster = None
    selected_bbox = None

    for cluster_id, cluster_pcd in purple_clusters:
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        indices_in_bbox = bbox.get_point_indices_within_bounding_box(cleaned_pcd.points)
        points_inside_bbox = len(indices_in_bbox)

        if points_inside_bbox > most_points_inside_bbox:
            most_points_inside_bbox = points_inside_bbox
            selected_cluster = cluster_pcd
            selected_bbox = bbox

    # Save points to CSV
    # this points will be used later as reference to other datasets 
    bbox_points = np.asarray(cleaned_pcd.points)[selected_bbox.get_point_indices_within_bounding_box(cleaned_pcd.points)]
    np.savetxt("largest_purple_cluster.csv", bbox_points, delimiter=",", header="x,y,z", comments="")

    selected_bbox.color = (1, 0, 0)  # Red bounding box
    return cleaned_pcd, [selected_bbox]

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Clusters After Noise Removal", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

file_path = "/Users/selma/Downloads/COSE416_HW1_tutorial/data/01_straight_walk/pcd/pcd_000288.pcd"
cleaned_pcd = clean_point_cloud(file_path)
cleaned_pcd, bboxes = cluster_and_save_largest_purple(cleaned_pcd, eps=0.3, min_points=10)
visualize_with_bounding_boxes(cleaned_pcd, bboxes, point_size=2.0)