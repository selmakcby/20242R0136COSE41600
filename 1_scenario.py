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

