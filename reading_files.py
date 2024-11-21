import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "/Users/selma/Downloads/COSE416_HW1_tutorial/data/07_straight_walk/pcd/pcd_000443.pcd"


# pcd 파일 불러오고 시각화하는 함수
def load_and_visualize_pcd(file_path, point_size=1.0):
    # pcd 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud has {len(pcd.points)} points.")

    # Applying three noise reduction methods 
    # Statistical Outlier Removal (SOR)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After SOR: {len(pcd.points)} points remain.")

    # Radius Outlier Removal (ROR)
    pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.5)
    print(f"After ROR: {len(pcd.points)} points remain.")

    # RANSAC for ground plane removal
    plane_model, ground_points = pcd.segment_plane(distance_threshold=0.1,  # Threshold for a point to be part of the plane
                                             ransac_n=3,             # Number of points to sample for plane fitting
                                             num_iterations=1000)    # Number of RANSAC iterations
    print(f"Plane model: {plane_model}")
    
    # Separate ground and non-ground points
    ground = pcd.select_by_index(ground_points)  # Ground points
    non_ground = pcd.select_by_index(ground_points, invert=True)  # Non-ground points
    print(f"Ground points: {len(ground.points)}")
    print(f"Non-ground points: {len(non_ground.points)}")

    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(non_ground)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# pcd 시각화 테스트
load_and_visualize_pcd(file_path, 0.5)




