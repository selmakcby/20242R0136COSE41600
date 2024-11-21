import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "/Users/selma/Downloads/COSE416_HW1_tutorial/data/01_straight_walk/pcd/pcd_000001.pcd"

# pcd 파일 불러오고 시각화하는 함수
def load_and_visualize_pcd(file_path, point_size=1.0):
    # pcd 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud has {len(pcd.points)} points.")
    
    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


# pcd 시각화 테스트
load_and_visualize_pcd(file_path, 0.5)



