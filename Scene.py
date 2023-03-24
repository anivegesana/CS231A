import numpy as np
import pandas as pd
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Scene():
    def __init__(self, data_dir, scene_id):
        self.scene_id = scene_id
        self.data_dir = os.path.join(data_dir, 'scenes', self.scene_id)
        self.metadata = json.loads(open(os.path.join(self.data_dir, f"{self.scene_id}.json")).read())

        self.weather = self.metadata["meta_info"]['weather']
        self.period = self.metadata["meta_info"]['period']
        self.image_size = self.metadata["meta_info"]['image_size']

        self.cams = list(self.metadata["calib"].keys())
        self.poses = np.array([self.metadata["frames"][i]['pose'] for i in range(len(self.metadata["frames"]))])
        self.frame_ids = np.array([int(self.metadata["frames"][i]['frame_id']) for i in range(len(self.metadata["frames"]))])


    def get_pose(self, frame_id):
        frame_idx = np.where(self.frame_ids == frame_id)[0][0]
        pose_vec = self.poses[frame_idx]
        pose = R.from_quat(pose_vec[:4]).as_matrix()
        pose = np.hstack((pose, np.zeros((3, 1))))
        pose[:3, 3] = pose_vec[4:]
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        return pose


    def get_camera_pose(self, cam_id):
        cam_pose = np.array(self.metadata["calib"][cam_id]["cam_to_velo"]) # 4x4
        return cam_pose


    def get_k(self, cam_id):
        return np.array(self.metadata["calib"][cam_id]["cam_intrinsic"]).reshape(3, 3)

    
    def get_lidar_frame(self, frame_id):
        lidar_file = os.path.join(self.data_dir, 'lidar_roof', f"{frame_id}.bin")
        lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return lidar_points


    def get_cam_frame(self, frame_id, cam_id):
        cam_file = os.path.join(self.data_dir, cam_id, f"{frame_id}.jpg")
        cam_image = np.array(Image.open(cam_file))
        return cam_image


    def plot_lidar_frame(self, frame_id):
        lidar_points = self.get_lidar_frame(frame_id)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], c=lidar_points[:, 3], cmap='gray')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()




if __name__ == "__main__":
    scene = Scene(scene_id='000027')
    # scene.plot_lidar_frame(frame_id=scene.frame_ids[0])
    scene.get_cam_frame(frame_id=scene.frame_ids[0], cam_id=scene.cams[0])