import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from Scene import Scene
from mayavi import mlab


def viz_3d(pt_3d):
    X = pt_3d[0,:]
    Y = pt_3d[1,:]
    Z = pt_3d[2,:]

    mlab.points3d(
        X,   # x
        Y,   # y
        Z,   # z
        mode="point", # How to render each point {'point', 'sphere' , 'cube' }
        colormap='copper',  # 'bone', 'copper',
        line_width=10,
        scale_factor=1
    )
    mlab.axes(xlabel='x', ylabel='y', zlabel='z',ranges=(0,20,0,20,0,10),nb_labels=10)
    mlab.show()


scene_id = '000028'
scene = Scene(data_dir='/Users/alexvesel/Downloads/oncedataset', scene_id=scene_id)

points_3d_all = []
for i, frame_id in tqdm(enumerate(scene.frame_ids)):
    lidar = scene.get_lidar_frame(frame_id=frame_id)
    points_3d = lidar[:, :3].T
    pose = scene.get_pose(frame_id=frame_id)
    points_3d = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
    points_3d = pose @ points_3d
    points_3d /= points_3d[3]
    points_3d_all.append(points_3d[:3, ::100])
    # points_3d_all = np.concatenate((points_3d_all, points_3d), axis=1)
# viz_3d(points_3d_all)
points_3d_all = np.concatenate(points_3d_all, axis=1)
points_3d_all = points_3d_all.T
# save as csv
df = pd.DataFrame(points_3d_all)
df.to_csv(f'{scene_id}/points_3d.csv', index=False, header=False)

