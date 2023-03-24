import numpy as np
import pandas as pd
import cv2
import os

from Scene import Scene

from mayavi import mlab
from tqdm import tqdm

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

# get pairs of frames and use normalized cross correlation to find corresponding points
def sfm(cam_idx, heuristic, estimate_E):
    orb = cv2.ORB_create()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))

    cam_id = scene.cams[cam_idx]
    K = scene.get_k(cam_id=cam_id)

    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))

    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    camera_pose = scene.get_camera_pose(cam_id=cam_id)

    for i, (frame_id1, frame_id2) in tqdm(enumerate(zip(scene.frame_ids[:-1], scene.frame_ids[1:])), total=len(scene.frame_ids)):
        # if i > 50:
        #     break
        # print(i)

        img1 = scene.get_cam_frame(frame_id=frame_id1, cam_id=cam_id)
        img2 = scene.get_cam_frame(frame_id=frame_id2, cam_id=cam_id)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # matches = bf.match(des1, des2)

        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        matches = [m for m in matches if len(m) == 2]
        new_matches = []
        # dists = []
        if heuristic:
            for i, (m,n) in enumerate(matches):
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                y_dist = np.abs(pt2[1] - pt1[1])
                # dists.append(y_dist)
                if y_dist < 15:
                # if np.abs(m.distance - n.distance) / m.distance <= 0.2:
                    good.append(m)
                    pts1.append(pt1)
                    pts2.append(pt2)
                    new_matches.append((m, n))
            matches = new_matches
        else:
            for m,n in matches:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        try:
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

            pose1 = scene.get_pose(frame_id=frame_id1)
            pose2 = scene.get_pose(frame_id=frame_id2)
            # find F from known poses
            pose1 = np.matmul(np.linalg.inv(camera_pose), pose1)
            pose2 = np.matmul(np.linalg.inv(camera_pose), pose2)
            R1 = pose1[:3,:3]
            t1 = pose1[:3,3]
            R2 = pose2[:3,:3]
            t2 = pose2[:3,3]
            def skew(v):
                return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            
            if estimate_E:
                E = np.matmul(np.matmul(np.transpose(K), F), K)
            else:
                E = np.matmul(skew(t2), np.matmul(R2, np.transpose(R1))) + np.matmul(skew(t1), R1)
                

            # E = np.linalg.norm(est) * E / np.linalg.norm(E)


            # import IPython ; IPython.embed() ; exit(1)

            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]
            # E = np.matmul(np.matmul(np.transpose(K), F), K)

            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        except:
            continue

        R_t_1[:3,:3] = np.matmul(R, R_t_0[:3,:3])
        R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],t.ravel())

        P2 = np.matmul(K, R_t_1)

        pts1 = np.transpose(pts1)
        pts2 = np.transpose(pts2)

        points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        # points_3d /= points_3d[3]

        # translate points to world frame
        pose = scene.get_pose(frame_id=frame_id1)
        # R = pose[:3,:3]
        # t = pose[:3,3]
        # points_3d[:3] = np.matmul(R, points_3d[:3]) + t.reshape(3,1)
        points_3d = pose @ camera_pose @ points_3d
        points_3d /= points_3d[3]

        X = np.concatenate((X, points_3d[0]))
        Y = np.concatenate((Y, points_3d[1]))
        Z = np.concatenate((Z, points_3d[2]))

        # matches = sorted(matches, key=lambda x: x.distance)
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, [m[0] for m in matches], None, flags=2)
        # cv2.imshow('img', img3)
        # cv2.waitKey(0)
    return X, Y, Z

scene_id = '000027'
heuristic = True
estimate_E = False

if not os.path.exists(scene_id):
    os.mkdir(scene_id)

scene = Scene(data_dir='/Users/alexvesel/Downloads/oncedataset', scene_id=scene_id)

pts_4d = []
for i in range(0, len(scene.cams)):
    X, Y, Z = sfm(i, heuristic, estimate_E)
    pts_4d.append(np.vstack((X, Y, Z, np.ones(X.shape))))

# remove points that are too far away
pts_4d = np.hstack(pts_4d)
pts_4d = pts_4d[:, np.logical_and(pts_4d[2] < 160, pts_4d[2] > -95)]
pts_4d = pts_4d[:, np.logical_and(pts_4d[1] < 2*1440, pts_4d[1] > 2*-950)]
pts_4d = pts_4d[:, np.logical_and(pts_4d[0] < 2*4250, pts_4d[0] > 2*-2500)]

# viz_3d(np.array(pts_4d))

pts_4d = pts_4d.T
df = pd.DataFrame(pts_4d)
name = f'{scene_id}_pred'
if heuristic:
    name += '_heuristic'
if estimate_E:
    name += '_estE'
df.to_csv(f'{scene_id}/{name}.csv', index=False, header=False)
