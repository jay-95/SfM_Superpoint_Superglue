from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import scipy as sp
import pandas as pd
import time

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, process_resize, frame2tensor)


t0= time.clock()
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs')
parser.add_argument(
    '--input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optionally,'
         'the visualization images are written')

parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
         'resize to the exact dimensions, if one number, resize the max '
         'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
         ' (requires ground truth pose and intrinsics)')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization with OpenCV instead of Matplotlib')
parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')
parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--force_cpu', action='store_true',
    help='Force pytorch to run in CPU mode.')

opt = parser.parse_args()
print(opt)

# Load the SuperPoint and SuperGlue models.
device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superpoint': {
        'nms_radius': opt.nms_radius,
        'keypoint_threshold': opt.keypoint_threshold,
        'max_keypoints': opt.max_keypoints
    },
    'superglue': {
        'weights': opt.superglue,
        'sinkhorn_iterations': opt.sinkhorn_iterations,
        'match_threshold': opt.match_threshold,
    }
}
matching = Matching(config).eval().to(device)


path1 = 'sfm01.JPG'
path2 = 'sfm02.JPG'


# Camera parameter
K = np.array([[1698.873755, 0.000000, 971.7497705],
              [0.000000, 1698.8796645, 647.7488275],
              [0.000000, 0.000000, 1.000000]])


img01 = cv2.imread(path1)
img02 = cv2.imread(path2)

img01_2_gray = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)
img02_2_gray = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)

# w1, h1 = img01_2_gray.shape[1], img01_2_gray.shape[0]
# w_new1, h_new1 = process_resize(w1, h1, opt.resize)
# scales1 = (float(w1) / float(w_new1), float(h1) / float(h_new1))
# input_image01 = cv2.resize(img01_2_gray, (w_new1, h_new1)).astype('float32')

# w2, h2 = img02_2_gray.shape[1], img02_2_gray.shape[0]
# w_new2, h_new2 = process_resize(w2, h2, opt.resize)
# scales2 = (float(w2) / float(w_new2), float(h2) / float(h_new2))
# input_image02 = cv2.resize(img02_2_gray, (w_new2, h_new2)).astype('float32')

inp01 = frame2tensor(img01_2_gray, device)
inp02 = frame2tensor(img02_2_gray, device)

pred = matching({'image0': inp01, 'image1': inp02})
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']

# Write the matches to disk.
# out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
#                'matches': matches, 'match_confidence': conf}

valid = matches > -1
pts1 = kpts0[valid]
pts2 = kpts1[matches[valid]]
mconf = conf[valid]
t1 = time.clock() - t0
print(t1)

img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)
img03 = np.hstack((img01, img02))
for i in range(len(pts1)):
    color = tuple([sp.random.randint(0, 255) for _ in range(3)])
    cv2.line(img03, (int(pts1[i,0]),int(pts1[i,1])), (int(pts2[i,0]+img01_2_gray.shape[1]),int(pts2[i,1])), color)

for j in range(len(pts1)):
    pt1 = (int(round(pts1[j,0])), int(round(pts1[j,1])))
    pt2 = (int(round(pts2[j,0])), int(round(pts2[j,1])))
    cv2.circle(img01, pt1, 3, (0, 255, 0), -1, lineType=16)
    cv2.circle(img02, pt2, 3, (0, 255, 0), -1, lineType=16)

cv2.imwrite('superglue_out01.JPG', img03)

"""
cv2.imshow('original', img03)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# =============================================================================
# Find essential matrix using RANSAC
# =============================================================================
# Make feature points into homogeneous form
homog_pts1 = np.vstack((pts1.T, np.ones(len(pts1))))
homog_pts2 = np.vstack((pts2.T, np.ones(len(pts2))))

# Transform pixel points onto normalized coordination
norm_pts1 = np.linalg.inv(K).dot(homog_pts1)
norm_pts2 = np.linalg.inv(K).dot(homog_pts2)

# Calculate essential matrix using RANSAC and 5-point algorithm
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)



# =============================================================================
# Derive the best projection matrix
# =============================================================================
# Calculate rotation and translation matrix using essential matrix decomposition
_, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, K)


P0 = np.eye(3, 4)
P1 = np.hstack((R_est, t_est))

proj_l = np.dot(K, P0)
proj_r = np.dot(K, P1)

points_4d_hom = cv2.triangulatePoints(P0, P1, norm_pts1[:2], norm_pts2[:2])

X = points_4d_hom/points_4d_hom[3]

x1 = np.dot(proj_l[:3], X)
x2 = np.dot(proj_r[:3], X)

x1 /= x1[2]
x2 /= x2[2]

"""
img01_rgb = Image.open(path1).load()
img02_rgb = Image.open(path2).load()
"""
"""
img02_rgb  = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)

colors = []
for x_idx, y_idx in pts2:
    # colors.append(img01[np.int8(x_idx), np.int8(y_idx), :])
    colors.append(img02_rgb[np.int8(y_idx), np.int8(x_idx)])
    
colors = np.array(colors)

write_pointcloud("super_out.ply", X[:3].T , np.int8(colors))
"""

df1 = pd.DataFrame(X[:3].T)
df1.to_csv('superglue_3d_pts.csv', header=False, index=False)

df2 = pd.DataFrame(pts1)
df2.to_csv('superglue_pts1.csv', header=False, index=False)

