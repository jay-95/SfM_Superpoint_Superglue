import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from superpoint import SuperPointFrontend
import time

"""
def write_pointcloud(filename,xyz_points,rgb_points=None):

     # creates a .pkl file of the point clouds generated
    

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()
"""

# =============================================================================
# Superpoint calculation
# =============================================================================
# Set the parameters for superpoint calculation
parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
# parser.add_argument('input', type=str, default='',
#   help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                    help='Path to pretrained weights file (default: superpoint_v1.pth).')
parser.add_argument('--img_glob', type=str, default='*.png',
                    help='Glob match if directory of images is specified (default: \'*.png\').')
parser.add_argument('--skip', type=int, default=1,
                    help='Images to skip if input is movie or directory (default: 1).')
parser.add_argument('--show_extra', action='store_true',
                    help='Show extra debug outputs (default: False).')
parser.add_argument('--H', type=int, default=120,
                    help='Input image height (default: 120).')
parser.add_argument('--W', type=int, default=160,
                    help='Input image width (default:160).')
parser.add_argument('--display_scale', type=int, default=2,
                    help='Factor to scale output visualization (default: 2).')
parser.add_argument('--min_length', type=int, default=2,
                    help='Minimum length of point tracks (default: 2).')
parser.add_argument('--max_length', type=int, default=5,
                    help='Maximum length of point tracks (default: 5).')
parser.add_argument('--nms_dist', type=int, default=4,
                    help='Non Maximum Suppression (NMS) distance (default: 4).')
parser.add_argument('--conf_thresh', type=float, default=0.015,
                    help='Detector confidence threshold (default: 0.015).')
parser.add_argument('--nn_thresh', type=float, default=0.7,
                    help='Descriptor matching threshold (default: 0.7).')
parser.add_argument('--camid', type=int, default=0,
                    help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
parser.add_argument('--waitkey', type=int, default=1,
                    help='OpenCV waitkey time in ms (default: 1).')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda GPU to speed up network processing speed (default: False)')
parser.add_argument('--no_display', action='store_true',
                    help='Do not display images to screen. Useful if running remotely (default: False).')
parser.add_argument('--write', action='store_true',
                    help='Save output frames to a directory (default: False)')
parser.add_argument('--write_dir', type=str, default='tracker_outputs/',
                    help='Directory where to write output frames (default: tracker_outputs/).')
opt = parser.parse_args()
print(opt)
# print('==> Loading pre-trained network.')

# Run the superpoint
t0= time.clock()
fe = SuperPointFrontend(weights_path=opt.weights_path,
                        nms_dist=opt.nms_dist,
                        conf_thresh=opt.conf_thresh,
                        nn_thresh=opt.nn_thresh,
                        cuda=opt.cuda)
t1 = time.clock() - t0
print(t1)


# =============================================================================
# Set the variables and Load the images
# =============================================================================
# Set the directory of the images
path1 = 'sfm01.JPG'
path2 = 'sfm02.JPG'

# Camera parameter
K = np.array([[1698.873755, 0.000000, 971.7497705],
              [0.000000, 1698.8796645, 647.7488275],
              [0.000000, 0.000000, 1.000000]])

# Load the images
img01 = cv2.imread(path1)
img02 = cv2.imread(path2)

img01_2_gray = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)
img02_2_gray = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)

input_image01 = img01_2_gray.astype('float') / 255.0
input_image02 = img02_2_gray.astype('float') / 255.0


input_image01 = input_image01.astype('float32')
input_image02 = input_image02.astype('float32')

kp1, desc1, heatmap1 = fe.run(input_image01)
kp2, desc2, heatmap2 = fe.run(input_image02)

# Feature matching using knn algorithm
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desc1.T,desc2.T,k=2)

pts1 = []
pts2 = []
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < n.distance:
        pts2.append(kp2.T[m.trainIdx])
        pts1.append(kp1.T[m.queryIdx])
        good.append([m])

pts1 = np.array(pts1)[:, :2]
pts2 = np.array(pts2)[:, :2]

# Check whether the features are extracted well
for j in range(len(pts1)):
    pt1 = (int(round(pts1[j,0])), int(round(pts1[j,1])))
    pt2 = (int(round(pts2[j,0])), int(round(pts2[j,1])))
    cv2.circle(img01, pt1, 3, (0, 255, 0), -1, lineType=16)
    cv2.circle(img02, pt2, 3, (0, 255, 0), -1, lineType=16)

cv2.imwrite('superpoint_out01.JPG', img01)
cv2.imwrite('superpoint_out02.JPG', img02)

# Check SIFT feature matching between two images
img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)
img03 = np.hstack((img01, img02))
for i in range(len(pts1)):
    color = tuple([sp.random.randint(0, 255) for _ in range(3)])
    cv2.line(img03, (int(pts1[i,0]),int(pts1[i,1])), (int(pts2[i,0]+input_image01.shape[1]),int(pts2[i,1])), color)

plt.imshow(img03),plt.show()


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
df1.to_csv('super_3d_pts.csv', header=False, index=False)

df2 = pd.DataFrame(pts1)
df2.to_csv('super_pts1.csv', header=False, index=False)
