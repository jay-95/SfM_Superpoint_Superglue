import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
#from rw1 import SuperPointFrontend

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

"""
plt.figure(1)
plt.imshow(img01)
plt.figure(2)
plt.imshow(img02)
plt.show()
"""

"""
fig,ax = plt.subplots(ncols=2)
ax[0].imshow(img01)
ax[1].imshow(img02)
"""
# Convert images in grayscale
img01_2_gray = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)
img02_2_gray = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)


"""
cv2.imshow('1', img01_2_gray)
cv2.waitKey()
cv2.destroyAllWindows()
"""

# =============================================================================
# Feature Extration and Matching
# =============================================================================
# Feature extraction using SIFT algorithm
t0 = time.clock()
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img01_2_gray, None)
kp2, des2 = sift.detectAndCompute(img02_2_gray, None)

# Feature matching using knn algorithm
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []


for i,(m,n) in enumerate(matches):
    if m.distance < 0.85*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        good.append([m])
pts1 = np.array(pts1)
pts2 = np.array(pts2)

t1 = time.clock() - t0
print(t1)
# Check whether the features are extracted well
for j in range(len(pts1)):
    pt1 = (int(round(pts1[j,0])), int(round(pts1[j,1])))
    pt2 = (int(round(pts2[j,0])), int(round(pts2[j,1])))
    cv2.circle(img01, pt1, 3, (0, 255, 0), -1, lineType=16)
    cv2.circle(img02, pt2, 3, (0, 255, 0), -1, lineType=16)

cv2.imwrite('sift_out01.JPG', img01)
cv2.imwrite('sift_out02.JPG', img02)

# Check SIFT feature matching between two images
img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2RGB)
img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)
img_sift_match = cv2.drawMatchesKnn(img01,kp1,img02,kp2,good,None,flags=2)

plt.imshow(img_sift_match),plt.show()


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


df1 = pd.DataFrame(X[:3].T)
df1.to_csv('sift_3d_pts.csv', header=False, index=False)

df2 = pd.DataFrame(pts1)
df2.to_csv('sift_pts1.csv', header=False, index=False)
"""
colors = []
for y_idx, x_idx in pts1:
    colors.append(img01[np.int8(x_idx), np.int8(y_idx), :])
    
colors = np.array(colors)

write_pointcloud("out3.ply", X[:3].T , colors)
"""