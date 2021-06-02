X = readtable('sift_3d_pts.csv');
X = X{:,:}';
pts1 = readtable('sift_pts1.csv');
pts1 = pts1{:,:}';

img = imread('sfm01.JPG');

X_with_color = []; % [6 x # of feature matrix] - XYZRGB
pixel_rgb = [];
for i = 1:size(pts1, 2)
    pixel_rgb = [pixel_rgb [img(int32(pts1(2,i)), int32(pts1(1,i)), 1); img(int32(pts1(2,i)), int32(pts1(1,i)), 2); img(int32(pts1(2,i)), int32(pts1(1,i)), 3)]];
end

pixel_rgb = double(pixel_rgb)./255;
X_with_color = [X; pixel_rgb];

SavePLY('sift_3d_globe_final.ply', X_with_color);