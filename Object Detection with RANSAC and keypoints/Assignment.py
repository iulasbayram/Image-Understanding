import cv2
import  numpy as np
import random
import time
import math

img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

orb = cv2.ORB_create(nfeatures=2000)

# Descriptors and Keypoints of both images
kp1 = orb.detect(img1,None)
kp1, des1 = orb.compute(img1, kp1)
kp2 = orb.detect(img2,None)
kp2, des2 = orb.compute(img2, kp2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

kp1_points = []
kp2_points = []
for point in matches:
    img1_idx = point.queryIdx
    img2_idx = point.trainIdx
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    kp1_points.append((x1, y1))
    kp2_points.append((x2, y2))

src_points, dst_points = np.array(kp1_points), np.array(kp2_points)
print("There are ", len(kp1_points) , " keypoints obtained.")

def DlT_to_find_Homography(corres_from_src, corres_from_dst):

    avg_corres_src_x, avg_corres_src_y = find_avg_of_points(corres_from_src)
    avg_corres_dst_x, avg_corres_dst_y = find_avg_of_points(corres_from_dst)

    moved_corres_from_src = find_moved_points(corres_from_src, avg_corres_src_x, avg_corres_src_y)
    moved_corres_from_dst = find_moved_points(corres_from_dst, avg_corres_dst_x, avg_corres_dst_y)

    total_dist_moved_corres_src_points = find_euclidean_distance(moved_corres_from_src, 0, 0)
    total_dist_moved_corres_dst_points = find_euclidean_distance(moved_corres_from_dst, 0, 0)

    scale_src = np.sqrt(2) / total_dist_moved_corres_src_points
    scale_dst = np.sqrt(2) / total_dist_moved_corres_dst_points

    corres_src_points_tilda = np.dot(scale_src, moved_corres_from_src)
    corres_dst_points_tilda = np.dot(scale_dst, moved_corres_from_dst)

    corres_from_src = np.float_(corres_from_src)  # To fit array form from list
    corres_from_dst = np.float_(corres_from_dst)

    hmg_corres_src_points_tilda = np.insert(corres_src_points_tilda, 2, 1, axis=1)
    hmg_corres_dst_points_tilda = np.insert(corres_dst_points_tilda, 2, 1, axis=1)

    T_src = np.array([[scale_src, 0, scale_src * -1 * avg_corres_src_x], [0, scale_src, scale_src * -1 * avg_corres_src_y],
                  [0, 0, 1]])
    T_dst = np.array([[scale_dst, 0, scale_dst * -1 * avg_corres_dst_x], [0, scale_dst, scale_dst * -1 * avg_corres_dst_y],
         [0, 0, 1]])

    height, width = hmg_corres_src_points_tilda.shape

    A = np.zeros((2 * height, 9), dtype=np.float64)
    for i in range(0, len(hmg_corres_src_points_tilda)):
        A[2 * i][3] = -hmg_corres_src_points_tilda[i][0]
        A[2 * i][4] = -hmg_corres_src_points_tilda[i][1]
        A[2 * i][5] = -hmg_corres_src_points_tilda[i][2]
        A[2 * i][6] = hmg_corres_dst_points_tilda[i][1] * hmg_corres_src_points_tilda[i][0]
        A[2 * i][7] = hmg_corres_dst_points_tilda[i][1] * hmg_corres_src_points_tilda[i][1]
        A[2 * i][8] = hmg_corres_dst_points_tilda[i][1] * hmg_corres_src_points_tilda[i][2]

        A[2 * i + 1][0] = hmg_corres_src_points_tilda[i][0]
        A[2 * i + 1][1] = hmg_corres_src_points_tilda[i][1]
        A[2 * i + 1][2] = hmg_corres_src_points_tilda[i][2]
        A[2 * i + 1][6] = -hmg_corres_dst_points_tilda[i][0] * hmg_corres_src_points_tilda[i][0]
        A[2 * i + 1][7] = -hmg_corres_dst_points_tilda[i][0] * hmg_corres_src_points_tilda[i][1]
        A[2 * i + 1][8] = -hmg_corres_dst_points_tilda[i][0] * hmg_corres_src_points_tilda[i][2]

    U, s, VT = cv2.SVDecomp(A)
    L = VT[-1]
    H_unnormalized = L.reshape(3, 3)
    H = np.dot(np.dot(np.linalg.inv(T_dst), H_unnormalized), T_src)
    return H

def find_avg_of_points(points):
	x = 0
	y = 0
	for point in points:
		x += point[0]
		y += point[1]
	return x/len(points) ,y/len(points)

def find_moved_points(points,avg_x, avg_y):
	moved_points = []
	for point in points:
		x = point[0] - avg_x
		y = point[1] - avg_y
		moved_points.append([x,y])
	return moved_points

def find_euclidean_distance(points,x,y):
	total_dist_of_points = 0
	for point in points:
		total_dist_of_points += ((point[0]-x)**2 + (point[1]-y)** 2)**0.5
	return total_dist_of_points / len(points)

# To find inlier points.
def find_inliers(points, points_prime,src_points,dst_points):
    inlier_src_list = []
    inlier_dst_list = []
    i = 0
    while i < len(points_prime):
        j = 0
        while j < len(points):
            if (int(((points_prime[i][0] - points[j][0])**2 + (points_prime[i][1] - points[j][1])**2)**0.5) <= 3):
                inlier_src_list.append([src_points[j][0], src_points[j][1]])
                inlier_dst_list.append([dst_points[j][0], dst_points[j][1]])
                break
            j += 1
        i += 1
    return inlier_src_list, inlier_dst_list

N = 10000
index = 0

H_best = 0

inlier_src_points = []
inlier_dst_points = []

while(index < N):
    random_list = []
    corres_from_src = []
    corres_from_dst = []

    # To select random 4 correspondances
    for i in range(0,4):
        rndm = random.randint(0,len(src_points)-1)
        random_list.append(rndm)
        corres_from_src.append([src_points[rndm][0],src_points[rndm][1]])
        corres_from_dst.append([dst_points[rndm][0],dst_points[rndm][1]])

    # to find homography matrix using DLT
    H = DlT_to_find_Homography(corres_from_src,corres_from_dst)

    # These five lines are to obtain destination prime points
    hmg_src_points = np.insert(src_points, 2, 1, axis=1)  # (1040,3)
    hmg_dst_points_prime = np.dot(H, np.transpose(hmg_src_points))  # (3,1040)
    hmg_dst_points_prime[0] = hmg_dst_points_prime[0] / hmg_dst_points_prime[2]
    hmg_dst_points_prime[1] = hmg_dst_points_prime[1] / hmg_dst_points_prime[2]
    dst_points_prime = np.transpose(np.delete(hmg_dst_points_prime, 2, 0))

    inlier_src_actual, inlier_dst_actual = find_inliers(dst_points, dst_points_prime, src_points, dst_points)

    print("Num of actual index: " , index, "-- Num of iteration: ",N , "-- Inlier count: ", len(inlier_src_actual))

    # to update N, Homography matrix and inlier points
    w_s = (len(inlier_src_actual)/len(src_points))**4
    denominator = np.log10(1-w_s)
    if(w_s != 0.0 and denominator != 0.0):
        new_N = np.log10(0.01)/denominator
        if(new_N < N and (new_N != float("Inf") and new_N != -float("Inf"))):
            N = int(new_N)
            H_best = H
            inlier_src_points = inlier_src_actual
            inlier_dst_points = inlier_dst_actual

    index += 1

print()
print("Number of iteration: ",index)
print()
print("Inlier source points: ",inlier_src_points)
print()
print("Inlier destination points: ",inlier_dst_points)
print()
print("Best homography after ransac: ", H_best)
print()

# to converge our inliers at one last time
converge = True
while(converge):
    print("Before reducing: ", len(inlier_src_points))
    H_inlier = DlT_to_find_Homography(inlier_src_points,inlier_dst_points)
    hmg_inlier_src_points = np.insert(inlier_src_points, 2, 1, axis=1)  # (1040,3)

    hmg_inlier_dst_points_prime = np.dot(H_inlier, np.transpose(hmg_inlier_src_points))  # (3,1040)
    hmg_inlier_dst_points_prime[0] = hmg_inlier_dst_points_prime[0] / hmg_inlier_dst_points_prime[2]
    hmg_inlier_dst_points_prime[1] = hmg_inlier_dst_points_prime[1] / hmg_inlier_dst_points_prime[2]
    inlier_dst_points_prime = np.transpose(np.delete(hmg_inlier_dst_points_prime, 2, 0))

    converged_inlier_src_points, converged_inlier_dst_points = find_inliers(inlier_dst_points,inlier_dst_points_prime,inlier_src_points,inlier_dst_points)
    print("After reducing: ", len(converged_inlier_src_points))
    print()
    if(len(inlier_src_points) == len(converged_inlier_src_points)): #If converged, loop is over
        converge = False
    else: #If not converged, our variables are updated
        inlier_src_points = converged_inlier_src_points
        inlier_dst_points = converged_inlier_dst_points
        H_best = H_inlier


# Source image corners
corners = np.array([[160,465,160,465],
                    [167,167,575,575],
                    [1,1,1,1]])

# New corners after multiplying homography matrix
corners_prime = np.dot(H_best,corners)

corners_prime[0] = corners_prime[0] / corners_prime[2]
corners_prime[1] = corners_prime[1] / corners_prime[2]

cv2.rectangle(img2,(int(corners_prime[0][0]),int(corners_prime[1][0])),(int(corners_prime[0][3]),int(corners_prime[1][3])),(0,0,255),2)
cv2.imwrite("object_detection.png",img2)

############################################ COMMENT - I ########################################################
# SIFT is the one of the oldest feature matching methodology. It uses difference of gaussian and it eliminates low contrast points.
# ORB is the one of the fastest feature matching methodology. It computes intensity weighed centroid of the patch with
# located corner at center. The direction of two vector from this corner point to centroid gives the orientation.
# For comparing images with varying intensity, ORB gives more keypoints than SIFT but SIFT gives better match rate.
# For comparing images with its rotated image, Altough ORB gives more keypoints again, SIFT gives better match rate.
# For matching rate with rotation angle, ORB gives better result at 0-90-180-270, SIFT gives better result at 45-135-225

############################################ COMMENT - II ########################################################
# We have keypoints and descriptors of two images (Source and Destination). For each descriptor of source image,
# we find minimum distance among descriptor of destination image and each source descriptor matches one destination descriptor.
# In this way, we can obtain each keypoint corresponding to its descriptor.

# SIFT uses floating point descriptors. To find distance between two SIFT descriptors, we can find euclidean distance
# between two descriptors in SIFT.

# ORB uses binary descriptors. To find distance between two ORB descriptors, we can convert their values binary to decimal
# for each ORB values and find distance by calculating differences between them.

############################################ COMMENT - III ########################################################
# When we obtain optimal homography matrix, we must carry our source points to destination points by multiplying
# homography matrix with source points. After that, we obtain translated points and check which translated points
# corresponds to destination points. If each of translated points is closer than 3 pixels to destination points,
# that is the inlier for our new homography matrix.

############################################ COMMENT - IV ########################################################
# Normalization affects our A matrix by decreasing condition number of it. After that, homography matrix is obtained
# by V matrix which is right orthonormal matrix of A. Then, using this homography matrix, we set our points
# a particular border to get more accurate result. It provides that noisy points in our points are more stable
# thanks to homography matrix.