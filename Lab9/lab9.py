import cv2
import  numpy as np
import random

img1 = cv2.imread("img1.jpg",0)
img2 = cv2.imread("img2.jpg",0)
fname = "corrs.txt"

def read_matches(fname):
	fin = open(fname)
	f = fin.read()
	fin.close()
	src_points = []
	dst_points = []
	for line in f.strip().split("\n"):
		points = line.strip().split(",")
		p1 = points[0].strip().split(" ")
		p2 = points[1].strip().split(" ")
		src_points.append((float(p1[0]),float(p1[1])))
		dst_points.append((float(p2[0]),float(p2[1])))
	return src_points,dst_points

src_points = []
dst_points = []
src_points,dst_points = read_matches(fname)
src_points = np.float_(src_points)
dst_points = np.float_(dst_points)

def random_check(list,num):
    check_num = 0
    for num_temp in list:
        if num == num_temp:
            check_num += 1

    if check_num == 0:
        return False
    else:
        return True

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
		total_dist_of_points += np.sqrt(np.power(point[0]-x, 2) + np.power(point[1]-y, 2))
	return total_dist_of_points / len(points)

N = 10000
H_updated = 0
index = 0
inlier_src = []
while(index < N):
    print("Num of iteration: ",N)
    print("Num of actual index: " , index)
    random_list = []
    corres_from_src = []
    corres_from_dst = []

    for i in range(0,4):
        rndm = random.randint(0,len(src_points)-1)
        check = True
        while (check):
            check = random_check(random_list, rndm)
            if check == True:
                rndm = random.randint(1,len(src_points))
        random_list.append(rndm)
        corres_from_src.append([src_points[rndm][0],src_points[rndm][1]])
        corres_from_dst.append([dst_points[rndm][0],dst_points[rndm][1]])

    def DlT_to_find_Homography(corres_from_src,corres_from_dst):


        avg_corres_src_x , avg_corres_src_y = find_avg_of_points(corres_from_src)
        avg_corres_dst_x , avg_corres_dst_y = find_avg_of_points(corres_from_dst)
        #print(avg_corres_src_x,avg_corres_src_y,avg_corres_dst_x,avg_corres_dst_y)
        moved_corres_from_src = find_moved_points(corres_from_src,avg_corres_src_x,avg_corres_src_y)
        moved_corres_from_dst = find_moved_points(corres_from_dst,avg_corres_dst_x,avg_corres_dst_y)

        total_dist_moved_corres_src_points = find_euclidean_distance(moved_corres_from_src,0,0)
        total_dist_moved_corres_dst_points = find_euclidean_distance(moved_corres_from_dst,0,0)

        scale_src = np.sqrt(2) / total_dist_moved_corres_src_points
        scale_dst = np.sqrt(2) / total_dist_moved_corres_dst_points

        corres_src_points_tilda = np.dot(scale_src,moved_corres_from_src)
        corres_dst_points_tilda = np.dot(scale_dst,moved_corres_from_dst)

        corres_from_src = np.float_(corres_from_src) #To fit array form from list
        corres_from_dst = np.float_(corres_from_dst)


        hmg_corres_src_points_tilda = np.insert(corres_src_points_tilda,2,1,axis=1)
        hmg_corres_dst_points_tilda = np.insert(corres_dst_points_tilda,2,1,axis=1)

        T = np.array( [ [scale_src, 0, scale_src*-1* avg_corres_src_x], [0, scale_src, scale_src*-1*avg_corres_src_y], [0, 0, 1] ] )
        Tprime = np.array( [ [scale_dst, 0, scale_dst*-1*avg_corres_dst_x], [0, scale_dst, scale_dst*-1*avg_corres_dst_y], [0, 0, 1] ] )


        height, width = hmg_corres_src_points_tilda.shape
        A = np.zeros((2*height,9),dtype=np.float64)

        for i in range(0,len(hmg_corres_src_points_tilda)):
            A[2*i][3] = -hmg_corres_src_points_tilda[i][0]
            A[2*i][4] = -hmg_corres_src_points_tilda[i][1]
            A[2*i][5] = -hmg_corres_src_points_tilda[i][2]
            A[2*i][6] = hmg_corres_dst_points_tilda[i][1]*hmg_corres_src_points_tilda[i][0]
            A[2*i][7] = hmg_corres_dst_points_tilda[i][1]*hmg_corres_src_points_tilda[i][1]
            A[2*i][8] = hmg_corres_dst_points_tilda[i][1]*hmg_corres_src_points_tilda[i][2]

            A[2*i+1][0] = hmg_corres_src_points_tilda[i][0]
            A[2*i+1][1] = hmg_corres_src_points_tilda[i][1]
            A[2*i+1][2] = hmg_corres_src_points_tilda[i][2]
            A[2*i+1][6] = -hmg_corres_dst_points_tilda[i][0]*hmg_corres_src_points_tilda[i][0]
            A[2*i+1][7] = -hmg_corres_dst_points_tilda[i][0]*hmg_corres_src_points_tilda[i][1]
            A[2*i+1][8] = -hmg_corres_dst_points_tilda[i][0]*hmg_corres_src_points_tilda[i][2]

        #print(A)
        U, s, VT = cv2.SVDecomp(A)
        L = VT[-1]
        H_unnormalized = L.reshape(3,3)
        H = np.dot(np.dot(np.linalg.inv(Tprime),H_unnormalized),T)
        return H

    H = DlT_to_find_Homography(corres_from_src,corres_from_dst)

    hmg_src_points = np.insert(src_points,2,1,axis=1) #(1040,3)

    hmg_dst_points_prime = np.dot(H,np.transpose(hmg_src_points))#(3,1040)
    hmg_dst_points_prime[0] = hmg_dst_points_prime[0] / hmg_dst_points_prime[2]
    hmg_dst_points_prime[1] = hmg_dst_points_prime[1] / hmg_dst_points_prime[2]
    dst_points_prime = np.transpose(np.delete(hmg_dst_points_prime,2,0))

    def find_inliers(points, points_prime):
        inlier_list = []
        for i in range(0, len(points_prime)):
            for j in range(0, len(points)):
                distance = np.sqrt(np.power(points_prime[i][0] - points[j][0], 2) + np.power(points_prime[i][1] - points[j][1], 2))
                if (distance <= 3):
                    inlier_list.append([src_points[i][0], src_points[i][1]])
                    break
        return inlier_list

    inlier_list= find_inliers(dst_points,dst_points_prime)

    print("Inlier count: ", len(inlier_list))

    inlier_ratio = (len(inlier_list) / len(src_points))

    w_s = np.power(len(inlier_list)/len(src_points),4)
    denominator = float(np.log10(1-w_s))
    if(w_s != 0.0 and denominator != 0.0):
        new_N = np.log10(0.01)/denominator
        print("new N: ",new_N)
        if(new_N < N and (new_N != float("Inf") and new_N != -float("Inf"))):
            N = int(new_N)
            H_updated = H
            inlier_src = inlier_list

    index += 1
    
print("Loop out inlier list: ", inlier_src)
print("Loop out H: ", H_updated)
print("Loop out number of iteration: ",N)
print("Loop out actual index: ",index)
