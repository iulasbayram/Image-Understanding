import cv2
import  numpy as np

image = cv2.imread('in.jpg')

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

src_points,dst_points = read_matches("corrs.txt")

src_points = np.float_(src_points)
dst_points = np.float_(dst_points)
# 2) FIND EACH AVG POINTS OF SRC AND DST LIST

def find_avg_of_points(points):
	x = 0
	y = 0
	for point in points:
		x += point[0]
		y += point[1]
	return x/len(points) ,y/len(points)

avg_src_x ,avg_src_y = find_avg_of_points(src_points)
avg_dst_x , avg_dst_y = find_avg_of_points(dst_points)

print(avg_src_x,avg_src_y,avg_dst_x,avg_dst_y)
#435.95 285.15 264.78235 356.28720000000004

# 3) FIND EACH MOVED POINTS OF SRC AND DST LIST


def find_moved_points(points,avg_x, avg_y):
	moved_points = []
	for point in points:
		x = point[0] - avg_x
		y = point[1] - avg_y
		moved_points.append([x,y])
	return moved_points

moved_src_points = find_moved_points(src_points,avg_src_x,avg_src_y)
moved_dst_points = find_moved_points(dst_points,avg_dst_x,avg_dst_y)


# 4) FIND EACH TOTAL DISTANCE OF SRC AND DST LIST
print("moved_dst_src: ", moved_dst_points)

def find_euclidean_distance(points,x,y):
	total_dist_of_points = 0
	for point in points:
		total_dist_of_points += np.sqrt(np.power(point[0]-x, 2) + np.power(point[1]-y, 2))
	return total_dist_of_points / len(points)

total_dist_of_moved_src_points = find_euclidean_distance(moved_src_points,0,0)
total_dist_of_moved_dst_points = find_euclidean_distance(moved_dst_points,0,0)

print("total_dist_of_moved_src_points: ", total_dist_of_moved_src_points)
print("total_dist_of_moved_dst_points: ", total_dist_of_moved_dst_points)

# 5) FIND SCALE

scale_src = np.sqrt(2) / total_dist_of_moved_src_points
scale_dst = np.sqrt(2) / total_dist_of_moved_dst_points

print("scale_of_total_src: " , scale_src)
print("scale_of_total_dst: " , scale_dst)

# 6) FIND TILDA POINTS OF SRC AND DST

src_points_tilda = np.dot(scale_src ,moved_src_points)
dst_points_tilda = np.dot(scale_dst, moved_dst_points)

src_points = np.float_(src_points)
dst_points = np.float_(dst_points)

print("src_points: " ,src_points)
print("src_points_tilda: ", src_points_tilda)
print()
print("dst_points: " ,dst_points)
print("dst_points_tilda: ", dst_points_tilda)

hmg_src_points = np.insert(src_points,2,1,axis=1)
hmg_dst_points = np.insert(dst_points,2,1,axis=1)

hmg_src_points_tilda = np.insert(src_points_tilda,2,1,axis=1)
hmg_dst_points_tilda = np.insert(dst_points_tilda,2,1,axis=1)

T_src = np.dot(np.transpose(hmg_src_points_tilda),np.linalg.pinv(np.transpose(hmg_src_points)))
T_dst = np.dot(np.transpose(hmg_dst_points_tilda),np.linalg.pinv(np.transpose(hmg_dst_points)))

T = np.array( [ [scale_src, 0, scale_src*-1* avg_src_x], [0, scale_src, scale_src*-1*avg_src_y], [0, 0, 1] ] )

Tprime = np.array( [ [scale_dst, 0, scale_dst*-1*avg_dst_x], [0, scale_dst, scale_dst*-1*avg_dst_y], [0, 0, 1] ] )

print("T_src: " ,T_src)
print("T_dst: " ,T_dst)

height, width = hmg_src_points_tilda.shape

A = np.zeros((2*height,9),dtype=np.float64)
print(len(A))
height, width = A.shape


for i in range(0,len(hmg_src_points_tilda)):
	A[2*i][3] = -hmg_src_points_tilda[i][0]
	A[2*i][4] = -hmg_src_points_tilda[i][1]
	A[2*i][5] = -hmg_src_points_tilda[i][2]
	A[2*i][6] = hmg_dst_points_tilda[i][1]*hmg_src_points_tilda[i][0]
	A[2*i][7] = hmg_dst_points_tilda[i][1]*hmg_src_points_tilda[i][1]
	A[2*i][8] = hmg_dst_points_tilda[i][1]*hmg_src_points_tilda[i][2]

	A[2*i+1][0] = hmg_src_points_tilda[i][0]
	A[2*i+1][1] = hmg_src_points_tilda[i][1]
	A[2*i+1][2] = hmg_src_points_tilda[i][2]
	A[2*i+1][6] = -hmg_dst_points_tilda[i][0]*hmg_src_points_tilda[i][0]
	A[2*i+1][7] = -hmg_dst_points_tilda[i][0]*hmg_src_points_tilda[i][1]
	A[2*i+1][8] = -hmg_dst_points_tilda[i][0]*hmg_src_points_tilda[i][2]


print("A: " , A)

U, s, VT = cv2.SVDecomp(A)
L = VT[-1]
H_normalized = L.reshape(3,3)

print("H_normalized: ", H_normalized)



H = np.dot(np.dot(np.linalg.inv(Tprime),H_normalized),T)

warp = cv2.warpPerspective(image, H, (1000, 1000))
cv2.imwrite('warp_perspective.jpg', warp)


