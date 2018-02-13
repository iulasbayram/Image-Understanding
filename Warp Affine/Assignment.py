import cv2
import numpy as np

#Below that, I found the size of warp image using Affine Deformation Matrix.
def finding_WarpedImage_Size(A,height,width):

    point_1 = np.dot(A,np.array([[0],[0]]))
    point_2 = np.dot(A,np.array([[0],[width]]))
    point_3 = np.dot(A,np.array([[height],[0]]))
    point_4 = np.dot(A,np.array([[height],[width]]))

    warpedHeight_Candidates = [point_1[0][0],point_2[0][0],point_3[0][0],point_4[0][0]]
    warpedWidth_Candidates = [point_1[1][0],point_2[1][0],point_3[1][0],point_4[1][0]]

    maxWarped_Width = max(warpedWidth_Candidates)
    minWarped_Width = min(warpedWidth_Candidates)

    maxWarped_Height = max(warpedHeight_Candidates)
    minWarped_Height = min(warpedHeight_Candidates)

    warpedHeight = int(round(maxWarped_Height - minWarped_Height))
    warpedWidth = int(round(maxWarped_Width - minWarped_Width))

    return warpedHeight , warpedWidth

#Below that, I found coordinates of warp image by multiplying homography matrix with reference image.
def create_WarpedCoordinates(H, width, height):
    list = []
    for x in range(0, height):
        for y in range(0, width):
            warpedPoints = np.dot(H, np.array([[x], [y], [1]]))
            i = int(warpedPoints[0][0])
            j = int(warpedPoints[1][0])
            list.append([i, j])

    return list

#Below that, After I found coordinates of warp image, we should find corresponding intensity values of coordinates of warp image by multiplying inverse of homography matrix with warp image.
def create_Inverse_WarpedCoordinates(H_inverse, warped_CoordList, ):
    list = []
    for x in range(0, len(warped_CoordList)):
        newPoints = np.dot(H_inverse, np.array([[warped_CoordList[x][0]], [warped_CoordList[x][1]], [1]]))
        list.append([newPoints[0][0], newPoints[1][0]])

    return list

#Below that, I applied our coordinates bilinear interpolation using its formula.
def applying_bilinear_interpolation(image, warpedImage, warpedList, inverse_Warpedlist):
    for x in range(0, len(inverse_Warpedlist)):
        i = inverse_Warpedlist[x][0]
        j = inverse_Warpedlist[x][1]
        alpha = j - int(j)
        beta = i - int(i)

        if (0 <= int(i + 1) < len(image) and 0 <= int(j + 1) < len(image[0])):
            R00 = image[int(i)][int(j)] * (1 - beta) * (1 - alpha)
            R01 = image[int(i)][int(j + 1)] * (1 - beta) * (alpha)
            R10 = image[int(i + 1)][int(j)] * (beta) * (1 - alpha)
            R11 = image[int(i + 1)][int(j + 1)] * (beta) * (alpha)
            warpedImage[warpedList[x][0]][warpedList[x][1]] = round(R00 + R01 + R10 + R11)

#Below that, I applied our coordinates nearest-neighbour interpolation using round formula.
def applying_nearest_neighbour_interpolation(image,warpedImage,warpedList,inverse_Warplist):
    for x in range(0, len(inverse_Warplist)):
        i = round(inverse_Warplist[x][0])
        j = round(inverse_Warplist[x][1])
        warpedImage[warpedList[x][0]][warpedList[x][1]] = image[int(i)][int(j)]

#Reference image
image = cv2.imread("img1.png",0)

#Affine deformation matrix
A = np.dot(0.5,np.dot(np.array([[np.cos(np.pi/6),np.sin(np.pi/6)],[-np.sin(np.pi/6),np.cos(np.pi/6)]]) , np.array([[1/(np.cos(np.pi*(5/18))), 0], [0, 1]])))

#Height and width of reference image
height , width = image.shape

#We can find height and width of warp image using the function.
warpedHeight , warpedWidth = finding_WarpedImage_Size(A,height,width)

#Homography matrix
H = np.dot(np.dot(np.array([[1,0,warpedHeight/2],[0,1,warpedWidth/2],[0,0,1]]) , np.array([[A[0][0],A[0][1],0],[A[1][0],A[1][1],0],[0,0,1]])), np.array([[1,0,-height/2],[0,1,-width/2],[0,0,1]]))

#Inverse of homography matrix
H_inverse = np.linalg.inv(H)

#I created two frames of warp image. One of them is bilinear interpolated warp image, other is nearest neighbour interpolated warp image.
bilinear_interpolated_warpImage = np.zeros(shape=(warpedHeight,warpedWidth))
nearest_neighbour_interpolated_warpImage = np.zeros(shape=(warpedHeight,warpedWidth))

#Some operations to obtain final indexes before applying interpolation
warped_CoordList = create_WarpedCoordinates(H,width,height)
inverse_Warped_CoordList = create_Inverse_WarpedCoordinates(H_inverse,warped_CoordList)

#Applying interpolations of images
applying_bilinear_interpolation(image,bilinear_interpolated_warpImage,warped_CoordList,inverse_Warped_CoordList)
cv2.imwrite("Bilinear_Interpolated_Warped_Image.png",bilinear_interpolated_warpImage)

applying_nearest_neighbour_interpolation(image,nearest_neighbour_interpolated_warpImage,warped_CoordList,inverse_Warped_CoordList)
cv2.imwrite("Nearest_Neighbour_Interpolated_Warped_Image.png",nearest_neighbour_interpolated_warpImage)

########################################## COMMENT-1 ##########################################

# First of all, I multiplied affine with left side of homography matrix. When multiplying this matrix with pixels of reference image,
# i obtained some indexes that are out of bounds (too high index values) properly. When printing indexes, i observed that the output image (warp image)
# is shifted to bottom-right of it.
# After that,  I multiplied affine with right side of homography matrix. When multiplying this matrix with pixels of reference image,
# i obtained different image with some indexes that are not out of bounds becasue most of them are negative numbers and python allows
# negative indexes for array. Normally (by ignoring python) output image (warped image) is shifted to top-left of it.
# This informations give me an idea about why we multiply affine matrix with teo matrices from its left side and right side to obtain homography matrix.
# We want to fit indexes of warped image by indexes of reference image. Combining these two matrices (left and right side matrices) gives us an image that include weighed center as near as center of reference image.
# Therefore, to find appropriate indexes of warped image, we combined these matrixes.

########################################## COMMENT-2 ##########################################

# First of all, when making some transformation operations to image, we prefer two options. These are forward warping and inverse warping.
# Forward Warping provides that a pixel is copied to its corresponding location to warped image. However, this approach gets some problem because of its limitations.
# Problem with forward warping is the appearance of cracks and holes, especially when magnifying an image. Filling such holes with their nearby neighbors can lead to further aliasing and blurring
# Because of these reasons of using forward warping, there is quite a loss of high-resolution detail.
# To avoid these problems, we need inverse warping (transforming warped image to the reference image with inverse homography). Since hˆ(x′) is defined for all pixels in warp image, we no longer have holes.
# More importantly, resampling an image at non-integer locations is a well-studied problem and high-quality filters that control aliasing can be used.

########################################## COMMENT-3 ##########################################

# Nearest-neighbor interpolation assings the value of the nearest pixel (in python, using round() function) to the pixel in the output visualization.
# This is one of the fastest interpolation method but the final image may contain jagged edges and noises.
# Bilinear interpolation surveys the 4 closest pixels, creates a weighted average based on the nearness and
# brightness of the surveyed pixels and assigns that value to the pixel in the output image.
# In my opinion, bilinear interpolation gives better smoothness and sharpness and less than nearest-neighbour interpolation. Resolution
# in bilinear interpolation is also better than resolutin in nearest neighbour when zooming in the warped image.
