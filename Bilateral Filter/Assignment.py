import numpy as np
import cv2
import math
import time

# Below that, there is bilateral filter function that takes four parameter which are original image, image size, intensity kernel, spatial kernel
def bilateral_filter(img, image_size, s_intensity, s_spatial):
    filtered_img = np.zeros(img.shape)

    # x represents row of image.
    x = 0
    while x < len(img):

        # y represents column of image.
        y = 0
        while y < len(img[0]):

            hl = int(round(image_size / 2))
            i_filtered = 0
            wp_vectoral = 0

            # i represents column row neighbour of centered image pixel.
            i = 0
            while i < image_size:

                # y represents column column neighbour of centered image pixel.
                j = 0
                while j < image_size:
                    neighbour_x = x - (hl - i)
                    neighbour_y = y - (hl - j)
                    if neighbour_x >= len(img):
                        neighbour_x -= len(img)
                    if neighbour_y >= len(img[0]):
                        neighbour_y -= len(img[0])

                    # Below that, these are gaussian intensity and gaussian splatial values used in filtered image and normalization term.
                    gaussian_intensity = gaussian_formula(img[neighbour_x][neighbour_y] - img[x][y], s_intensity)
                    gaussian_spatial = gaussian_formula(math.sqrt(math.pow(neighbour_x-x,2) + math.pow(neighbour_y-y,2)), s_spatial)

                    i_filtered += img[neighbour_x][neighbour_y] * gaussian_intensity * gaussian_spatial

                    # Below that, normalization term includes summation of all possible gaussian intensity and gaussian spatial differences between neigbour pixel and centered pixel.
                    wp_vectoral += gaussian_intensity * gaussian_spatial

                    j += 1

                i += 1

            i_filtered = i_filtered / wp_vectoral

            # Below that, we round each value of pixel to integer value
            cont = 0
            i_filtered = round(i_filtered)
            filtered_img[x][y] = i_filtered

            y += 1

        x += 1

    return filtered_img


# Below that, I described gaussian formula inspired by homework pdf.
def gaussian_formula(x,sigma):
    return (1.0 / (2 * 3.14 * (sigma ** 2))) * 2.71 ** (- (x ** 2) / (2 * sigma ** 2))


image = cv2.imread('in_img.jpg',0)
my_filter = bilateral_filter(image, 7, 10, 10)
cv2.imwrite('filtered_image_own.png', my_filter)

openCV_filter = cv2.bilateralFilter(image, 7, 10, 10)
cv2.imwrite('filtered_image_OpenCV.png', openCV_filter)

# I selected 10 for intensity kernel and 10 for spatial kernel. The reason that i select these values is that;
# When selecting lower value for intensity kernel, image looked like original version. On the other hand, as value of intensity kernel was high, image looked like blur.
# When selecting lower value for spatial kernel, image look like clear which means there is no noise (i'm not sure that because there is no detail in image when changing spatial kernel), On the other hand, as value of spatial kernel was high, image looked like noisy.
# I decided that 10 is the optimum value for both of kernel values.


