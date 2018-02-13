import cv2
import numpy as np
import math

# DOUBLE LEFT CLICK DRAWS AND EDITS POINTS.
# RIGHT CLICK SELECTS EDITED POINT BY MAKING MAGENTA.
# 'S' BUTTON TAKES AN INPUT AND SAVES DATAS TO TEXT FILE.
# 'ESC' BUTTON TO EXIT PROGRAM

# Below that, I described main parameters and control parameters.
coordList = []
flag = False
rad = 10
checkingIntersection = False
editControl = False
numFirst = 0
numSecond = 0

# Below that, This draw function is main function of the homework.
def draw(event, x, y, flags, param):

    global flag, coordList, editControl, numFirst, numSecond, checkingIntersection, editMagenta

    # Below that, This if statement represents taking and drawing first two points.
    if event == cv2.EVENT_LBUTTONDBLCLK and flag == False and checkingIntersection == False:
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(img, (x, y), rad, (0, 255, 255), 0)
        coordList.append(x)
        coordList.append(y)
        checkingIntersection = True

    # Below that, This elif statement represents that taking and drawing second two points depending on first two points (above if part).
    # The reason which I code this part is that checking 10 pixel distance between first and second points.
    elif event == cv2.EVENT_LBUTTONDBLCLK and flag == False and checkingIntersection == True and (inside_point(coordList[0],coordList[1],rad,x,y) == 0) and (x > coordList[0] and y > coordList[1]):
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv2.circle(img, (x, y), rad, (0, 255, 255), 0)
        coordList.append(x)
        coordList.append(y)

        if (len(coordList) == 4):
            flag = True
            cv2.rectangle(img, (coordList[0], coordList[1]), (coordList[2], coordList[3]), (0, 0, 255), 2)

    # Below that, This elif statement represents that making selected point magenta.
    elif event == cv2.EVENT_RBUTTONDOWN and flag == True:
        numFirst = inside_point(coordList[0],coordList[1],rad,x,y)
        numSecond = inside_point(coordList[2],coordList[3],rad,x,y)

        if numFirst == 1 and numSecond == 0:
            editControl = True
            edit_img(coordList[0],coordList[1], coordList[2], coordList[3], 1)

        elif numFirst == 0 and numSecond == 1:
            editControl = True
            edit_img(coordList[0],coordList[1], coordList[2], coordList[3], 2)

    # Below that, This elif statement is the complicated part of the homework.
    # This code represents that relationships between two points for editing part.
    # There are 2 different scenarios to select points.
    # Every possibility depends on distance between two points, sequence order of the image (e.g. top-left(the first click) and bottom-right(the second click))
    elif event == cv2.EVENT_LBUTTONDBLCLK and flag == True and editControl == True:

        if numFirst == 1 and numSecond == 0 and coordList[0] < coordList[2] and coordList[1] < coordList[3] and not (x > coordList[2] or y > coordList[3]) and (inside_point(coordList[2],coordList[3],rad,x,y)==0):
            coordList[0] = x
            coordList[1] = y
            edit_img(coordList[0],coordList[1],coordList[2],coordList[3],0)

        elif numFirst == 0 and numSecond == 1 and coordList[0] < coordList[2] and coordList[1] < coordList[3] and not (x < coordList[0] or y < coordList[1]) and (inside_point(coordList[0],coordList[1],rad,x,y)==0):
            coordList[2] = x
            coordList[3] = y
            edit_img(coordList[0],coordList[1],coordList[2],coordList[3],0)

# This function represents that editing image to import new and old points by creating first image on same window.
def edit_img(x1,y1,x2,y2,magentaCheck):
    rad = 10
    img = cv2.imread('img2.jpg',1)
    cv2.namedWindow('image')
    if magentaCheck == 1:
        cv2.circle(img, (x1, y1), 4, (255, 0, 255), -1)
    else:
        cv2.circle(img, (x1, y1), 4, (0, 255, 0), -1)
    cv2.circle(img, (x1, y1), rad, (0, 255, 255), 0)

    if magentaCheck == 2:
        cv2.circle(img, (x2, y2), 4, (255, 0, 255), -1)
    else:
        cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)

    cv2.circle(img, (x2, y2), rad, (0, 255, 255), 0)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.setMouseCallback('image',draw)
    
    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('S') or k == ord('s') :
            name = input("Please enter name of object: ")
            save_file(x1,y1,x2,y2,name)

    cv2.destroyAllWindows()

# This function represents that checking whether the green dot is inside the yellow circle or not.
def inside_point(x,y,rad,x_exp,y_exp):
    if ((math.pow(x-x_exp,2) + math.pow(y-y_exp,2)) <= math.pow(rad,2)):
        return 1
    else:
        return 0

# This function represents that saving datas to text file.
def save_file(x1,y1,x2,y2,objName):
    textFile = open("roi_file.txt", "w")
    textFile.write("\"" + objName + ", (" + str(x1) + ", " + str(y1) + "), (" + str(x2) + ", " + str(y2) + ")\"")
    textFile.close()

img = cv2.imread('img2.jpg', 1)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while (1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('S') or k == ord('s'):
        name = input("Please enter name of object: ")
        save_file(coordList[0],coordList[1],coordList[2],coordList[3],name)

cv2.destroyAllWindows()