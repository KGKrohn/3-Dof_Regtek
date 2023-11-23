import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2

prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

camera_port = 0  # this is the camera port number (This can vary from, 0 - 10 from pc to pc)
cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)  # Set the camera capture device

# set width and height of the camera capture area
#cap.set(1,1000)
#cap.set(1, 700)

get, img = cap.read()
h,w,_ = img.shape

myColorFinder = ColorFinder(False) # if you want to find the color and calibrate the program we use this *(Debugging)
hsvVals = {'hmin': 10, 'smin': 50, 'vmin': 162, 'hmax': 120, 'smax': 255, 'vmax': 255}  # this is hsv values for orange color

center_point = [626,337,2210] # this center point is found by placing the ball at the center of the plate and calibrating it.

while True:
    ret, frame = cap.read()
    if not ret: break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame,(17, 17), 0)

    #imgColor, mask = myColorFinder.update(img,hsvVals)
    #imgContour, countours = cvzone.findContours(img,mask)

    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=30, minRadius=20, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        cv2.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2],(255, 0, 255), 3)
        prevCircle = chosen

    cv2.imshow("circles", frame)
    cv2.waitKey(1)



