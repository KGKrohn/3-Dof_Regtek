import multiprocessing as mp
from multiprocessing import Queue
import cvzone
from cvzone.ColorModule import ColorFinder
import cv2
import serial
import numpy as np
import time
import sympy as sp
import math
from scipy.signal import butter, filtfilt
from collections import deque


#---------- globale verdier ---------
center_point = [610, 367, 2210]
s, t = sp.symbols('s t')
# --------------------------
def ball_track(key1, queue):
    camera_port = 0
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    get, img = cap.read()
    h, w, _ = img.shape
    if key1:
        print('Ball tracking is initiated')

    myColorFinder = ColorFinder(False)  # if you want to find the color and calibrate the program we use this *(Debugging)
    hsvVals = {'hmin': 19, 'smin': 0, 'vmin': 170, 'hmax': 106, 'smax': 255, 'vmax': 255}

    #Trail funksjoner
    lineLen = 20
    pts = deque(maxlen= lineLen)

    def scaling(value, iMin, iMax, qMin, qMax):
        return qMin + (((value - (iMin)) / (iMax - iMin)) * (qMax - qMin))

    while True:
        get, img = cap.read()

        mask_plat = np.zeros(img.shape[:2], dtype = 'uint8')
        cv2.circle(mask_plat, (610, 367), 280, (255, 255, 255), -1)

        masked = cv2.bitwise_and(img, img, mask=mask_plat)


        imgColor, mask = myColorFinder.update(masked, hsvVals)
        imgContour, countours = cvzone.findContours(masked, mask)

        if countours:

            data = round((countours[0]['center'][0] - center_point[0]) / 10), \
                   round((h - countours[0]['center'][1] - center_point[1]) / 10), \
                   int(countours[0]['area'] - center_point[2])

            queue.put(data)
            #print("The got coordinates for the ball are :", data)

        else:
            data = 'nil'
            #queue.put(data)
            queue.put([0, 0, 0])

        imgStack = cvzone.stackImages([imgContour], 1, 1)

        # Trailing
        trailing_data = [0, 0]

        if data[0] == 'n' or data[1] == 'n':
            cv2.circle(imgStack, center= (center_point[0], center_point[1]), radius=42, color=(0, 255, 0), thickness=3)
            cv2.putText(img=imgStack, text='Referansepunkt', org=(center_point[0], center_point[1]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 255), thickness=1)
            cv2.imshow("Image", imgStack)
        else:
            trailing_data[0] = scaling(data[0], iMin=-26, iMax=26, qMin=350, qMax=870)
            trailing_data[1] = scaling(data[1], iMin=26, iMax=-26, qMin=88, qMax=610)

            trailing_data[0] = int(trailing_data[0])
            trailing_data[1] = int(trailing_data[1])

            pts.appendleft((trailing_data[0], trailing_data[1]))

            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue

                cv2.line(imgStack, pts[i - 1], pts[i], (0, 255, 0), 9)

            #Sikrel for refereansepunkt
            cv2.circle(imgStack, center=(center_point[0], center_point[1]), radius=42, color=(0, 255, 0), thickness=3)
            cv2.putText(img=imgStack, text='Referansepunkt', org=(center_point[0], center_point[1]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 255), thickness=1)
            #-------------------
            cv2.imshow("Image", imgStack)
        #-----

        # imgStack = cvzone.stackImages([img,imgColor, mask, imgContour],2,0.5) #use for calibration and correction
        #cv2.imshow("Image", imgStack)
        cv2.waitKey(1)



def servo_control(key2, queue):
    SERVO_HORN_LENGTH = 4.0  # lengths of servohornarm, in cm
    PLAT_CONNECTOR_RADIUS = 13  # distance from center of platform to magnetmount point, in cm
    PLATFORM_SIDE_LENGTH = np.sqrt(
        3) * PLAT_CONNECTOR_RADIUS  # sidelengths of the equilateral triangle made by the three connection points
    #-----------------------------------
    port_id = 'COM3'
    arduino = serial.Serial(port=port_id, baudrate=250000, timeout=0.1)
    if key2:
        print('Servo controls are initiated')

    def scaling(value, iMin, iMax, qMin, qMax):
        return qMin + (((value - (iMin)) / (iMax - iMin)) * (qMax - qMin))

    def write_arduino(data):
        #print('The angles send to the arduino : ', data)
        #print('The position of the ball : ', queue.get())
        arduino.write(bytes(data, 'utf-8'))

    def write_servo(rad1, rad2, rad3):

        angles: tuple = (rad1), (rad2), (rad3)

        write_arduino(str(angles))

    def movingAverage(x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    def butter_lowpass_filter(data, cutoff, fs, order):
        nyq = fs * 0.5
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def platAngleToServo(pitch, roll, heave):
        point1 = (((math.sqrt(3) * PLATFORM_SIDE_LENGTH) / 6) * math.sin(pitch) * math.cos(roll)
                  + (PLATFORM_SIDE_LENGTH / 2) * math.sin(roll) - heave)
        point2 = (((math.sqrt(3) * PLATFORM_SIDE_LENGTH) / 6) * math.sin(pitch) * math.cos(roll)
                  - (PLATFORM_SIDE_LENGTH / 2) * math.sin(roll) - heave)
        point3 = (((-math.sqrt(3) * PLATFORM_SIDE_LENGTH) / 3) * math.sin(pitch) * math.cos(roll) - heave)

        if point1 >= SERVO_HORN_LENGTH:
            point1 = SERVO_HORN_LENGTH - 0.0001
        if point2 >= SERVO_HORN_LENGTH:
            point2 = SERVO_HORN_LENGTH - 0.0001
        if point3 >= SERVO_HORN_LENGTH:
            point3 = SERVO_HORN_LENGTH - 0.0001

        if point1 <= -SERVO_HORN_LENGTH:
            point1 = -SERVO_HORN_LENGTH + 0.0001
        if point2 <= -SERVO_HORN_LENGTH:
            point2 = -SERVO_HORN_LENGTH + 0.0001
        if point3 <= -SERVO_HORN_LENGTH:
            point3 = -SERVO_HORN_LENGTH + 0.0001

        angles: tuple = (round(math.degrees(math.asin(point1 / SERVO_HORN_LENGTH)), 1),
                         round(math.degrees(math.asin(point2 / SERVO_HORN_LENGTH)), 1),
                         round(math.degrees(math.asin(point3 / SERVO_HORN_LENGTH)), 1))
        return angles

    pos_x_o = 0
    pos_y_o = 0
    stop_time = 0
    I_x = 0
    I_y = 0
    filter_data_x = np.zeros(20)
    filter_data_y = np.zeros(20)

    ref_pos_x = 0
    ref_pos_y = 13.538461538461547207

    while True:

        start_time = time.time()

        d_t = start_time-stop_time

        if d_t > 0.5:
            d_t = 0.1

        positions = queue.get()

        pos_x = (ref_pos_x) + (scaling(positions[0], iMin = -26, iMax = 26, qMin = -175, qMax = 175))
        pos_y = (ref_pos_y) + (scaling(positions[1], iMin = -26, iMax = 26, qMin = -175, qMax = 175))


        if pos_x_o != 0:
            speed_x = (pos_x - pos_x_o) / d_t
        else:
            speed_x = 0

        if pos_y_o != 0:
            speed_y = (pos_y - pos_y_o) / d_t
        else:
            speed_y = 0

        K1 = 0.4     #0.4 #K_p
        K2 = 0.8     #0.8 #K_d
        K3 = 0.25          #K_i

        I_reset = movingAverage((pos_x+pos_y)/2, 10)
        if I_reset == (ref_pos_x + ref_pos_y) / 2:
            I_x = 0
            I_y = 0
        I_x = I_x + (positions[0] * d_t)
        I_y = I_y + (positions[1] * d_t)

        data_x = np.append(filter_data_x,  speed_x)
        data_y = np.append(filter_data_y, speed_y)

        speed_x_filtered = butter_lowpass_filter(data = data_x, cutoff = 27, fs = 1000, order = 2)
        speed_y_filtered = butter_lowpass_filter(data = data_y, cutoff = 27, fs = 1000, order = 2)


        #u_i = P * D * I

        u_x = (-K1 * pos_x) + (-K2 * speed_x_filtered[len(speed_x_filtered) - 1]) + (-K3 * I_x)
        u_y = (-K1 * pos_y) + (-K2 * speed_y_filtered[len(speed_y_filtered) - 1]) + (-K3 * I_y)


        iMin = 1700
        iMax = 1700
        u_r = scaling(u_x, iMin = -iMin, iMax = iMax, qMin = -30*(math.pi/180) , qMax = 30*(math.pi/180) )
        u_p = scaling(u_y, iMin = -iMin, iMax = iMax, qMin = -30*(math.pi/180) , qMax = 30*(math.pi/180) )

        if (positions[0] != 0) and (positions[2] != 0):
            angles = platAngleToServo(u_p, u_r, 0)

            servo_ang_2 = scaling(angles[0], iMin=-25, iMax=25, qMin=0, qMax=90)
            servo_ang_1 = scaling(angles[1], iMin=-25, iMax=25, qMin=0, qMax=90)
            servo_ang_3 = scaling(angles[2], iMin=-25, iMax=25, qMin=0, qMax=90)

            print(servo_ang_1, servo_ang_2, servo_ang_3)

            write_servo(round((servo_ang_1),1), round((servo_ang_2),1), round((servo_ang_3),1))
        else:
            write_servo(45,45,45)

        pos_x_o = pos_x
        pos_y_o = pos_y
        stop_time = start_time

if __name__ == '__main__':

    queue = Queue() # The queue is done inorder for the communication between the two processes.
    key1 = 1 # just two dummy arguments passed for the processes
    key2 = 2
    p1 = mp.Process(target= ball_track, args=(key1, queue)) # initiate ball tracking process
    p2 = mp.Process(target=servo_control,args=(key2, queue)) # initiate servo controls
    p1.start()
    p2.start()
    #p1.join()
    #p2.join()