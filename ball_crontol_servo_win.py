import multiprocessing as mp
from multiprocessing import Queue
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2
import serial
import time
from tkinter import *

hmin_v = 21
hmax_v = 45
smin_v = 123
smax_v = 255
vmin_v = 193
vmax_v = 255
# define servo angles and set a value
servo1_angle = 0
servo2_angle = 0
servo3_angle = 0
all_angle = 0

# Set a limit to upto which you want to rotate the servos (You can do it according to your needs)
servo1_angle_zero = 11.2
servo1_angle_limit_positive = 40
servo1_angle_limit_negative = -50

servo2_angle_zero = -9.3
servo2_angle_limit_positive = 30
servo2_angle_limit_negative = -73

servo3_angle_zero = 0
servo3_angle_limit_positive = 40
servo3_angle_limit_negative = -53


def ball_track(key1, queue):
    camera_port = 0
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    get, img = cap.read()
    h, w, _ = img.shape

    if key1:
        print('Ball tracking is initiated')

    myColorFinder = ColorFinder(
        False)  # if you want to find the color and calibrate the program we use this *(Debugging)
    hsvVals = {'hmin': hmin_v, 'smin': smin_v, 'vmin': vmin_v, 'hmax': hmax_v, 'smax': smax_v,
               'vmax': vmax_v}

    center_point = [640, 360, 2210]  # center point of the plate, calibrated

    while True:
        get, img = cap.read()
        mask_plat = np.zeros(img.shape[:2], dtype='uint8')
        cv2.circle(mask_plat, (640, 360), 300, (255, 255, 255), -1)

        # Make circular mask
        masked = cv2.bitwise_and(img, img, mask=mask_plat)
        imgColor, mask = myColorFinder.update(masked, hsvVals)
        imgContour, countours = cvzone.findContours(masked, mask, minArea=3000, maxArea=5500)

        if countours:
            data = round((countours[0]['center'][0] - center_point[0]) / 10), \
                round((h - countours[0]['center'][1] - center_point[1]) / 10), \
                round(int(countours[0]['area'] - center_point[2]) / 100)

            queue.put(data)
        else:
            data = 'nil'  # returns nil if we cant find the ball
            queue.put(data)

        imgStack = cvzone.stackImages([imgContour], 1, 1)
        # mgStack = cvzone.stackImages([img, imgColor, mask, imgContour], 2, 0.5)  # use for calibration and correction
        cv2.imshow("Image", imgStack)
        cv2.waitKey(1)


def servo_control(key2, queue):
    port_id = 'COM4'
    # initialise serial interface
    arduino = serial.Serial(port=port_id, baudrate=250000, timeout=0.1)

    if key2:
        print('Servo controls are initiated')

    def rotatematrix(Z, rotZdeg, rotYdeg, rotXdeg):
        L = 21
        R = 4

        rotZ = (rotZdeg)  # Example value for rotation around Z-axis in degrees
        rotY = (rotYdeg / 10)  # Example value for rotation around Y-axis in degrees
        rotX = (rotXdeg / 10)  # Example value for rotation around X-axis in degrees

        # Define the position matrix
        pos = np.array([
            [L / 2, -(L / 2), 0],
            [L / (2 * np.sqrt(3)), L / (2 * np.sqrt(3)), -(L / np.sqrt(3))],
            [0, 0, 0]
        ])

        # Define the rotation matrix
        rotations_matrix = np.array([
            [np.cos(0) * np.cos(rotY), np.cos(0) * np.sin(rotY) * np.sin(rotX) - np.sin(0) * np.cos(rotX),
             np.cos(0) * np.sin(rotY) * np.cos(rotX) + np.sin(0) * np.sin(rotX)],
            [np.sin(0) * np.cos(rotY), np.sin(0) * np.sin(rotY) * np.sin(rotX) + np.cos(0) * np.cos(rotX),
             np.sin(0) * np.sin(rotY) * np.cos(rotX) - np.cos(0) * np.sin(rotX)],
            [-np.sin(rotY), np.cos(rotY) * np.sin(rotX), np.cos(rotY) * np.cos(rotX)]
        ])

        rotations_matrix_z = np.array([
            [np.cos(rotZ) * np.cos(0), np.cos(rotZ) * np.sin(0) * np.sin(0) - np.sin(rotZ) * np.cos(0),
             np.cos(rotZ) * np.sin(0) * np.cos(0) + np.sin(rotZ) * np.sin(0)],
            [np.sin(rotZ) * np.cos(0), np.sin(rotZ) * np.sin(0) * np.sin(0) + np.cos(rotZ) * np.cos(0),
             np.sin(rotZ) * np.sin(0) * np.cos(0) - np.cos(rotZ) * np.sin(0)],
            [-np.sin(0), np.cos(0) * np.sin(0), np.cos(0) * np.cos(0)]
        ])

        newpos = np.dot(rotations_matrix_z, pos)
        # Perform matrix multiplication
        all_the_rot = np.dot(rotations_matrix, newpos)

        # print(rotZ, rotY, rotX)
        # print(all_the_rot[0, 2] / R, all_the_rot[1, 2] / R, all_the_rot[2, 2] / R)

        angle1 = np.arcsin((all_the_rot[2, 0]) + Z / R)
        angle2 = np.arcsin((all_the_rot[2, 1]) + Z / R)
        angle3 = np.arcsin((all_the_rot[2, 2]) + Z / R)

        return angle1, angle2, angle3

    root = Tk()

    # root.resizable(0,0)

    def map_x_to_y(value, x_min, x_max, y_min, y_max):
        return y_min + (((value - x_min) / (x_max - x_min)) * (y_max - y_min))

    def get_ball_pos():
        corrd_info = queue.get()
        return corrd_info[0], corrd_info[1]

    def P_Reg(pos_x, pos_y):  # out = kp*e   e = reff - pos
        kp = 1.7
        reff_val_x = 0
        reff_val_y = 0
        if (pos_x == 'nil') or (pos_y == 'nil'):
            output_x = 0
            output_y = 0
        else:
            error_x = reff_val_x - pos_x
            error_y = reff_val_y - pos_y
            output_x = error_x * kp
            output_y = error_y * kp
        return output_x, output_y

    def ballpos_to_servo_angle(x_pos, y_pos):
        x_cord, y_cord = P_Reg(x_pos, y_pos)

        # convert the distance to center to angle.
        x_ang = map_x_to_y(x_cord, x_min=28, x_max=-28, y_min=-20, y_max=20)
        y_ang = map_x_to_y(y_cord, x_min=28, x_max=-28, y_min=-20, y_max=20)

        # x and y angle(deg) in and servoangle out(rad)
        servo_ang1, servo_ang2, servo_ang3 = rotatematrix(0, 22, y_ang, x_ang)
        # print("angle", np.rad2deg(servo_ang1), " ", np.rad2deg(servo_ang2), " ", np.rad2deg(servo_ang3))

        return np.rad2deg(servo_ang1), np.rad2deg(servo_ang2), np.rad2deg(servo_ang3)

    def filter_write_angle_servo(servo1_angle_deg, servo2_angle_deg, servo3_angle_deg):
        if (-90 < servo1_angle_deg < 90) and (-90 < servo2_angle_deg < 90) and (-90 < servo3_angle_deg < 90):
            write_servo(servo1_angle_deg, servo2_angle_deg, servo3_angle_deg)
        else:
            write_servo(0, 0, 0)

    def write_servo(ang1, ang2, ang3):
        angles: tuple = (round(ang1, 1),
                         round(ang2, 1),
                         round(ang3, 1))
        write_arduino(str(angles))

    def write_arduino(data):
        print('The angles send to the arduino : ', data)
        arduino.write(bytes(data, 'utf-8'))

    while key2:
        pos_x, pos_y = get_ball_pos()  # Ballpos
        servo_ang1, servo_ang2, servo_ang3 = ballpos_to_servo_angle(pos_x, pos_y)  # Ballpos to servo angle
        filter_write_angle_servo(servo_ang1, servo_ang2, servo_ang3)  # Servo angle to arduino
    root.mainloop()  # running loop


if __name__ == '__main__':
    queue = Queue()  # The queue is done inorder for the communication between the two processes.
    key1 = 1  # just two dummy arguments passed for the processes
    key2 = 2
    p1 = mp.Process(target=ball_track, args=(key1, queue))  # initiate ball tracking process
    p2 = mp.Process(target=servo_control, args=(key2, queue))  # initiate servo controls
    p1.start()
    p2.start()
    p1.join()
    p2.join()
