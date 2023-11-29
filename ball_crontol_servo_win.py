import multiprocessing as mp
from multiprocessing import Queue
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
import cv2
import serial
import time
import array
from scipy.signal import butter, filtfilt
from tkinter import *

hmin_v = 30
hmax_v = 90
smin_v = 30
smax_v = 255
vmin_v = 30
vmax_v = 255

# define servo angles and set a value
servo1_angle = 0
servo2_angle = 0
servo3_angle = 0
all_angle = 0

# Set a limit to upto which you want to rotate the servos (You can do it according to your needs)

servo1_angle_limit_positive = 40
servo1_angle_limit_negative = -50

servo2_angle_limit_positive = 30
servo2_angle_limit_negative = -73

servo3_angle_limit_positive = 40
servo3_angle_limit_negative = -53


def ball_track(key1, queue):
    prevX = 0
    prevY = 0

    camera_port = 1
    cap = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    cap.set(3, 960)
    cap.set(4, 540)

    get, img = cap.read()
    h, w, _ = img.shape

    if key1:
        print('Ball tracking is initiated')

    myColorFinder = ColorFinder(
        FALSE)  # if you want to find the color and calibrate the program we use this *(Debugging)
    hsvVals = {'hmin': hmin_v, 'smin': smin_v, 'vmin': vmin_v, 'hmax': hmax_v, 'smax': smax_v,
               'vmax': vmax_v}

    center_point = [480, 270, 2210]  # center point of the plate, calibrated

    while True:
        get, img = cap.read()
        mask_plat = np.zeros(img.shape[:2], dtype='uint8')
        cv2.circle(mask_plat, (480, 270), 240, (255, 255, 255), -1)

        # Make circular mask
        masked = cv2.bitwise_and(img, img, mask=mask_plat)

        imgColor, mask = myColorFinder.update(masked, hsvVals)
        imgContour, countours = cvzone.findContours(masked, mask, minArea=2000, maxArea=5500)

        x = -120
        y = -120

        if countours:
            x = round((countours[0]['center'][0]))
            y = round((countours[0]['center'][1]))

            data = round((countours[0]['center'][0] - center_point[0]) / 1), \
                round((h - countours[0]['center'][1] - center_point[1]) / 1), \
                round(int(countours[0]['area'] - center_point[2]) / 100)

            queue.put(data)
        else:
            data = 'nil'  # returns nil if we cant find the ball
            queue.put(data)

        imgStack = cvzone.stackImages([imgContour], 1, 1)
        #imgStack = cvzone.stackImages([img, imgColor, mask, imgContour], 2, 0.5)  # use for calibration and correction
        #cv2.circle(imgStack, (center_point[0], center_point[1]), 270, (255, 20, 20), 6)
        cv2.circle(imgStack, (center_point[0], center_point[1]), 2, (20, 20, 255), 2)

        cv2.circle(imgStack, (x, y), 5, (20, 20, 255), 2)
        cv2.circle(imgStack, (x, y), 40, (180, 120, 255), 2)

        vector = [prevX - x, prevY - y]
        cv2.arrowedLine(imgStack, (x, y), (x - vector[0] * 10, y - vector[1] * 10), (39, 237, 250), 4)

        cv2.circle(imgStack,
                   (center_point[0] + int(80 * np.cos(time.time())), center_point[1] - int(80 * np.sin(time.time()))),
                   5, (20, 20, 255), 2)

        prevX = x
        prevY = y

        cv2.imshow("Image", imgStack)
        start_time = time.time()
        cv2.waitKey(1)


def servo_control(key2, queue):
    port_id = 'COM3'
    # initialise serial interface
    arduino = serial.Serial(port=port_id, baudrate=250000, timeout=0.1)

    if key2:
        print('Servo controls are initiated')

    def kinematics(Z, rotZdeg, rotYdeg, rotXdeg):
        L = 19
        R = 4

        # Convert degrees to radians
        rotZ = np.deg2rad(rotZdeg)
        rotY = np.deg2rad(rotYdeg)
        rotX = np.deg2rad(rotXdeg)

        # Define the position matrix
        pos = np.array([
            [L / 2, -(L / 2), 0],
            [L / (2 * np.sqrt(3)), L / (2 * np.sqrt(3)), -(L / np.sqrt(3))],
            [Z, Z, Z]
        ])

        # Define the rotation matrix for Y and X axes
        rotations_matrix_yx = np.array([
            [np.cos(0) * np.cos(rotY), np.cos(0) * np.sin(rotY) * np.sin(rotX) - np.sin(0) * np.cos(rotX),
             np.cos(0) * np.sin(rotY) * np.cos(rotX) + np.sin(0) * np.sin(rotX)],
            [np.sin(0) * np.cos(rotY), np.sin(0) * np.sin(rotY) * np.sin(rotX) + np.cos(0) * np.cos(rotX),
             np.sin(0) * np.sin(rotY) * np.cos(rotX) - np.cos(0) * np.sin(rotX)],
            [-np.sin(rotY), np.cos(rotY) * np.sin(rotX), np.cos(rotY) * np.cos(rotX)]
        ])

        # Define the rotation matrix for Z axis
        rotations_matrix_z = np.array([
            [np.cos(rotZ), -np.sin(rotZ), 0],
            [np.sin(rotZ), np.cos(rotZ), 0],
            [0, 0, 1]
        ])

        # Apply rotations in sequence
        newpos = np.dot(rotations_matrix_z, pos)
        all_the_rot = np.dot(rotations_matrix_yx, newpos)

        # Calculate angles using arcsin and normalize them
        angle1 = np.rad2deg(np.arcsin(np.clip(all_the_rot[2, 0] / R, -1, 1)))
        angle2 = np.rad2deg(np.arcsin(np.clip(all_the_rot[2, 1] / R, -1, 1)))
        angle3 = np.rad2deg(np.arcsin(np.clip(all_the_rot[2, 2] / R, -1, 1)))

        # Clip angles to be within -50 and 50 degrees
        angle1 = np.clip(angle1, -43, 43)
        angle2 = np.clip(angle2, -43, 43)
        angle3 = np.clip(angle3, -43, 43)
        return angle1, angle2, angle3

    root = Tk()

    # root.resizable(0,0)

    def butter_lowpass_filter(data, cutoff, fs,
                              order):  # https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
        normal_cutoff = cutoff / (0.5 * fs)
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def map_x_to_y(value, x_min, x_max, y_min, y_max):
        return y_min + (((value - x_min) / (x_max - x_min)) * (y_max - y_min))

    def get_ball_pos():
        cord_info = queue.get()
        # print("Yooooo: ", cord_info[0], " ", cord_info[1])
        return cord_info

    def P_Reg(pos_x, pos_y):  # out = kp*e   e = reff - pos
        kp = 1.2
        reff_val_x = 0
        reff_val_y = 0
        if (pos_x == 'nil') or (pos_y == 'nil'):
            pos_x = 0
            pos_y = 0

        error_x = reff_val_x - pos_x
        error_y = reff_val_y - pos_y
        output_x = error_x * kp
        output_y = error_y * kp
        return output_x, output_y

    def ballpos_to_servo_angle(x_cord, y_cord):
        """ convert the distance to center to angle.
        x_ang = map_x_to_y(x_cord/10, x_min=28.00, x_max=-28.00, y_min=-35.00, y_max=35.00)
        y_ang = map_x_to_y(y_cord/10, x_min=28.00, x_max=-28.00, y_min=-35.00, y_max=35.00)
        """
        # x and y angle(deg) in and servoangle out(rad)
        servo_ang1, servo_ang2, servo_ang3 = kinematics(0, 22, y_cord, x_cord)
        # print("angle", servo_ang1, " ", (servo_ang2), " ", (servo_ang3))

        return servo_ang1, servo_ang2, servo_ang3

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
        # print('The angles send to the arduino : ', data)
        arduino.write(bytes(data, 'utf-8'))

    kp = 0.39 #0.39
    ki = 0.64 #0.62
    kd = 0.345 #0.32
    integral_error_x = 0
    integral_error_y = 0
    last_error_x = 0
    last_error_y = 0
    start_time = 0
    deriv_data_x = array.array('f', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    deriv_data_y = array.array('f', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    while key2:

        cord_info = get_ball_pos()  # Ballpos
        reff_val_x = 0#(80*np.cos(time.time()))/10
        reff_val_y = 0#(80*np.sin(time.time()))/10

        if cord_info == 'nil':
            reff_val_x = 0
            reff_val_y = 0
            integral_error_x = 0
            integral_error_y = 0
            last_error_x = 0
            last_error_y = 0
            pos_x = 0
            pos_y = 0
        else:
            pos_x = (float(cord_info[0]) / 10)
            pos_y = (float(cord_info[1]) / 10)
            print("pos_x:", pos_x)
            print("pos_y:", pos_y)

        dt = time.time() - start_time
        # print("dt: Matte: ", dt)
        error_x = reff_val_x - pos_x
        error_y = reff_val_y - pos_y
        integral_error_x += error_x * dt
        integral_error_y += error_y * dt
        #print("integral error:  ", integral_error_x, "   ", integral_error_y)
        deriv_error_x = (error_x - last_error_x) / dt
        deriv_error_y = (error_y - last_error_y) / dt
        print("D_error_x ", deriv_error_x)
        print("D_error_y ", deriv_error_y)

        if deriv_error_x != 0:
            deriv_data_x.append(deriv_error_x)

        if deriv_error_y != 0:
            deriv_data_y.append(deriv_error_y)


        #print("combinded_data x ", deriv_data_x)
        #print("combinded_data y ", deriv_data_y)

        if ((len(deriv_data_x) >= 15) and (len(deriv_data_y) >= 15) and -2 < deriv_error_x < 2 and -2 < deriv_error_y < 2):
            filtered_error_data_x = butter_lowpass_filter(data=deriv_data_x, cutoff=0.5, fs=100, order=2)
            filtered_error_data_y = butter_lowpass_filter(data=deriv_data_y, cutoff=0.5, fs=100, order=2)
            print("Filter_data x-----------", filtered_error_data_x[len(filtered_error_data_x) - 1])
            print("Filter_data y-----------", filtered_error_data_y[len(filtered_error_data_y) - 1])
            filter_deriv_error_x = filtered_error_data_x[len(filtered_error_data_x)-1]
            filter_deriv_error_y = filtered_error_data_y[len(filtered_error_data_y)-1]
        else:
            filter_deriv_error_x = deriv_error_x
            filter_deriv_error_y = deriv_error_y


        # print("deriv error:  ", deriv_error_x, "   ", deriv_error_y)
        last_error_x = error_x
        last_error_y = error_y

        output_x = (-kp * error_x) + (-ki * integral_error_x) + (-kd * deriv_error_x)
        output_y = (-kp * error_y) + (-ki * integral_error_y) + (-kd * deriv_error_y)
        #output_x = (-kp * error_x) + (-ki * integral_error_x) + (-kd * filter_deriv_error_x)
        #output_y = (-kp * error_y) + (-ki * integral_error_y) + (-kd * filter_deriv_error_y)
        print(output_x, "   ", output_y)

        servo_ang1, servo_ang2, servo_ang3 = ballpos_to_servo_angle(output_x, output_y)  # Ballpos to servo angle
        filter_write_angle_servo(servo_ang1, servo_ang2, servo_ang3)  # Servo angle to arduino

        start_time = time.time()
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
