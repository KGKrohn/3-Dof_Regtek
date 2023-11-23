import time
import math
import numpy as np
d_t=0.1
x_pos=0
y_pos=0
prev_x_pos=0
prev_y_pos=0
stop_time=0

prev_positions=[0, 0, 0]
kp=1
x=1
y=1
z=1
refferanse_x=0
refferanse_y=0
positions= [x,y,z]

def scaling(value, aMin, aMax, bMin, bMax):
    return bMin + (((value - (aMin)) / (aMax - aMin)) * (bMax - bMin))


x_direction = (prev_positions[0]-positions[0])/d_t
y_direction = (prev_positions[1]-positions[1])/d_t

pos_x = refferanse_x + (scaling(positions[0], aMin=-28, aMax=28, bMin=-175, bMax=175))
pos_y = refferanse_y + (scaling(positions[1], aMin=-28, aMax=28, bMin=-175, bMax=175))


def P_Reg():
    if (positions[0] != 'nil'):
        error_x = refferanse_x - positions[0]
        error_y = refferanse_y - positions[1]
        output_x = error_x * kp
        output_y = error_y * kp
    else:
        output_x= 0
        output_y= 0

    return output_x, output_y







while True:
    start_time = time.time()
    d_t = start_time - stop_time
    positions=[0, 0, 0]#queue.get()




    pos_x_o = pos_x
    pos_y_o = pos_y
    stop_time = start_time