import time
import numpy as np

x_pos=0
y_pos=0
prev_x_pos=0
prev_y_pos=0
stop_time=0

prev_positions=[0, 0, 0]

def speed(vec_pos,vec_pos_prev,d_t):
    Speed_V = (vec_pos_prev-vec_pos)/d_t
    return Speed_V


while True:
    start_time = time.time()
    d_t = start_time - stop_time
    positions=[0, 0, 0]#queue.get()

    #pos_x =
    #pos_y =

    stop_time = start_time
    prev_positions = positions