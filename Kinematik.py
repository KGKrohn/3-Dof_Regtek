import numpy as np

xrotdeg= 0
xrot = np.deg2rad(xrotdeg)
yrotdeg=10
yrot = np.deg2rad(yrotdeg)
z=1
L=19
r=4
A= np.array([
    [(6*L*np.cos(yrot)-np.sqrt(3)*L*np.cos(xrot+yrot)+np.sqrt(3)*L*np.cos(xrot-yrot))/12,       (-6*L*np.cos(yrot)-np.sqrt(3)*L*np.cos(xrot+yrot)+np.sqrt(3)*L*np.cos(xrot-yrot))/12,       (np.sqrt(3)*L*np.cos(xrot+yrot)+np.sqrt(3)*L*np.cos(xrot-yrot))/6],
    [(np.sqrt(3)*L*np.cos(xrot))/6,                                                             (np.sqrt(3)*L*np.cos(xrot))/6,                                                              (-np.sqrt(3)*L*np.cos(xrot))/3],
    [(-6*L*np.sin(yrot)+np.sqrt(3)*L*np.sin(xrot+yrot)+np.sqrt(3)*L*np.sin(xrot-yrot))/12+z,    (6*L*np.sin(yrot)+np.sqrt(3)*L*np.sin(xrot+yrot)+np.sqrt(3)*L*np.sin(xrot-yrot))/12+z,      (-np.sqrt(3)*L*np.sin(xrot+yrot)-np.sqrt(3)*L*np.sin(xrot-yrot))/6+z]
])
print(A)

Va1= np.arcsin(A[2,0]/r)
Va2= np.arcsin(A[2,1]/r)
Va3= np.arcsin(A[2,2]/r)
print(Va1)
print(Va2)
print(Va3)

if ((servo1_angle_limit_negative < corrd_info[0] < servo1_angle_limit_positive)
        and (servo2_angle_limit_negative < corrd_info[1] < servo2_angle_limit_positive)
        and (servo3_angle_limit_negative < corrd_info[2] < servo3_angle_limit_positive)):
    all_angle_assign(corrd_info[0], corrd_info[1], corrd_info[2])