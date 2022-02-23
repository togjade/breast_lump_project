#!/usr/bin/python

import numpy as np
from math import pi
import serial
import time
from madgwick_py.madgwickahrs import MadgwickAHRS

AHRS1 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)
AHRS2 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)
AHRS3 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)
AHRS4 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)
AHRS5 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)
AHRS6 = MadgwickAHRS(sampleperiod = 1/100, beta = 10)   
AHRS=[AHRS1, AHRS2, AHRS3, AHRS4, AHRS5, AHRS6]
quaternion=[]

# conversion of raw IMU values to AHRS ready
def conversion_My(values):
    data_line=values
    
    m_min = np.zeros((6,3))
    m_min[0,:] = [ -4927,  -3709, -13970]
    m_min[1,:] = [ -4505,  -3986,  -4353]
    m_min[2,:] = [ -4412,  -4069,  -4037]
    m_min[3,:] = [ -4268,  -3916,  -4156]
    m_min[4,:] = [ -4058,  -3816,  -3689]
    m_min[5,:] = [ -3792,  -3711,  -1783]
    m_max = np.zeros((6,3))
    m_max[0,:] = [ +3275,  +3488,  +2830]
    m_max[1,:] = [ +3275,  +3343,  +2807]
    m_max[2,:] = [ +3487,  +3681,  +2827]
    m_max[3,:] = [ +3616,  +3853,  +2492]
    m_max[4,:] = [ +3945,  +3786,  +2945]
    m_max[5,:] = [ +3663,  +3946,  +5202]
    # default running_min = {32767, 32767, 32767}, running_max = {-32768, -32768, -32768};
    
    g_offset = np.zeros((6,3))
    g_offset[0,:] = [   1.7579,  -34.1473,  -13.2838]
    g_offset[1,:] = [  20.4412,  -14.2838,   -4.1726]
    g_offset[2,:] = [   8.0815,  -21.2029,  -21.4305]
    g_offset[3,:] = [   3.1176,  -13.7971,    9.8047]
    g_offset[4,:] = [   2.0164,   53.8477,   -2.7118]
    g_offset[5,:] = [  -1.0973,   -3.8622,    8.5626]
    
    
    
    for j in range(1,6):                           
        # dps(degrees per second)
        # gyro: 2000 dps full scale, normal power mode, all axes enabled, 100 Hz ODR 12.5Hz Bandwith
        values[9*(j)+1-1:9*(j)+3]=data_line[9*(j)+1-1:9*(j)+3]-g_offset[j,:] * 0.07 * (pi/180)  #  rad/s
        values[9*(j)+4-1:9*(j)+6]=data_line[9*(j)+4-1:9*(j)+6] * 0.000244  # accelerom: 8 g full scale, 16bit representation, all axes enabled, 100 Hz ODR
        values[9*(j)+7-1:9*(j)+9]=((data_line[9*(j)+7-1:9*(j)+9]-m_min[j,:])/(m_max[j,:]-m_min[j,:]) * 2 -1 ) #* 4/65535;  % magnetom 
        # compass.m_min = (LSM303::vector<int16_t>){-32767, -32767, -32767};
        # compass.m_max = (LSM303::vector<int16_t>){+32767, +32767, +32767};
        # gyroscope units must be radians
    
    
    return values

# connect to arduino
s = serial.Serial('/dev/ttyUSB0', 1000000, timeout=1)


# Start loop

s.reset_input_buffer()
try:
    while 1:
        s.write('a')
        data = s.readline()
        
        while len(data) < 110:
            s.reset_input_buffer()
            s.write('a')
            data = s.readline()s
            print ("here len ", len(data))

        
        values = np.zeros(55)
        
        for i in range(0,54):
            print ("here 3")
            values[i] = int(bin(data[2*(i-1)+2])[2:]+bin(data[2*(i-1)+1])[2:],2)
            print ("here 4")
        
        data_line = conversion_My(values)
        print ("here 1")
        current_set = []
        for j in range(0,6):
            AHRS[j].update(data_line[9*(j)+1-1:9*(j)+3],
                           data_line[9*(j)+4-1:9*(j)+6],
                           data_line[9*(j)+7-1:9*(j)+9]
                          ) # gyroscope units must be radians
            current_set.append(AHRS[j].quaternion._get_q())
        print(current_set)
        quaternion.append(current_set)
        print('------------- end loop ---------------')
except:
    print('program stopped')
