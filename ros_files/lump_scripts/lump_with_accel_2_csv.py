#!/usr/bin/env python3
from webbrowser import BackgroundBrowser
import pyaudio
import numpy as np
import rospy
import time
from imu_files.msg import Accel
#!/usr/bin/env python3
from lib2to3.pgen2.token import NEWLINE
from logging import shutdown
import re
import rospy
import copy
import os
import time
import csv
import pandas as pd
import numpy as np
from std_msgs.msg import Int32
from takktile_ros.msg import Touch, Contact
from imu_files.msg import  collect_data2, Accel, lump_states
from datetime import datetime
import threading 
from threading import Thread

# pressure data variables
###############
P = Touch()
P = [0.0] * 30
P_ = [0.0] * 30
P_filtered = [0.0] * 30
###############

###############
start_process = 0
A = Accel()
###############

# other functions' variables 
###############
counter_idx = 1
old_idx = 0
start_idx = 0
idx = 0
elapsed = 0
counter = 0
start = time.time()
dateTimeObj = datetime.now()
filename = str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S"))
data_now = pd.DataFrame()
data_all = pd.DataFrame()

###############
accel_data = Accel()
collect_data = collect_data2()
#####################################################
def highPass(z, z_, out):
    T = 0.2
    dt = 0.02
    res = (out + z - z_)*T/(T + dt)
    return res
#####################################################
#####################################################
def subscribe_data_array(data):
    global P, P_, P_filtered, data_P
    if(len(data.pressure) < 30):
        print('The number of pressure sensors: ', len(P))
        exit
    P_ = copy.deepcopy(P[:])
    for i in range(30):
        P[i] = data.pressure[i]
        P_filtered[i] = highPass(P[i], P_[i], P_filtered[i])
    data_P = P

def listener1():
    rospy.Subscriber("takktile/calibrated", Touch, subscribe_data_array)
#####################################################
def rec():       
    global start_idx, counter_idx, old_idx, P_filtered, P, idx, start_process, elapsed, data_all, data_now, counter, start,dateTimeObj, filename, accel_data
    old_idx = copy.deepcopy(idx)
    sum_pres = abs(sum(P_filtered))
    
    if sum_pres < 50:   
        sum_pres = abs(sum(P_filtered))
        idx = 0
    elif sum_pres > 50:
        idx = 1    

    if((idx == 1) & (old_idx == 0)):
        start_process = 1
        counter = counter + 1

    if (counter == 1):
        dateTimeObj = datetime.now()
        filename = str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S"))
        start = time.time()
        counter = counter + 1 
        counter_idx = counter_idx + 1
        if((counter_idx)%2 == 0):
            print("recording...")
    time_k = 15            
    if(start_process):

        if((counter_idx)%2 == 0):

            if (elapsed <= time_k): # some time period
                data_now = pd.concat([pd.DataFrame(P), pd.DataFrame(accel_data.accel1_x), pd.DataFrame(accel_data.accel1_y), pd.DataFrame(accel_data.accel2_x), pd.DataFrame(accel_data.accel2_y)], axis = 0)
                data_all = pd.concat([data_all, data_now.transpose()], axis = 0)

                elapsed = time.time() - start

            if (elapsed >= time_k):
                print("stopped recording...")   
                print("saving data") 
                data_all = pd.DataFrame(data_all)

                data_2_save = copy.deepcopy(data_all)
                data_2_save.to_csv(filename + ".csv")

                start_process = 0
                data_now = pd.DataFrame()
                data_all = pd.DataFrame()
                counter = 0
                elapsed = 0

        else:
            print("released...")
            elapsed = 0
            start_process = 0
            counter = 0
            time.sleep(3)
            print("You can start new trial")  

def accel_pub(stream):  
    global accel_data, sample
    data_a = stream.read(CHUNK)
    # read data from stream
    for i in range (CHUNK):
        for j in range(CHANNELS):
            sample[j,i]=int.from_bytes([data_a[j*2+i*8],data_a[j*2+i*8+1]], "little", signed=True)     
    sample = sample/32768      
    accel_data.accel1_x = sample[0,:].transpose()
    accel_data.accel1_y = sample[1,:].transpose()
    accel_data.accel2_x = sample[2,:].transpose()
    accel_data.accel2_y = sample[3,:].transpose()

    pub.publish(accel_data)
    sample = np.zeros([CHANNELS, CHUNK])

def talker():

    global P, P_, P_filtered, accel_data
    global stream, CHUNK, CHANNELS, audio, sample
    FORMAT = pyaudio.paInt16
    CHANNELS = 4
    RATE = 8000
    CHUNK = 50
    audio = pyaudio.PyAudio()
        
    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=CHUNK,input_device_index=10)

    sample = np.zeros([CHANNELS, CHUNK])   

    print("started accel...")

    while not rospy.is_shutdown():
        accel_pub(stream)
        rec()
        rate.sleep()

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == '__main__':
    
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records"

    os.chdir(dir1)

    rospy.init_node('talker', anonymous=True)

    pub = rospy.Publisher('lump_data', Accel, queue_size = 100)

    rate = rospy.Rate(160) #10hz

    listener1()

    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()
    
    
