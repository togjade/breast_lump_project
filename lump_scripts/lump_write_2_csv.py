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

def subscribe_data_accel(data):
    global accel_data
    accel_data.accel1_x = data.accel1_x
    accel_data.accel1_y = data.accel1_y
    accel_data.accel2_x = data.accel2_x
    accel_data.accel2_y = data.accel2_y
#####################################################
def listener1():
    rospy.Subscriber("takktile/calibrated", Touch, subscribe_data_array)
    rospy.Subscriber("accel_data", Accel, subscribe_data_accel)
#####################################################
def rec():       
    # while not rospy.is_shutdown():
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
    if(start_process):

        if((counter_idx)%2 == 0):

            if (elapsed <= 5): # some time period
                # data_now = pd.concat([pd.DataFrame(P), pd.DataFrame(accel_data.accel1_x), pd.DataFrame(accel_data.accel1_y), pd.DataFrame(accel_data.accel2_x), pd.DataFrame(accel_data.accel2_y)], axis = 0)
                # data_all = pd.concat([data_all, data_now.transpose()], axis = 1)
                data_now = pd.concat([pd.DataFrame(P), pd.DataFrame(accel_data.accel1_x), pd.DataFrame(accel_data.accel1_y), pd.DataFrame(accel_data.accel2_x), pd.DataFrame(accel_data.accel2_y)], axis = 0)
                    # print(np.shape((data_now.transpose())))
                data_all = pd.concat([data_all, data_now.transpose()], axis = 0)

                elapsed = time.time() - start

            if (elapsed >= 5):
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

# class AsyncWrite(threading.Thread):
#    def __init__(self):
#       threading.Thread.__init__(self)
#     #   self.data_all = data_all
#     #   self.filename = filename

#    def run(self):
#       accel_pub(stream)
#       time.sleep(3)
    #   print ("Finished Background file write to " + self.filename)

# def save_data(data_all, filename):

#     data_2_save = copy.deepcopy(data_all)
#     background = AsyncWrite(data_2_save,filename)
#     backgroun#####################################################
# def accel_pub(stream):

#     global sample, accel_data

#     data_a = stream.read(CHUNK)
#     # read data from stream
#     for i in range (CHUNK):
#         for j in range(CHANNELS):
#             sample[j,i]=int.from_bytes([data_a[j*2+i*8],data_a[j*2+i*8+1]], "little", signed=True)     
#     sample = sample/32768      
#     accel_data.accel1_x = sample[0,:].transpose()
#     accel_data.accel1_y = sample[1,:].transpose()
#     accel_data.accel2_x = sample[2,:].transpose()
#     accel_data.accel2_y = sample[3,:].transpose()

#     pub.publish(accel_data)

#     # rec(accel_data)

#     sample = np.zeros([CHANNELS, CHUNK])

#     background.join()
    # data_2_save.to_csv(filename + ".csv")

# #####################################################
# def accel_pub(stream):

#     global sample, accel_data

#     data_a = stream.read(CHUNK)
#     # read data from stream
#     for i in range (CHUNK):
#         for j in range(CHANNELS):
#             sample[j,i]=int.from_bytes([data_a[j*2+i*8],data_a[j*2+i*8+1]], "little", signed=True)     
#     sample = sample/32768      
#     accel_data.accel1_x = sample[0,:].transpose()
#     accel_data.accel1_y = sample[1,:].transpose()
#     accel_data.accel2_x = sample[2,:].transpose()
#     accel_data.accel2_y = sample[3,:].transpose()

#     pub.publish(accel_data)

#     # rec(accel_data)

#     sample = np.zeros([CHANNELS, CHUNK])

        # rate.sleep()

def talker():
    global P, P_, P_filtered, accel_data

    while not rospy.is_shutdown():
        rec()
        collect_data.pressure = P
        collect_data.accel1_x = accel_data.accel1_x
        collect_data.accel1_y = accel_data.accel1_y
        collect_data.accel2_x = accel_data.accel2_x
        collect_data.accel2_y = accel_data.accel2_y
        rate.sleep()

    # data = stream.read(CHUNK)
        
    # while not rospy.is_shutdown():
    #     data = stream.read(CHUNK)
    #     # read data from stream
    #     for i in range (CHUNK):
    #         for j in range(CHANNELS):
    #             sample[j,i]=int.from_bytes([data[j*2+i*8],data[j*2+i*8+1]], "little", signed=True)
                
    #     sample = sample/32768    
        
    #     accel_data.accel1_x = sample[0,:].transpose()
    #     accel_data.accel1_y = sample[1,:].transpose()
    #     accel_data.accel2_x = sample[2,:].transpose()
    #     accel_data.accel2_y = sample[3,:].transpose()
    #     #accel_data.accel3_x = sample[4,:].transpose()
    #     #accel_data.accel3_y = sample[5,:].transpose()
    #     pub.publish(accel_data)
    #     sample = np.zeros([CHANNELS, CHUNK])
    #     rec(accel_data)
    #     # background.join()
        

    # stream.stop_stream()
    # stream.close()
    # audio.terminate()
   


if __name__ == '__main__':
    
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records"

    os.chdir(dir1)

    rospy.init_node('talker', anonymous=True)

    pub = rospy.Publisher('lump_data', collect_data2, queue_size = 100)

    rate = rospy.Rate(160) #10hz

    listener1()

    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()
    
    
