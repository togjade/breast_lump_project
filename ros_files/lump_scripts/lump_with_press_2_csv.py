#!/usr/bin/env python3
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
from imu_files.msg import  collect_data2, Accel
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
counter_idx = 0
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
# data_all = [0.0] * 30
###############
accel_data = Accel()
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
    
#####################################################
def listener1():
    rospy.Subscriber("takktile/calibrated", Touch, subscribe_data_array)
    rec()
#####################################################
def rec():       
    # while not rospy.is_shutdown():
    global start_idx, counter_idx, old_idx, P_filtered, P, idx, start_process, elapsed, data_all, data_now, counter, start,dateTimeObj, filename
    old_idx = copy.deepcopy(idx)
    sum_pres = abs(sum(P_filtered))
    
    if sum_pres < 50:   
        sum_pres = abs(sum(P_filtered))
        idx = 0
    elif sum_pres > 50:
        idx = 1    

    if((idx == 1) & (old_idx == 0)):
        # print("idx", idx)
        # print("old_idx", old_idx)
        start_process = 1
        counter = counter + 1
        # counter_idx = counter_idx + 1

    if (counter == 1):
        dateTimeObj = datetime.now()
        filename = str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S"))
        start = time.time()
        counter = counter + 1 
        counter_idx = counter_idx + 1
        if((counter_idx)%2 == 0):
            print("recording...1")
        # print(counter_idx)
    if(start_process):

        if((counter_idx)%2 == 0):
            # print("here")
            # print("recording...")
            if (elapsed <= 5): # some time period
                # print(np.shape(P.transpose()))
                # data_now = pd.concat([pd.DataFrame(P), pd.DataFrame(accel_data.accel1_x), pd.DataFrame(accel_data.accel1_y), pd.DataFrame(accel_data.accel2_x), pd.DataFrame(accel_data.accel2_y)], axis = 0)
                # data_all = data_all.values[:]
                P_d = pd.DataFrame(P)
                # P_d = P_d.transpose()
                # data_all = np.concatenate([(data_all), ((P))], axis = 1)
                data_all = pd.concat([data_all, (P_d.transpose())], axis = 0)

                elapsed = time.time() - start
                # print(data)

            if (elapsed >= 5):
                print("stopped recording...")   
                print("saving data") 
                data_all = pd.DataFrame(data_all)
                data_all.to_csv(filename + ".csv")
                
                # save_data(data_all)
                print("Data is saved")
                start_process = 0
                data_now = pd.DataFrame()
                data_all = pd.DataFrame()
                counter = 0
                elapsed = 0
                # print(counter_idx)

        else:
            print("released...")
            # print(counter_idx)
            elapsed = 0
            start_process = 0
            counter = 0
            time.sleep(3)
            print("You can start new trial")  
        # rate.sleep()


def talker():
    while not rospy.is_shutdown():
        rec()
        rate.sleep()
  
if __name__ == '__main__':
    # dir1 = "/media/togzhan/Samsung_T5/project files/lump_project/data/trial_records"

    # dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records/Karina_trial_1/without"
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records/Karina_trial_1/with_big"

    os.chdir(dir1)

    rospy.init_node('talker', anonymous=True)

    pub = rospy.Publisher('accel_data', Accel, queue_size = 100)

    rate = rospy.Rate(100) #10hz

    listener1()

    try:
        talker()
        # listener1()
    except rospy.ROSInterruptException:
        pass
    
    rospy.spin()
    
    
