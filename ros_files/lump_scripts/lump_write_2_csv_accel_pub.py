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

# pressure data variables
###############
P = Touch()
P = [0.0] * 30
P_ = [0.0] * 30
P_filtered = [0.0] * 30
###############

# accel and imu data variables
###############
# imu = imu_data()
states  = lump_states()
A = Accel()
# message = collect_data2()
###############

# other functions' variables 
###############
counter_idx = 1
old_idx = 0
start_idx = 0
idx = 0
###############
#####################################################
def create_column_label():
    label = []
    # label_a = []
    str3 = "p"
    str4 = ["ax1", "ay1", "ax2", "ay2"]
    for i in range(30):
        str3 = str3 + str(i)
        label.append(str3)
        str3 = "p"
    for j in str4:
        for k in range(50):
            # s = "str" + str(j)
            label.append(j)
    # label.append(label_a)
    label = pd.DataFrame(label)
    label = label.transpose()
    label.to_csv("column_labels.csv")
    # print(label_a)
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

#####################################################
def subscribe_data_accel(data):
    global A
    A.accel1_x = data.accel1_x
    A.accel1_y = data.accel1_y
    A.accel2_x = data.accel2_x
    A.accel2_y = data.accel2_y
    # print(A)
#####################################################

#####################################################
def listener1():
    rospy.Subscriber("takktile/calibrated", Touch, subscribe_data_array)
    rospy.Subscriber("accel_data", Accel, subscribe_data_accel)
    # rec()
    # print("here5")
    # rospy.Subscriber("imu__data", imu_data, subscribe_data_imu)
#####################################################

#####################################################
def highPass(z, z_, out):
    T = 0.2
    dt = 0.02
    res = (out + z - z_)*T/(T + dt)
    return res
#####################################################

#####################################################
def rec():        
    global start_idx, counter_idx, old_idx, labels, P_filtered, idx

    while not rospy.is_shutdown():
        old_idx = copy.deepcopy(idx)
        sum_pres = abs(sum(P_filtered))
        if sum_pres < 50:   
            sum_pres = abs(sum(P_filtered))
            idx = 0
            # pub_rec.publish(0)
        elif sum_pres > 50:
            idx = 1
            # pub_rec.publish(1)

        # print(labels)
        data_now = pd.DataFrame()
        data = pd.DataFrame()
        start = time.time()         #the variable that holds the starting time
        dateTimeObj = datetime.now()
        filename = str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S"))
        elapsed = 0 
        # print("here")
        if((idx == 1) & (old_idx == 0)):
            if((counter_idx)%2 == 0):
                print("here")
                print("recording...")
                while(elapsed < 3): # some time period

                    data_now = pd.concat([pd.DataFrame(P), pd.DataFrame(A.accel1_x), pd.DataFrame(A.accel1_y), pd.DataFrame(A.accel2_x), pd.DataFrame(A.accel2_y)], axis = 0)
                    # print(np.shape((data_now.transpose())))
                    data = pd.concat([data, data_now.transpose()], axis = 0)
                    # print(data_now)
                    # print(elapsed)
                    elapsed = time.time() - start
                    
                print("stopped recording...")   
                # print(np.shape(data))
                # data = pd.DataFrame(data)
                data.to_csv(filename + ".csv")
            else:
                print("released...")
            counter_idx = counter_idx + 1
#####################################################

#####################################################
if __name__ == '__main__':
    #############
    # create labels
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/"
    os.chdir(dir1)
    labels = pd.read_csv("column_labels.csv")
    dir2 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records"
    os.chdir(dir2)
    rospy.init_node('glove', anonymous=True) 
    
    listener1()
    rospy.spin()

    try:
        rec()
    except rospy.ROSInterruptException:
        pass

    

#####################################################





