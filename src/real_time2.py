#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from imu_files.msg import collect_data2, Accel
from model_arch import  *
from takktile_ros.msg import Touch
import rospy
from std_msgs.msg import Float32
import time
import glob
import rospy
import torch
import torch.nn.functional as F
import copy

#######################################################
A = Accel()
P = [0.0] * 30
P_ = [0.0] * 30
P_filtered = [0.0] * 30
############
isR = Float32()
isR = 0
############
isClass = Float32()
isClass = 1
############
count = Float32()
count = 0
############
classI = np.int64()
############
data_all = np.array([]).astype("float")
data_P = np.array([]).astype("float")
# data_A = np.array([]).astype("float")
# data_all = np.array([]).astype("float")
# data_A1x = np.array([]).astype("float")
# data_A1y = np.array([]).astype("float")
# data_A2x = np.array([]).astype("float")
# data_A2y = np.array([]).astype("float")
data_P2 = np.array([]).astype("float")
#######################################################
flag = 0
with torch.no_grad():
    m = "/home/togzhan/sample/data/model/server_model/biclass_fnn_l1_h512_bs16_c4_drop0.3_epoch10000_P_A"
    #P_40/biclass_cnn_l1_h_ohe512_bs8_ks16_c4_drop0.3_epoch10000_P40_450"
    #P_20_450/3/biclass_cnn_l1_h_ohe256_bs8_ks3_c4_drop0.4_epoch10000_P20_450"
    #P_40/biclass_cnn_l1_h_ohe512_bs8_ks8_c4_drop0.3_epoch10000_P40_450"
    #P_40/biclass_cnn_l1_h_ohe512_bs8_ks8_c4_drop0.3_epoch10000_P40_450"

    model = torch.load(m, map_location = torch.device('cpu'))
    device = torch.device("cpu")
#######################################################
def highPass(z, z_, out):
    T = 0.2
    dt = 0.02
    res = (out + z - z_)*T/(T + dt)
    return res
def to_grasp():
    global P_filtered
    sum_pres = abs(sum(P_filtered))
    while sum_pres < 90:   
        sum_spres = abs(sum(P_filtered))
        pub2.publish(0) 
    return sum_pres > 90
#######################################################
def subscribe_all(data):
    global P, P_, P_filtered
    global A, data_P
    A.accel1_x = data.accel1_x
    A.accel1_y = data.accel1_y
    A.accel2_x = data.accel2_x
    A.accel2_y = data.accel2_y
    P_ = copy.deepcopy(P[:])
    for i in range(30):
        P[i] = data.pressure[i]
        P_filtered[i] = highPass(P[i], P_[i], P_filtered[i])
    data_P = np.concatenate([P[0:18], P[21:30]], axis=0)
    # data_A = np.concatenate(A.accel1_x, A.accel1_y, A.accel2_x, A.accel2_y)
    
        #[P[4:5], P[9:10], P[15:16], P[21:22], P[27:28], P[0:1], P[1:2], P[2:3], P[6:7], P[7:8], P[8:9], P[12:13], P[13:14], P[14:15], P[24:25], P[25:26], P[26:27]], axis = 0)
    #([P[0:22], P[23:30]], axis = 0)
    #([P[4:5], P[9:10], P[15:16], P[21:22], P[27:28]], axis = 0)
#######################################################
def listener():
    rospy.Subscriber("/isReady", Float32, subscribe_r)
    rospy.Subscriber("pressure_sub", collect_data2, subscribe_all)
#######################################################
def subscribe_r(data):
    global isR
    isR = data.data
######################################################
def real_time():
    # global data_A1x, data_A1y, data_A2x, data_A2y, data_A
    data_A = np.array([]).astype("float")
    data_all = np.array([]).astype("float")
    data_A1x = np.array([]).astype("float")
    data_A1y = np.array([]).astype("float")
    data_A2x = np.array([]).astype("float")
    data_A2y = np.array([]).astype("float")
    k = int()
    k = 27
    data_P2 = np.array([]).astype("float")
    d = "/home/togzhan/sample/real_time_data/"
    data_p = np.empty((0,k), float)
    # data_a = np.empty((0,200), float)
    m = []
    while(len(data_p) < 744):
        data_p = np.vstack([data_p, data_P])

        data_A1x = np.concatenate([data_A1x, A.accel1_x], axis = 0)
        data_A1y = np.concatenate([data_A1y, A.accel1_x], axis = 0)
        data_A2x = np.concatenate([data_A2x, A.accel1_x], axis = 0)
        data_A2y = np.concatenate([data_A2y, A.accel1_x], axis = 0)
    data_A = np.concatenate([data_A1x, data_A1y, data_A2x ,data_A2y], axis = 0)
    data_p = data_p - np.min(data_p)
    # print(data_p)
    for i in range(k):
        data_P2 = np.concatenate([data_P2, data_p[:,i]],axis=0)
    
    data_P2 = np.divide(data_P2, 450)
    data_all = np.concatenate([data_P2, data_A], axis = 0)
    return pd.DataFrame(data_all)
######################################################
def classification():
    global isClass
    global count
    k = np.int64()
    print ("---------------------------------\n", P[27])# , P[21], P[28], P[14])
    print("grasp")
    # time.sleep(2)
    clas = pd.DataFrame()
    clas = real_time()
    clas = clas.transpose()
    t = torch.FloatTensor(clas.values[:,:].astype('float64'))
    model.eval().to(device)
    output = model(t)
    print(output)
    output = (output>0.5).float()
    print(output)
    isClass = 0
    
    k = output.numpy()
    # k = np.argmax(k)
    count = count + 1

    return k

######################################################

def talker():
    global classI
    global ink, count, isClass
    global flag
    # classI = np.argmax(classI)
    while not rospy.is_shutdown():
        classI = 0
        # count = 0
        # print(isClass)
        if(ink == 1):
            time.sleep(1)
            while(count!=3):
                
                if(isClass & ((P[27] > 100) )):#| (P[21] > 50) | (P[28] > 50) | (P[14] > 50))):
                    # time.sleep(0.5)
                    #print(count)
                    # a = classification()
                    classI = classification() + classI
                    # print("1",classI)
                    if(count == 3):
                        print("2 ",classI/3)
                        classI = np.around(classI/3)
                        
                        pub.publish(classI)
                        print("final classififcation: ")
                        print(classI)
                    time.sleep(1)
                    if (to_grasp()): 
                        print ("released")
                        if(count == 3):
                            pub2.publish(1)
                            # count = 0

                    if(count < 3):
                        isClass = 1
                    time.sleep(2)
            isClass = 1
            count = 0        
            # while(not isR):
            #     isClass = 1
            #     count = 0       
#####################################################################
        else:
            while(count!=1):
                if not flag:
                    c = input("New trial?\n")
                if (c == 'y'):
                    flag = 1        
                if(isClass & ((P[27] > 100)) & flag):#| (P[21] > 50) | (P[28] > 50) | (P[14] > 50))):
                    # time.sleep(0.5)
                    pub.publish(classification())
                    # isClass = 0
                    if (to_grasp()): 
                        print ("released")
                        pub2.publish(1)
                            # isClass = 1
                        isClass = 1
                        flag = 0
            count = 0
            # while(not isR):
            #     isClass = 1
            #     count = 0
#####################################################################            
        rate.sleep()  

if __name__ == '__main__':
    print("enter 1 for the 3 trial case, any number otherwise")
    ink = int(input())
    d = "/home/togzhan/sample/real_time_data/"
    
    rospy.init_node('glove', anonymous=True)
    pub = rospy.Publisher('class', Float32, queue_size =100) #pressure_sub
    pub2 = rospy.Publisher('release', Float32, queue_size =100) #pressure_sub
    rate = rospy.Rate(10) # 10hz
    listener()
    try:
        talker()
    except rospy.ROSInterruptException:
        rospy.spin()

