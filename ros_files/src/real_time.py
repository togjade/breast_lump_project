#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from imu_files.msg import collect_data2
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
flag = 0
############
reward1 = 0.5
reward2 = 1
reward3 = 1.5
scoreSoft = 0
scoreRigid = 0
############
data_all = np.array([]).astype("float")
data_P = np.array([]).astype("float")
data_P2 = np.array([]).astype("float")
#######################################################
with torch.no_grad():
    m = "/home/togzhan/sample/data/model/server_model/P_40/biclass_cnn_l1_h_ohe512_bs8_ks8_c4_drop0.3_epoch10000_P40_450"
    #P_20_450/3/biclass_cnn_l1_h_ohe256_bs8_ks8_c4_drop0.1_epoch10000_P20_450"
    #P_40/biclass_cnn_l1_h_ohe512_bs8_ks8_c4_drop0.3_epoch10000_P40_450"
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
    while sum_pres < 60:   
        sum_pres = abs(sum(P_filtered))
        pub2.publish(0) 
    return sum_pres > 60
#######################################################
def subscribe_all(data):
    global P
    global P_
    global P_filtered
    global data_all, data_P
    
    P_ = copy.deepcopy(P[:])
    for i in range(30):
        P[i] = data.pressure[i]
        P_filtered[i] = highPass(P[i], P_[i], P_filtered[i])
    data_P = np.concatenate([P[4:5], P[9:10], P[15:16], P[21:22], P[27:28], P[0:1], P[1:2], P[2:3], P[6:7], P[7:8], P[8:9], P[12:13], P[13:14], P[14:15], P[24:25], P[25:26], P[26:27]], axis = 0)
        #[P[0:22], P[23:30]], axis = 0)
        #
    #([P[0:22], P[23:30]], axis = 0)
    #([P[4:5], P[9:10], P[15:16], P[21:22], P[27:28], P[0:1], P[1:2], P[2:3], P[6:7], P[7:8], P[8:9], P[12:13], P[13:14], P[14:15], P[24:25], P[25:26], P[26:27]], axis = 0)##copy.deepcopy(P[:]) 
    #([P[4:5], P[9:10], P[15:16], P[21:22], P[27:28]], axis = 0)
    #([P[4:5], P[9:10], P[15:16], P[21:22], P[27:28], P[0:1], P[1:2], P[2:3], P[6:7], P[7:8], P[8:9], P[12:13], P[13:14], P[14:15], P[24:25], P[25:26], P[26:27]], axis = 0)##copy.deepcopy(P[:])  
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
    k = int()
    k = 17
    data_P2 = np.array([]).astype("float")
    d = "/home/togzhan/sample/real_time_data/"
    data_p = np.empty((0,k), float)
    m = []
    while(len(data_p) < 450):
          data_p = np.vstack([data_p, data_P])
    data_p = data_p - np.min(data_p)

    for i in range(k):
        data_P2 = np.concatenate([data_P2, data_p[:,i]],axis=0)
    data_P2 = np.divide(data_P2, 450)
    return pd.DataFrame(data_P2)
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
    k = np.argmax(k)
    count = count + 1
    return k
######################################################

def talker():
    global classI, flag
    global ink, count, isClass, scoreRigid, scoreSoft, reward1, reward2, reward3
    # classI = np.argmax(classI)
    while not rospy.is_shutdown():
        classI = 0
        # count = 0
        # print(isClass)
        # a = np.float()
        if(ink == 1):
            time.sleep(1)
            while(count!=3):
                if(isClass & ((P[27] > 120)  | (P[15] > 120))):#| (P[21] > 50) | (P[28] > 50) | (P[14] > 50))):# before was 200
                    # time.sleep(0.5)
                    #print(count)
                    if(count == 0): # 3 finger case 1.5 reward for rigid
                        a = classification()
                        if (a == 0):
                            scoreRigid = scoreRigid + reward3 # 1.5
                        else:
                            scoreSoft = scoreSoft + reward1 # 0.5

                    elif(count == 1): # palm finger case 1.5 reward for soft
                        a = classification()
                        if (a == 0):
                            scoreRigid = scoreRigid + reward1 # 0.5
                        else:
                            scoreSoft = scoreSoft + reward3 # 1.5
                    elif(count == 2):
                        a = classification()
                        if (a == 0):
                            scoreRigid = scoreRigid + reward2 # 1.0
                        else:
                            scoreSoft = scoreSoft + reward2 # 1.0

                    # classI = classification() + classI
                    # print("1",classI)
                    if(count == 3):
                        # print("2 ",classI/3)
                        # classI = np.around(classI/3)
                        if(scoreRigid > scoreSoft):
                            pub.publish(0.0)
                            print("final classififcation: Rigid ", scoreRigid)
                        else:
                            pub.publish(1.0)
                            print("final classififcation: Soft ", scoreSoft)
                        time.sleep(1)
                        if (to_grasp()): 
                            print ("released")
                        if(count == 3):
                            pub2.publish(1)
                            scoreSoft = 0
                            scoreRigid = 0
                            # count = 0

                    # time.sleep(1)
                    # if (to_grasp()): 
                    #     print ("released")
                    #     if(count == 3):
                    #         pub2.publish(1)
                    #         scoreSoft = 0
                    #         scoreRigid = 0
                    #         # count = 0
                    time.sleep(0.5)
                    if(count < 3):
                        isClass = 1
                    time.sleep(2)
            # isClass = 1
            # count = 0        
            while(not isR):
                isClass = 1
                count = 0       
#####################################################################
        else:
            while(count!=1):
                # if not flag:
                #     c = input("New trial?\n")
                # if (c == 'y'):
                #     flag = 1        
                if(isClass & ((P[27] > 100)) ):#| (P[21] > 50) | (P[28] > 50) | (P[14] > 50))):
                    # time.sleep(0.5)
                    pub.publish(classification())
                    # isClass = 0
                    if (to_grasp()): 
                        print ("released")
                        pub2.publish(1)
                            # isClass = 1
                        # isClass = 1
                        flag = 0
            count = 0
            while(not isR):
                isClass = 1
                count = 0
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

