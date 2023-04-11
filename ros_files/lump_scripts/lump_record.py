#!/usr/bin/env python3
from multiprocessing.connection import wait
import os 
import time
import matplotlib
import pandas as pd
import numpy as np
import glob
from std_msgs.msg import Int32
import rospy
import copy
counter_idx = 1
counter_idx2 = 0
old_idx = 0
start_idx = 0
idx = 0
#######################################################
def subscribe_data_idx(data):
    global idx, old_idx
    old_idx = copy.deepcopy(idx)
    idx = data.data
#######################################################

#######################################################
def listener1():
    rospy.Subscriber("rec_idx", Int32, subscribe_data_idx)
#######################################################

def rec():
    global start_idx, counter_idx, counter_idx2, old_idx
    if((idx == 1) & (old_idx == 0)):
        if((counter_idx)%2 == 1):
            print("recording...")
            os.system("rosbag record /lump_data --duration=5")
            # time.sleep(5)
            print("stopped recording...") 
        else:
            print("released...")
        counter_idx = counter_idx + 1
    return counter_idx

def talker():

    while not rospy.is_shutdown():
        rec()
        rate.sleep() 
        
if __name__ == '__main__':
    
    str1 = "/media/togzhan/windows 2/documents/lump_files/data/"
    os.chdir(str1)
    rospy.init_node('glove', anonymous=True)
    rate = rospy.Rate(100) #10hz
    listener1()
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

