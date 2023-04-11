#!/usr/bin/env python3
import pyaudio
import numpy as np
import rospy
import time
from std_msgs.msg import Float32MultiArray

from takktile_ros.msg import Touch, Contact
from imu_files.msg import  collect_data2, Accel
i = int
P = Touch()
# T = Contact()
# imu = imu_data()
A = Accel()
message = collect_data2()

#######################################################
def subscribe_data_array(data):
    global P
    P = data.pressure
#######################################################

# #######################################################
# def subscribe_touch_array(data):

#     global T
#     T = data.pressure
# #######################################################

# ######################################################
# def subscribe_data_imu(data):
#     global imu
#     imu = data.imu_mat_data
# ######################################################

# ######################################################
def subscribe_data_accel(data):
    global A
    A.accel1_x = data.accel1_x
    A.accel1_y = data.accel1_y
    A.accel2_x = data.accel2_x
    A.accel2_y = data.accel2_y
# ######################################################

#######################################################
def listener1():

    rospy.Subscriber("takktile/calibrated", Touch, subscribe_data_array)
    # rospy.Subscriber("takktile/contact", Contact, subscribe_touch_array)
    # rospy.Subscriber("imu__data", imu_data, subscribe_data_imu)
    rospy.Subscriber("accel_data", Accel, subscribe_data_accel)
#######################################################

#######################################################
def talker():
    while not rospy.is_shutdown():
        #######################################################
        # time header
        message.header.stamp = rospy.Time.now()

        # pressure data
        message.pressure = P
        # pressure touch
        # message.contact = T
        # # imu data
        # message.imu_data_collect = imu
        # # accelerometer data 
        message.accel1_x = A.accel1_x
        message.accel1_y = A.accel1_y
        message.accel2_x = A.accel1_x
        message.accel2_y = A.accel2_y

        # #######################################################
        pub.publish(message)
        
        rate.sleep() 
#######################################################

#######################################################
if __name__ == '__main__':

    rospy.init_node('glove', anonymous=True)

    pub = rospy.Publisher('pressure_sub', collect_data2, queue_size =100) #pressure_sub

    rate = rospy.Rate(60) #10hz

    listener1()

    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

####################################################### 





