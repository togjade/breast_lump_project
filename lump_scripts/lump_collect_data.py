#!/usr/bin/env python3
import rospy
import copy
import os
from std_msgs.msg import Int32

from takktile_ros.msg import Touch, Contact
from imu_files.msg import  collect_data2, Accel
i = int
P = Touch()
P = [0.0] * 30
P_ = [0.0] * 30
P_filtered = [0.0] * 30
# T = Contact()
# imu = imu_data()
A = Accel()
message = collect_data2()
start_idx = 0
counter_idx = 1
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
        pub_rec.publish(0)
    return sum_pres > 60
#######################################################
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
#######################################################

# ######################################################
# def subscribe_data_imu(data):
#     global imu
#     imu = data.imu_mat_data
# ######################################################

# ######################################################
def subscribe_data_accel(data):
    global A
    # A.accel1_x = data.accel1_x
    # A.accel1_y = data.accel1_y
    # A.accel2_x = data.accel2_x
    # A.accel2_y = data.accel2_y
    A = data
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
    global start_idx, counter_idx, P_filtered
    counter_idx = 0
    start_idx = 0
    while not rospy.is_shutdown():
        #######################################################
        # time header
        # global P_filtered
        # sum_pres = abs(sum(P_filtered))
        # if sum_pres < 50:   
        #     sum_pres = abs(sum(P_filtered))
        #     pub_rec.publish(0)
        # elif sum_pres > 50:
        #     pub_rec.publish(1)
        message.header.stamp = rospy.Time.now()
        # pressure data
        message.pressure = P
                # # imu data
        # message.imu_data_collect = imu
        #### accelerometer data 
        message.accel1_x = A.accel1_x
        message.accel1_y = A.accel1_y
        message.accel2_x = A.accel2_x
        message.accel2_y = A.accel2_y
        ###################
        # print(start_idx)
        pub.publish(message)

        rate.sleep() 
#######################################################

#######################################################
if __name__ == '__main__':
    str1 = "/home/togzhan/lump_files/data/trial_records"
    os.chdir(str1)
    rospy.init_node('glove', anonymous=True)
    print("start publishing ...")
    pub = rospy.Publisher('lump_data', collect_data2, queue_size = 200) #pressure_sub
    pub_rec = rospy.Publisher('rec_idx', Int32, queue_size = 100) #pressure_sub      
    rate = rospy.Rate(80) #10hz

    listener1()

    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

####################################################### 





