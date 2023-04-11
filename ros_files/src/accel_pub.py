#!/usr/bin/env python3
import pyaudio
import numpy as np
import rospy
import time
from imu_files.msg import Accel
 
accel_data = Accel()
# publisher

def talker():
    FORMAT = pyaudio.paInt16
    CHANNELS = 4
    RATE = 8000
    CHUNK = 50
    audio = pyaudio.PyAudio()
        
    stream = audio.open(format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=CHUNK,input_device_index=10)
    
    sample = np.zeros([CHANNELS, CHUNK])
    
    print("recording...")
    # data = stream.read(CHUNK)
    
    while not rospy.is_shutdown():
        data = stream.read(CHUNK)
        # read data from stream
        for i in range (CHUNK):
            for j in range(CHANNELS):
                sample[j,i]=int.from_bytes([data[j*2+i*8],data[j*2+i*8+1]], "little", signed=True)
                
        sample = sample/32768    
        
        accel_data.accel1_x = sample[0,:].transpose()
        accel_data.accel1_y = sample[1,:].transpose()
        accel_data.accel2_x = sample[2,:].transpose()
        accel_data.accel2_y = sample[3,:].transpose()
        #accel_data.accel3_x = sample[4,:].transpose()
        #accel_data.accel3_y = sample[5,:].transpose()

        pub.publish(accel_data)
        
        sample = np.zeros([CHANNELS, CHUNK])
        # accel_data.data[:] = []
        rate.sleep()
        
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == '__main__':
    
    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('accel_data', Accel, queue_size = 100)
    rate = rospy.Rate(8000) #10hz

    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
    
    
