lump_scripts

git init . 

git add .

git commit "description"

if the problem with the accelerometer check the device id with checkAudio.py
# STM-accelerometer-files
Configure STM for ADC module  + accelerometers


Compass files are for the configuring the STM32f3discovery to read the Accelerometer data as a audio input. 
  1) to upload the code to the STM32:
		* install CoIDE (check the server -> Software -> CoIDE installers -> CoIDE-1.7.8.exe)
		* install gcc-arm (in the same directory as previous)
		* Download the compass folder (in the same directory) OR [here](https://github.com/togjade/yerkebulan-s-adc_accel.git)
		* Double-click on compass_new with .coproj extension
		* Choose the main.h file from the sidebar and check the necessary # of channels by changing teh value of SPI1_datasize variable
		* Connect board (USB ST-Link) to computer
		* Click on the icon (download code to Flash)
		* When Finished connect to USB User and feel free to use
  2) For the ADC connection check the datasheet attached [here](https://github.com/togjade/yerkebulan-s-adc_accel/blob/main/ADC%20documentation.docx)
  3) Look [here](https://github.com/togjade/yerkebulan-s-adc_accel/blob/main/accel_pub.py) for the sampling data with ROS.
  
checkaudio.py file to check if the stm is connected. 
  * check the device ID with the checkAudio.py script -> run python checkAudio.py
  * Check if the ID are the same as in the script where you access the stream (input_device_index) -> 
		stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK,input_device_index=10)

recordAudio.py read the one loop of the data. 



