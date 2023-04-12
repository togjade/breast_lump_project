lump_scripts

git init . 

git add .

git commit "description"

######################################

### lump_record.py -> records the data when squeezed and then detects the release. Not good way to do, bacause of delay and reduced sampling rate due to subscription. 

### process_trial -> to convert all bag files to csv and pllot the content of specific one or all

### lump_with_press_2_csv.py -> listener to P, data to panda dataframe and saves to csv, automated naming and change the directory in the script. Work for many trials. 

### lump_with_accel_2_csv -> listener to A, data to panda dataframe and saves to csv, automated naming and change the directory in the script. Work for one trial. 

### plot..py 
	-> all work fine, change dir inside the code, plots for accel, pressure seaprately, data for both. Asks for input number to display the data (counting from the end).

######################################


if the problem with the accelerometer check the device id with checkAudio.py
# STM-accelerometer-files
Configure STM for ADC module  + accelerometers (please refer to the information listed [here](https://github.com/togjade/yerkebulan-s-adc_accel))




