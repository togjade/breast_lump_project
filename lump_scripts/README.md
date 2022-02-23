lump_scripts

git init . 

git add .

git commit "description"

######################################

### (working) lump_collect.py 
	-> combines the topic and checks the changes in the pressure deriv sum to detect the spikes

### (working) lump_record.py 
	-> records the data when squeezed and then detects the release 

### (in process)  process_trial 
	-> to convert all bag files to csv and pllot the content of specific one or all

### (in process)  lump_write_2_csv.py 
	-> listener to P+A, data to panda dataframe and saves to csv, but autometed naming for csv was not done yet

### (in process)  lump_with_accel_2_csv.py 
	-> listener to P+A, data to panda dataframe and saves to csv, automated naming and change the directory in the script, but the parallel file saving is not done yet. Works for 1 trial. time_k is a variable for the sampling duration

### (working)  lump_with_press_2_csv.py 
	-> listener to P, data to panda dataframe and saves to csv, automated naming and change the directory in the script. Work for many trials. 

### (in process)  lump_accel_2_csv_back.py 
	-> works for 1 trial, 

### (working)  plot..py 
	-> all work fine, change dir inside the code, lump_press_2_csv.py work good for many trials (sampling only P data)"

######################################


if the problem with the accelerometer check the device id with checkAudio.py
# STM-accelerometer-files
Configure STM for ADC module  + accelerometers (please refer to the information listed [here](https://github.com/togjade/yerkebulan-s-adc_accel))




