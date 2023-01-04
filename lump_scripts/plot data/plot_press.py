#!/usr/bin/env python3
from itertools import count
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
sensor_names = ["row1_1", "row1_2", "row1_3", "pinky1", "pinky2", "pinky3", "row2_1", "row2_2", "row2_3", "ring1", "ring2", "ring3", "row3_1", "row3_2", "row3_3", "middle1", "middle2", "middle3",
"row4_3", "row4_2", "row4_1", "index1", "index2", "index3", "row5_1", "row5_2", "row5_3", "thumb1", "thumb2", "thumb3"]
def read_csv(csv_name):
    global pressure_data, accel_x1, accel_y1, accel_x2, accel_y2, data
    data = []
    data = pd.read_csv(csv_name)
    data = data.values[:]
    pressure_data = preprocess_matrix(data)
    print(np.shape(pressure_data))
    plot_pressure(pressure_data)

def preprocess_matrix(data):
    pressure_data = data[:, 1:31]
    return pressure_data

def plot_pressure(pressure_data):
    counter = 0
    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True)

    for i in range(5):
        for j in range(6):
            # plot_counter = str(counter)
            axs[i, j].plot(pressure_data[:, counter])
            axs[i, j].title.set_text('Sensor: ' + sensor_names[counter])
            counter = counter + 1
    plt.draw()
    # plt.show()

def read_csv_all(csv_name):
    for i in range(len(csv_name)):
        csv_namE = csv_name[i]
        read_csv(csv_namE)
    plt.show()

if __name__ == '__main__':
    # dir1 = "/home/togzhan/lump_files/data/trial_records"
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records/Karina_trial_1/without"

    os.chdir(dir1)
    csv_name = glob.glob("*.csv")
    while True:
        try:
            mode2 = int(input("Enter # of csv file to display from end: "))
            if( (mode2 < 0)  | (mode2 > len(csv_name)) ): #| (mode2 == 0)
                raise ValueError()
        except ValueError:
            print("Enter again")
            continue
        else:
            break
    # print(csv_name[len(csv_name) - mode2])
    if (mode2 == 0):
        read_csv_all(csv_name)
    else:
        read_csv(csv_name[len(csv_name) - mode2])

