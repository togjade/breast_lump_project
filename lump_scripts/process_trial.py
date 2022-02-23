#!/usr/bin/env python3
from itertools import count
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

def bag_2_csv(bag_files, mode1):
    l_stop = len(bag_files)

    if (mode1 == 0):
        l_start = 0
    else:
        l_start = len(bag_files) - mode1

    for i in range(l_start, l_stop, 1):
        pathname, extension = os.path.splitext(bag_files[i])
        str1 = "rostopic echo /lump_data -b "
        str3 = " -p > "
        str4 = ".csv"
        str_bag_2_csv = str1 + pathname + extension + str3 + pathname + str4
        # print(str_bag_2_csv)
        os.system(str_bag_2_csv)
    print("Files converted sucessfully!")

def read_csv(csv_name):
    data = pd.read_csv(csv_name)
    data = data.values[:]
    pressure_data, accel_x1, accel_y1,accel_x2, accel_y2 = preprocess_matrix(data)
    # plot the pressure
    plot_pressure(pressure_data)
    plot_accel(accel_x1, accel_y1, accel_x2, accel_y2)

def preprocess_matrix(data):
    pressure_data = data[:, 4:34]
    accel_d = data[:, 34:234]
    accel_x1 = []
    accel_y1 = []
    accel_x2 = []
    accel_y2 = []
    print((np.size(accel_d, 1)))
    for i in range(np.size(accel_d, 0)):
        accel_x1 = np.concatenate([accel_x1, accel_d[i, 0:50]])
        accel_y1 = np.concatenate([accel_y1, accel_d[i, 50:100]])
        accel_x2 = np.concatenate([accel_x2, accel_d[i, 100:150]])
        accel_y2 = np.concatenate([accel_y2, accel_d[i, 150:200]])
        
    return pressure_data, accel_x1, accel_y1,accel_x2, accel_y2

def plot_pressure(pressure_data):
    counter = 0
    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True)

    for i in range(5):
        for j in range(6):
            plot_counter = str(counter)
            axs[i, j].plot(pressure_data[:, counter])
            axs[i, j].title.set_text('Sensor: ' + plot_counter)
            counter = counter + 1
    plt.show()

def plot_accel(accel_x1, accel_y1, accel_x2, accel_y2):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs[0, 0].plot(accel_x1)
    axs[0, 0].title.set_text('Sensor: ' + 'ax1')

    axs[0, 1].plot(accel_y1)
    axs[0, 1].title.set_text('Sensor: ' + 'ay1')

    axs[1, 0].plot(accel_x2)
    axs[1, 0].title.set_text('Sensor: ' + 'ax2')

    axs[1, 1].plot(accel_y2)
    axs[1, 1].title.set_text('Sensor: ' + 'ay2')

    plt.draw()
    plt.show()

if __name__ == '__main__':
    dir1 = "/home/togzhan/lump_files/data/trial_records"
    os.chdir(dir1)
    bag_files = glob.glob("*.bag")

    while True:
        try:
            mode1 = int(input("Enter 0 - all bags and # - specific bag from end: "))
            if( (mode1 < 0) | (mode1 > len(bag_files)) ):
                raise ValueError()
        except ValueError:
            print("Enter again")
            continue
        else:
            break
    bag_2_csv(bag_files, mode1)

    csv_name = glob.glob("*.csv")
    while True:
        try:
            mode2 = int(input("Enter # of csv file to display from end: "))
            if( (mode2 < 0) | (mode2 == 0) | (mode2 > len(csv_name)) ):
                raise ValueError()
        except ValueError:
            print("Enter again")
            continue
        else:
            break
    read_csv(csv_name[len(csv_name) - mode2])

