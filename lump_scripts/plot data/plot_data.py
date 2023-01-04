#!/usr/bin/env python3
from itertools import count
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

def read_csv(csv_name):
    
    global pressure_data, accel_x1, accel_y1, accel_x2, accel_y2, data
    data = []
    data = pd.read_csv(csv_name)
    # print(data)
    data = data.values[:]
    pressure_data, accel_x1, accel_y1,accel_x2, accel_y2 = preprocess_matrix(data)
    # plot the pressure
    # print(pressure_data)
    print(np.shape(pressure_data))
    plot_pressure(pressure_data)
    plot_accel(accel_x1, accel_y1, accel_x2, accel_y2)

def preprocess_matrix(data):
    pressure_data = data[:, 1:31]
    # print(np.shape(pressure_data))
    accel_d = data[:, 31:231]
    # print(pressure_data)
    
    accel_x1 = []
    accel_y1 = []
    accel_x2 = []
    accel_y2 = []
    # print((np.size(accel_d, 1)))
    for i in range(np.size(accel_d, 0)):
        accel_x1 = np.concatenate([accel_x1, accel_d[i, 0:50]])
        accel_y1 = np.concatenate([accel_y1, accel_d[i, 51:100]])
        accel_x2 = np.concatenate([accel_x2, accel_d[i, 101:150]])
        accel_y2 = np.concatenate([accel_y2, accel_d[i, 151:201]])

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
    plt.draw()
    # plt.show()

def plot_accel(accel_x1, accel_y1, accel_x2, accel_y2):
    line = 0.5
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, linewidth = 0.5)
    axs[0, 0].plot(accel_x1, linewidth = line)
    axs[0, 0].title.set_text('Sensor: ' + 'ax1')

    axs[0, 1].plot(accel_y1, linewidth = line)
    axs[0, 1].title.set_text('Sensor: ' + 'ay1')

    axs[1, 0].plot(accel_x2, linewidth = line)
    axs[1, 0].title.set_text('Sensor: ' + 'ax2')

    axs[1, 1].plot(accel_y2, linewidth = line)
    axs[1, 1].title.set_text('Sensor: ' + 'ay2')

    # plt.draw()
    plt.show()
if __name__ == '__main__':
    dir1 = "/home/togzhan/lump_files/data/trial_records"
    dir1 = "/media/togzhan/windows 2/documents/lump_files/data/trial_records"
    os.chdir(dir1)

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
    print(csv_name[len(csv_name) - mode2])
    read_csv(csv_name[len(csv_name) - mode2])

