#!/usr/bin/env python3
from itertools import count
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

def read_csv(csv_name):
    
    global  accel_x1, accel_y1, accel_x2, accel_y2, data
    data = []
    data = pd.read_csv(csv_name)
    # print(data)
    data = data.values[:]
    print(np.shape(data))
    accel_x1, accel_y1, accel_x2, accel_y2 = preprocess_matrix(data)
    # print(np.shape(accel_x1))
    plot_accel(accel_x1, accel_y1, accel_x2, accel_y2)
    # plot_accel_sep(accel_x1)
    # plt.show()
    
def preprocess_matrix(data):
    # print(np.shape(pressure_data))
    accel_d = data[1:441, 1:201]
    # print(pressure_data)
    accel_x1 = []
    accel_y1 = []
    accel_x2 = []
    accel_y2 = []
    print(np.shape(accel_d))
    # np.size(accel_d, 0)
    for i in range(np.size(accel_d, 0)):

        accel_x1 = np.concatenate([accel_x1, accel_d[i, 0:50]])

        accel_y1 = np.concatenate([accel_y1, accel_d[i, 51:100]])

        accel_x2 = np.concatenate([accel_x2, accel_d[i, 101:150]])

        accel_y2 = np.concatenate([accel_y2, accel_d[i, 151:201]])

    return accel_x1, accel_y1,accel_x2, accel_y2

def plot_accel(accel_x1, accel_y1, accel_x2, accel_y2):
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs[0, 0].plot((accel_x1))
    axs[0, 0].title.set_text('Sensor: ' + 'ax1')

    axs[0, 1].plot(accel_y1)
    axs[0, 1].title.set_text('Sensor: ' + 'ay1')

    axs[1, 0].plot(accel_x2)
    axs[1, 0].title.set_text('Sensor: ' + 'ax2')

    axs[1, 1].plot(accel_y2)
    axs[1, 1].title.set_text('Sensor: ' + 'ay2')

    # plt.draw()
    plt.show()

# def plot_accel_sep(accel):
    
#     # fig = plt(sharex=True, sharey=True)
#     plt.plot((accel))
   
#     plt.draw()
#     # plt.show()
    
if __name__ == '__main__':
    # dir1 = "/home/togzhan/lump_files/data/trial_records"
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

