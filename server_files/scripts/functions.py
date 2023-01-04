import pandas as pd
import os
import numpy as np

def compute_spectogram(df):
    df_1 = np.zeros([np.shape(df)[0], 129, 4])
    for i in range(15):
        fs = 160
        f, t, data = spectrogram(df.values[:, 1120*i:(i+1)*1120], fs, axis = 1)
        df_1 = np.concatenate([df_1, 10*np.log10(data)], axis = 1)
        data = []
    return df_1[:, 129:, :],t, f

def get_data(idx, df_train1, df_dev1, df_test1):
#     import pandas as pd
    df_train_mean = pd.DataFrame()
    df_train_stdev = pd.DataFrame()

    df_test_mean = pd.DataFrame()
    df_test_stdev = pd.DataFrame()

    df_dev_mean = pd.DataFrame()
    df_dev_stdev = pd.DataFrame()
    if (idx == 0):
        for i in range(15):
            df_train_mean = pd.concat([df_train_mean, pd.DataFrame(np.mean(df_train1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)
            df_train_stdev = pd.concat([df_train_stdev, pd.DataFrame(np.std(df_train1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)

            df_dev_mean = pd.concat([df_dev_mean, pd.DataFrame(np.mean(df_dev1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)
            df_dev_stdev = pd.concat([df_dev_stdev, pd.DataFrame(np.std(df_dev1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)

            df_test_mean = pd.concat([df_test_mean, pd.DataFrame(np.mean(df_test1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)
            df_test_stdev = pd.concat([df_test_stdev, pd.DataFrame(np.std(df_test1.values[:, i*1120:(i+1)*1120], axis = 1))], axis = 1)

        df_train = pd.concat([df_train_mean, df_train_stdev], axis = 1)
        df_dev = pd.concat([df_dev_mean, df_dev_stdev], axis = 1)
        df_test = pd.concat([df_test_mean, df_test_stdev], axis = 1)
        return df_train, df_dev, df_test

    elif (idx == 1):
        train_y2 = pd.DataFrame()
        dev_y2 = pd.DataFrame()
        test_y2 = pd.DataFrame()

        for j in range(7):
            df_train_mean_s = pd.DataFrame()
            df_train_stdev_s = pd.DataFrame()

            df_test_mean_s = pd.DataFrame()
            df_test_stdev_s = pd.DataFrame()

            df_dev_mean_s = pd.DataFrame()
            df_dev_stdev_s = pd.DataFrame()

            for i in range(15):
                index1 = i*1120+j*160
                index2 = i*1120+(j+1)*160
                df_train_mean_s = pd.concat([df_train_mean_s, pd.DataFrame(np.mean(df_train1.values[:, index1:index2], axis = 1))], axis = 1)
                df_train_stdev_s = pd.concat([df_train_stdev_s, pd.DataFrame(np.std(df_train1.values[:, index1:index2], axis = 1))], axis = 1)

                df_dev_mean_s = pd.concat([df_dev_mean_s, pd.DataFrame(np.mean(df_dev1.values[:, index1:index2], axis = 1))], axis = 1)
                df_dev_stdev_s = pd.concat([df_dev_stdev_s, pd.DataFrame(np.std(df_dev1.values[:, index1:index2], axis = 1))], axis = 1)

                df_test_mean_s = pd.concat([df_test_mean_s, pd.DataFrame(np.mean(df_test1.values[:, index1:index2], axis = 1))], axis = 1)
                df_test_stdev_s = pd.concat([df_test_stdev_s, pd.DataFrame(np.std(df_test1.values[:, index1:index2], axis = 1))], axis = 1)

            df_train_mean = pd.concat([df_train_mean, df_train_mean_s], axis = 0)
            df_train_stdev = pd.concat([df_train_stdev, df_train_stdev_s], axis = 0)

            df_test_mean = pd.concat([df_test_mean, df_test_mean_s], axis = 0)
            df_test_stdev = pd.concat([df_test_stdev, df_test_stdev_s], axis = 0)

            df_dev_mean = pd.concat([df_dev_mean, df_dev_mean_s], axis = 0)
            df_dev_stdev = pd.concat([df_dev_stdev, df_dev_stdev_s], axis = 0)

            # concat labels 7 times corresponding to 7 second
            train_y2 = pd.concat([train_y2, train_y11], axis = 0)
            dev_y2 = pd.concat([dev_y2, dev_y11], axis = 0)
            test_y2 = pd.concat([test_y2, test_y11], axis = 0)


        df_train = pd.concat([df_train_mean, df_train_stdev], axis = 1)
        df_dev = pd.concat([df_dev_mean, df_dev_stdev], axis = 1)
        df_test = pd.concat([df_test_mean, df_test_stdev], axis = 1)

    elif(idx == 2):
        df_train = df_train1
        df_dev = df_dev1
        df_test = df_test1
        return df_train, df_dev, df_test

    elif(idx == 3): #COMPUTE z-score
        df_train_z = pd.DataFrame()
        df_test_z = pd.DataFrame()
        df_dev_z = pd.DataFrame()
        for i in range(15):
            df_train_z1 = pd.DataFrame()
            df_dev_z1 = pd.DataFrame()
            df_test_z1 = pd.DataFrame()

            index1 = i*1120
            index2 = (i+1)*1120

            d_train = df_train1.values[:, i*1120:(i+1)*1120]
            d_dev = df_dev1.values[:, i*1120:(i+1)*1120]
            d_test = df_test1.values[:, i*1120:(i+1)*1120] 

            df_train_z1 = pd.DataFrame(stats.zscore(d_train, axis = 1))
            df_dev_z1 = pd.DataFrame(stats.zscore(d_dev, axis = 1))
            df_test_z1 = pd.DataFrame(stats.zscore(d_test, axis = 1))

            df_train_z1.columns = [''] * len(df_train_z1.columns)
            df_dev_z1.columns = [''] * len(df_dev_z1.columns)
            df_test_z1.columns = [''] * len(df_test_z1.columns)

        #     df_train_z1 = df_train_z1.reset_indext_index()
            df_train_z = pd.concat([df_train_z, df_train_z1], axis = 1)
            df_dev_z = pd.concat([df_dev_z, df_dev_z1], axis = 1)
            df_test_z = pd.concat([df_test_z, df_test_z1], axis = 1)

        return df_train_z, df_dev_z, df_test_z

    elif(idx == 4): #COMPUTE z-score
        df_train_z = pd.DataFrame()
        df_test_z = pd.DataFrame()
        df_dev_z = pd.DataFrame()
        df_train_z = pd.DataFrame(stats.zscore(df_train1, axis = 1))
        df_dev_z = pd.DataFrame(stats.zscore(df_dev1, axis = 1))
        df_test_z = pd.DataFrame(stats.zscore(df_test1, axis = 1))

        return df_train_z, df_dev_z, df_test_z