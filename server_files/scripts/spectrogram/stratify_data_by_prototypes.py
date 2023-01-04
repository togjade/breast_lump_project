from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import lumpDataset, ToTensor
from utils import *
from multiprocessing import Pool
from sklearn.utils import shuffle

def take_one_prototype_data(j, var):
    ############################################
# WORKS FINE
    directory = '/raid/togzhan_syrymova/lump_project/data/'
    os.chdir(directory)
    all_data_with = pd.read_csv('all_subject_pressure_data_with_raw.csv', sep = ',', header = None)
    all_data_with_1st = pd.DataFrame()
    all_data_with_1st = pd.DataFrame()
    k = 9
    for i in range(8): # of first 
        v1 = (i*k)*32+j*32
        v2 = (i*k+1)*32+j*32
        all_data_with_1st = pd.concat([all_data_with_1st, pd.DataFrame(all_data_with.values[:, v1:v2])], axis = 1)

    all_data_with_1st = all_data_with_1st.transpose()
    all_data_with_1st = all_data_with_1st.reset_index(drop = True)
    # all_data_with_1st.columns = [''] * len(all_data_with_1st.columns)
    print(np.shape(all_data_with_1st))
    ############################################
    # take first 8 people data
    l = 8*72*4
    all_data_without = pd.read_csv('all_subject_pressure_data_without_raw.csv', sep = ',', header = None)  
    all_data_without_1st = pd.DataFrame()
    all_data_without_1st = pd.DataFrame(all_data_without.values[:, 0:l])
    all_data_without_1st = all_data_without_1st.transpose()
    all_data_without_1st = shuffle(all_data_without_1st)
    all_data_without_1st = all_data_without_1st.reset_index(drop = True)
    all_data_without_1st.columns = [''] * len(all_data_without_1st.columns)
    all_data_without_1st = pd.DataFrame(all_data_without_1st.values[0:np.shape(all_data_with_1st)[0], :])
    ##############################################
    test_with = pd.DataFrame()
    k = 9
    for i in range(8, 10): # of first 
        v1 = (i*k)*32+j*32
        v2 = (i*k+1)*32+j*32
        test_with = pd.concat([test_with, pd.DataFrame(all_data_with.values[:, v1:v2])], axis = 1)
    test_with = test_with.reset_index(drop = True)
    test_with.columns = [''] * len(test_with.columns)
    print(np.shape(test_with))
    i = 3
    ##############################################
    test_without = pd.DataFrame(all_data_without.values[:, (10*4*72-np.shape(test_with)[1]):])
    test_without = test_without.reset_index(drop = True)
    test_without.columns = [''] * len(test_without.columns)
    print(np.shape(test_without))
    ##############################################
    label_with = []
    label_without = []
    for i in range(np.shape(all_data_with_1st)[0]):
        label_with.append(1)
        label_without.append(0)

    bin_labels = np.concatenate([label_without, label_with], axis = 0)
    bin_labels = pd.DataFrame(bin_labels)
    print(np.shape(bin_labels))
    ##############################################
    df_data = pd.DataFrame()
    df_data = pd.concat([all_data_without_1st, all_data_with_1st], axis = 0)
    df_data = df_data.reset_index(drop = True)
    df_data.columns = [''] * len(df_data.columns)
    df_data = pd.concat([df_data, bin_labels], axis = 1)
    df_train, df_eval = train_test_split(df_data, test_size = 1.0/3, random_state=777, stratify = bin_labels)
    ##############################################
    df_train.to_csv(directory + "df_train" + var + ".csv", index=False) # , header = False
    df_eval.to_csv(directory + "df_dev" + var + ".csv", index=False)
    test_with.to_csv(directory + "test_pressure_data_with" + var + ".csv", index=False, header = False)
    test_without.to_csv(directory + "test_pressure_data_without" + var + ".csv", index=False, header = False)

    
if __name__ == '__main__':
    var = input("Enter the abbreviation(ex. _L_h2): ")
#     var = '_L_h2'
    j = input("Enter the index(ex. 8): ")
    j = int(j)
#     j = 8
    take_one_prototype_data(j, var)