import numpy as np
import scipy
import glob as glob
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd

# Define the folder path
folder_path = 'doctor1'

# Define the relevant columns
relevant_columns = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29]
samples_per_trial = 1120  # Total samples per trial

# Initialize a list to store the final data
final_data = []

# Iterate through each trial folder
for trial in range(13):  # From 0 to 12
    trial_folder = os.path.join(folder_path, str(trial))
    trial_files = os.listdir(trial_folder)
    
    # Determine the label and valid trial numbers based on the trial folder
    if trial in [9, 10, 11, 12]:
        label = 0
        max_trial_no = 35
    else:
        label = 1
        max_trial_no = 15
    
    for file in trial_files:
        file_path = os.path.join(trial_folder, file)
        data = pd.read_csv(file_path, header=0)
        
        # Select only the first 1120 rows and the first 30 columns, then filter the relevant ones
        first_1120_rows = data.iloc[:samples_per_trial, :30]
        filtered_data = first_1120_rows.iloc[:, relevant_columns]
        
        # Convert the filtered data to a flattened list
        data_list = filtered_data.values.flatten("F").tolist()
        
        # Ensure the data_list length matches the expected data size
        if len(data_list) == 16800:
            trial_no = len(final_data) % (max_trial_no + 1)
            final_data.append([0, trial, trial_no, data_list, label])

# Create a DataFrame from the final data
final_df = pd.DataFrame(final_data, columns=['Person No', 'Type', 'Trial No', 'Data', 'Label'])

final_df.to_pickle('doctors_data_labeled.pkl')