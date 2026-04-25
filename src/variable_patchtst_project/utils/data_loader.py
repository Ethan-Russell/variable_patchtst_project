import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_electricity_data():
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        IN_COLAB = True
        path_prefix = '/content/drive/MyDrive/CS6787-time-series-forecast'
    except ImportError:
        IN_COLAB = False
        path_prefix = '../../../data'
    
    

    # Define a target directory in your Google Drive, within the shared folder
    data_path = f'{path_prefix}/data'

    # Create the data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Set the destination path
    destination_path = os.path.join(data_path, "electricity-load-forecasting")

    # Download Dataset from KaggleHub and transfer to Google Drive if needed
    if not os.path.exists(destination_path):
        import kagglehub

        # Download latest version
        path = kagglehub.dataset_download("saurabhshahane/electricity-load-forecasting")

        # If the destination already exists, remove it before copying to ensure a clean copy
        if os.path.exists(destination_path):
            shutil.rmtree(destination_path)

        # Copy the entire directory
        shutil.copytree(path, destination_path)

        print(f"Dataset copied from {path} to {destination_path}")

    # Read the dataset from file
    dataset_path = os.path.join(destination_path, 'continuous dataset.csv')
    
    return dataset_path

def read_electricity_data(dataset_path=None):
    
    if dataset_path is None:
        dataset_path = load_electricity_data()
    
    # Read the CSV
    df = pd.read_csv(dataset_path)

    # Ensure datetime field is actually a datetime object
    df['datetime'] = pd.to_datetime(df.datetime)

    # Sort the dataframe by datetime to ensure chronological order
    df = df.sort_values(by='datetime').reset_index(drop=True)


    # Identify numerical features excluding 'datetime' and 'Holiday_ID', 'holiday', 'school' for now
    # We'll use all other numerical columns as features for multivariate forecasting
    features_columns = [col for col in df.columns if col not in ['datetime', 'Holiday_ID', 'holiday', 'school']]
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df = df.drop(columns=['hour', 'datetime', 'Holiday_ID', 'holiday', 'school'])

    print(f"DataFrame after datetime conversion, sorting, and adding/removing columns:\n")
    display(df.head())

    print(f"\nDataFrame info:\n")
    df.info()

    # Convert to numpy array
    data = df.values.astype(np.float32)

    print(f"\nData has been converted into numpy array with shape: {data.shape}")
    return data, df


################################################################################
# Create Dataset types
################################################################################
class ElectricityLoadDataset(Dataset):
    def __init__(self, data, sequence_length, forecast_length, target_idx):
        self.data = data
        self.L = sequence_length
        self.T = forecast_length
        self.target_idx = target_idx

    def __len__(self):
        return self.data.shape[0] - (self.L + self.T) + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.L, :]

        # Y must be multi-channel
        y = self.data[idx + self.L : idx + self.T + self.L, :]
        # y = self.data[idx + self.L : idx + self.T + self.L, target_idx]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        # y = y.unsqueeze(-1)
        # x = x.unsqueeze(-1)
        return x, y

class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, target_idx=None):
        if target_idx is not None:
            return data * self.std[target_idx] + self.mean[target_idx]
        return data * self.std + self.mean