import os.path
import h5py
import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelBinarizer


def min_max_encode(data, min, max):
    return (data - min) / (max - min + 1e-8)


def min_max_decode(data, min, max):
    return data * (max - min) + min


# Normalize function
def normalize_data(data):
    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    return (data - min_val) / (max_val - min_val)


class MotionData(torch.utils.data.Dataset):
    """
    Dataset representing the Salzburg Ticket sales data.
    """

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def split_sequences(data_df):
    # Creating input 'x'
    x = [np.concatenate([
            row['object_location'],
            row['hand_location'][0],
            row['hand_quaternion'][0],
            row['wrist_angles'][0],
            row['elbow_angles'][0],
            row['shoulder_angles'][0]
        ]) for _, row in data_df.iterrows()]

    # Creating target 'y'
    y = [np.concatenate([
            row['hand_location'][1:],
            row['hand_quaternion'][1:],
            row['wrist_angles'][1:],
            row['elbow_angles'][1:],
            row['shoulder_angles'][1:]
        ], axis=1) for _, row in data_df.iterrows()]

    return np.array(x), np.array(y)


def load_df(trajectories='./Data/pouring_training_data.hdf5', reference='./Data/pouring_training_referenceObject_positions.hdf5'):
    training_data_file = h5py.File(trajectories, 'r')
    reference_object_file = h5py.File(reference, 'r')

    reference_object_location = np.squeeze(np.array(reference_object_file['position_m']))

    feature_matrices = np.squeeze(np.array(training_data_file['feature_matrices']))
    labels = training_data_file['labels']
    labels = [label.decode('utf-8') for label in labels]

    df = pd.DataFrame({
        'labels': labels,
        'hand_location': [feature_matrices[i, :, 0:3] for i in range(len(labels))],
        'hand_quaternion': [feature_matrices[i, :, 9:13] for i in range(len(labels))],
        'wrist_angles': [feature_matrices[i, :, 21:24] for i in range(len(labels))],
        'elbow_angles': [feature_matrices[i, :, 24:27] for i in range(len(labels))],
        'shoulder_angles': [feature_matrices[i, :, 27:30] for i in range(len(labels))],
        'object_location': [reference_object_location[i] for i in range(len(labels))],
    })
    # Now, filter the DataFrame
    humans = df[df['labels'] == 'human']
    humans = humans.drop(columns=['labels'])

    return humans


def normalise_data_frame(df):
    array_col1 = np.concatenate(df['hand_location'].values)
    array_col2 = np.stack(df['object_location'].values)

    # Combine both arrays along the first axis
    combined_array = np.vstack((array_col1, array_col2))

    # Find the min and max for each of the 3 values
    mins = np.min(combined_array, axis=0)
    maxs = np.max(combined_array, axis=0)

    # Normalize columns
    df['hand_location'] = [(arr - mins) / (maxs - mins) for arr in df['hand_location']]
    df['object_location'] = [(arr - mins) / (maxs - mins) for arr in df['object_location']]

    # df['hand_location'] = df['hand_location'].apply(lambda x: normalize_data(np.array(x)).tolist())
    # object_location_values = np.array(df['object_location'].tolist())
    # normalized_object_locations = normalize_data(object_location_values)
    # df['object_location'] = normalized_object_locations.tolist()
    #

    # df['hand_quaternion'] = df['hand_quaternion'] / np.pi
    # df['wrist_angles'] = df['wrist_angles'] / np.pi
    # df['elbow_angles'] = df['elbow_angles'] / np.pi
    # df['shoulder_angles'] = df['shoulder_angles'] / np.pi
    # df['hand_quaternion'] = df['hand_quaternion'].apply(lambda x: normalize_data(np.array(x)).tolist())
    # df['wrist_angles'] = df['wrist_angles'].apply(lambda x: normalize_data(np.array(x)).tolist())

    # Display the normalized columns
    # print(df[['hand_location', 'object_location']].head())
    return df


def load_and_prep_df(normalize=False):
    # df_train1 = load_df(trajectories='./Data/10/pouring_training_data.hdf5', reference='./Data/10/pouring_training_referenceObject_positions.hdf5')
    # df_train2 = load_df(trajectories='./Data/11/pouring_training_data.hdf5', reference='./Data/11/pouring_training_referenceObject_positions.hdf5')
    # df_train = pd.concat([df_train1, df_train2], ignore_index=True)
    # df_test = load_df(trajectories='./Data/00/pouring_training_data.hdf5', reference='./Data/00/pouring_training_referenceObject_positions.hdf5')

    df_train1 = load_df(trajectories='./Data/pouring_training_data_S10.hdf5',
                        reference='./Data/pouring_training_referenceObject_positions_S10.hdf5')
    df_train2 = load_df(trajectories='./Data/pouring_training_data_S11.hdf5',
                        reference='./Data/pouring_training_referenceObject_positions_S11.hdf5')
    df_train = pd.concat([df_train1, df_train2], ignore_index=True)
    df_test = load_df(trajectories='./Data/pouring_training_data_S00.hdf5',
                      reference='./Data/pouring_training_referenceObject_positions_S00.hdf5')

    # print(df[['hand_location', 'object_location']].head())
    # print(df['hand_location'][0])
    df = pd.concat([df_train, df_test], ignore_index=True)


    mins, maxs = None, None # TODO
    if normalize:
        df = normalise_data_frame(df)
        df_train = normalise_data_frame(df_train)
        df_test = normalise_data_frame(df_test)


    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test, mins, maxs


def prepare_torch_datasets(device='cpu', normalize=False):
    """
    :param predictions_len: The number of consecutive output predictions made by the neural network model.
    :return: train and eval datasets
    """
    train, test, mins, maxs = load_and_prep_df(normalize=normalize)
    train_x, train_y = split_sequences(train)
    test_x, test_y = split_sequences(test)

    print(train_x.shape)
    print(train_y.shape)

    ds_train = MotionData(train_x, train_y)
    ds_test = MotionData(test_x, test_y)
    return ds_train, ds_test, mins, maxs


if __name__ == "__main__":
    # main()
    print(prepare_torch_datasets(normalize=True))
