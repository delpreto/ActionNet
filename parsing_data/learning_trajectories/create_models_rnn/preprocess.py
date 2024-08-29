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
            row['object_location_polar'],
            row['hand_location_polar'][0],
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
            row['shoulder_angles'][1:],
            row['hand_location_polar'][1:],
        ], axis=1) for _, row in data_df.iterrows()]

    return np.array(x), np.array(y)


def cartesian_to_polar(xyz):
    if len(xyz.shape) == 2:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)  # Note: arctan2 handles the division by zero issue
        phi = np.arccos(z / r)
        return np.column_stack((r, theta, phi))
    else:
        x, y, z = xyz[0], xyz[1], xyz[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return np.array([r, theta, phi])



def load_df(trajectories='./Data/pouring_training_data.hdf5', reference='./Data/pouring_training_referenceObject_positions.hdf5'):
    training_data_file = h5py.File(trajectories, 'r')
    reference_object_file = h5py.File(reference, 'r')

    reference_object_location = np.squeeze(np.array(reference_object_file['position_m']))

    feature_matrices = np.squeeze(np.array(training_data_file['feature_matrices']))
    labels = training_data_file['labels']

    labels = [label.decode('utf-8') for label in labels]
    print(feature_matrices.shape, reference_object_location.shape)
    hand_location_polar = [cartesian_to_polar(feature_matrices[i, :, 0:3]) for i in range(len(labels))]
    object_location_polar = [cartesian_to_polar(reference_object_location[i]) for i in range(len(labels))]

    df = pd.DataFrame({
        'labels': labels,
        'hand_location': [feature_matrices[i, :, 0:3] for i in range(len(labels))],
        'hand_location_polar': hand_location_polar,
        'hand_quaternion': [feature_matrices[i, :, 9:13] for i in range(len(labels))],
        'wrist_angles': [feature_matrices[i, :, 21:24] for i in range(len(labels))],
        'elbow_angles': [feature_matrices[i, :, 24:27] for i in range(len(labels))],
        'shoulder_angles': [feature_matrices[i, :, 27:30] for i in range(len(labels))],
        'object_location': [reference_object_location[i] for i in range(len(labels))],
        'object_location_polar': object_location_polar,
    })
    # Now, filter the DataFrame
    humans = df[df['labels'] == 'human']
    humans = humans.drop(columns=['labels'])

    return humans


def normalise_data_frame_object_at_origin(df):
    # Extract hand and object locations
    hand_locations = np.stack(df['hand_location'].values)
    object_locations = np.stack(df['object_location'].values)
    print(hand_locations.shape, object_locations.shape)

    normalized_hand_locations = []

    for i in range(len(hand_locations)):
        # Subtract each object location from all corresponding hand locations
        normalized_hand_locations.append(hand_locations[i] - object_locations[i])

        # Update the dataframe with the normalized data
    df['object_location'] = np.zeros_like(object_locations)  # This will be an array of zeros
    df['hand_location'] = normalized_hand_locations

    normalized_hand_locations = np.array(normalized_hand_locations)


    min_vals = normalized_hand_locations.min(axis=(0, 1))  # Axis 0 is the sample, Axis 1 is the point in the sample
    max_vals = normalized_hand_locations.max(axis=(0, 1))

    # Normalize each coordinate of each hand location to range [0, 1]
    df['hand_location'] = [(hand - min_vals) / (max_vals - min_vals) for hand in normalized_hand_locations]

    return df


def normalise_data_frame(df):
    array_col1 = np.concatenate(df['hand_location'].values)
    array_col2 = np.stack(df['object_location'].values)

    # Combine both arrays along the first axis
    combined_array = np.vstack((array_col1, array_col2))

    # Find the min and max for each of the 3 values
    mins = np.min(combined_array, axis=0)
    maxs = np.max(combined_array, axis=0)
    print(mins, maxs)

    # Normalize columns
    df['hand_location'] = [(arr - mins) / (maxs - mins) for arr in df['hand_location']]
    df['object_location'] = [(arr - mins) / (maxs - mins) for arr in df['object_location']]

    array_col1 = np.concatenate(df['hand_location_polar'].values)
    array_col2 = np.stack(df['object_location_polar'].values)

    # Combine both arrays along the first axis
    combined_array = np.vstack((array_col1, array_col2))

    # Find the min and max for each of the 3 values
    mins_polar = np.min(combined_array, axis=0)
    maxs_polar = np.max(combined_array, axis=0)


    df['object_location_polar'] = [(arr - mins_polar) / (maxs_polar - mins_polar) for arr in df['object_location_polar']]
    df['hand_location_polar'] = [(arr - mins_polar) / (maxs_polar - mins_polar) for arr in df['hand_location_polar']]
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
    return df, mins, maxs


def load_and_prep_df(normalize=False, train_set="S00", test_size=0.2):
    if train_set == "S00":
        df = load_df(trajectories='./Data/pouring_training_data_S00.hdf5',
                        reference='./Data/pouring_training_referenceObject_positions_S00.hdf5')
    elif train_set == "S10":
        df = load_df(trajectories='./Data/pouring_training_data_S10.hdf5',
                        reference='./Data/pouring_training_referenceObject_positions_S10.hdf5')
    elif train_set == "S11":
        df = load_df(trajectories='./Data/pouring_training_data_S11.hdf5',
                        reference='./Data/pouring_training_referenceObject_positions_S11.hdf5')
    elif train_set == "all":
        df_S00 = load_df(trajectories='./Data/pouring_training_data_S10.hdf5',
                            reference='./Data/pouring_training_referenceObject_positions_S10.hdf5')
        df_S10 = load_df(trajectories='./Data/pouring_training_data_S00.hdf5',
                            reference='./Data/pouring_training_referenceObject_positions_S00.hdf5')
        df_S11 = load_df(trajectories='./Data/pouring_training_data_S11.hdf5',
                          reference='./Data/pouring_training_referenceObject_positions_S11.hdf5')
        df = pd.concat([df_S00, df_S10, df_S11], ignore_index=True)
    else:
        raise ValueError("Invalid test set specified. Please choose from 'S00', 'S10', 'S11', 'all'.")

    if normalize:
        df, mins, maxs = normalise_data_frame(df)

    train, test = train_test_split(df, train_size=1-test_size, random_state=42)

    return train, test, mins, maxs


def prepare_torch_datasets(device='cpu', normalize=False, train_set='S00', test_size=0.2):
    """
    :param predictions_len: The number of consecutive output predictions made by the neural network model.
    :return: train and eval datasets
    """
    train, test, mins, maxs = load_and_prep_df(normalize=normalize, train_set=train_set, test_size=test_size)
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
