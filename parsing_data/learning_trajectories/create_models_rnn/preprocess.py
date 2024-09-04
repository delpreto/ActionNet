import copy
import os.path
import h5py
import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelBinarizer

# def min_max_encode(data, min, max):
#     return (data - min) / (max - min + 1e-8)
#
#
# def min_max_decode(data, min, max):
#     return data * (max - min) + min
#
#
# # Normalize function
# def normalize_data(data):
#     min_val = np.min(data, axis=(0, 1), keepdims=True)
#     max_val = np.max(data, axis=(0, 1), keepdims=True)
#     return (data - min_val) / (max_val - min_val)


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
            # row['object_location_polar'],
            # row['hand_location_polar'][0],
            row['hand_location'][0],
            # row['hand_quaternion'][0],
            # row['wrist_angles'][0],
            # row['elbow_angles'][0],
            # row['shoulder_angles'][0]
        ]) for _, row in data_df.iterrows()]

    # Creating target 'y'
    y = [np.concatenate([
            row['hand_location'][:],
            # row['hand_quaternion'][1:],
            # row['wrist_angles'][1:],
            # row['elbow_angles'][1:],
            # row['shoulder_angles'][1:],
            # row['hand_location_polar'][1:],
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



def load_df(data_filepath):
    print('Loading dataset from: %s' % data_filepath)
    training_data_file = h5py.File(data_filepath, 'r')

    reference_object_location = np.squeeze(np.array(training_data_file['referenceObject_position_m']))
    print('  reference_object_location.shape:', reference_object_location.shape)

    labels = training_data_file['labels']
    labels = [label.decode('utf-8') for label in labels]
    
    hand_location_polar = [cartesian_to_polar(np.array(training_data_file['hand_position_m'][i, :, :])) for i in range(len(labels))]
    object_location_polar = [cartesian_to_polar(reference_object_location[i]) for i in range(len(labels))]

    df = pd.DataFrame({
        'labels': labels,
        'hand_location': [np.array(training_data_file['hand_position_m'][i, :, :]) for i in range(len(labels))],
        'hand_location_polar': hand_location_polar,
        'hand_quaternion': [np.array(training_data_file['hand_quaternion_wijk'][i, :, :]) for i in range(len(labels))],
        'wrist_angles': [np.array(training_data_file['wrist_joint_angle_xyz_rad'][i, :, :]) for i in range(len(labels))],
        'elbow_angles': [np.array(training_data_file['elbow_joint_angle_xyz_rad'][i, :, :]) for i in range(len(labels))],
        'shoulder_angles': [np.array(training_data_file['shoulder_joint_angle_xyz_rad'][i, :, :]) for i in range(len(labels))],
        'object_location': [reference_object_location[i, :] for i in range(len(labels))],
        'object_location_polar': object_location_polar,
    })
    # Now, filter the DataFrame
    humans = df[df['labels'] == 'human']
    humans = humans.drop(columns=['labels'])

    return humans


# def normalise_data_frame_object_at_origin(df):
#     # Extract hand and object locations
#     hand_locations = np.stack(df['hand_location'].values)
#     object_locations = np.stack(df['object_location'].values)
#     print(hand_locations.shape, object_locations.shape)
#
#     normalized_hand_locations = []
#
#     for i in range(len(hand_locations)):
#         # Subtract each object location from all corresponding hand locations
#         normalized_hand_locations.append(hand_locations[i] - object_locations[i])
#
#         # Update the dataframe with the normalized data
#     df['object_location'] = np.zeros_like(object_locations)  # This will be an array of zeros
#     df['hand_location'] = normalized_hand_locations
#
#     normalized_hand_locations = np.array(normalized_hand_locations)
#
#
#     min_vals = normalized_hand_locations.min(axis=(0, 1))  # Axis 0 is the sample, Axis 1 is the point in the sample
#     max_vals = normalized_hand_locations.max(axis=(0, 1))
#
#     # Normalize each coordinate of each hand location to range [0, 1]
#     df['hand_location'] = [(hand - min_vals) / (max_vals - min_vals) for hand in normalized_hand_locations]
#
#     return df


def normalise_data_frame(df, mins_byFrame=None, maxs_byFrame=None):
    print('Normalizing data frames')
    
    if mins_byFrame is None or maxs_byFrame is None:
        mins_byFrame = {}
        maxs_byFrame = {}
        
        # Normalize cartesian spatial features.
        array_col1 = np.concatenate(df['hand_location'].values)
        array_col2 = np.stack(df['object_location'].values)
        combined_array = np.vstack((array_col1, array_col2))
        mins = np.min(combined_array, axis=0)
        maxs = np.max(combined_array, axis=0)
        print('  Hand and object position min/max:', mins, maxs)
        mins_byFrame['hand_location'] = mins
        maxs_byFrame['hand_location'] = maxs
        mins_byFrame['object_location'] = mins
        maxs_byFrame['object_location'] = maxs
        # new_array_col2 = np.stack(df['object_location'].values)
        # print('-'*50)
        # print('Object location normalizing')
        # for i in range(new_array_col2.shape[0]):
        #     print(i, array_col2[i], ' >> ', new_array_col2[i])
        # print('-'*50)
        # print('Starting hand location normalizing')
        # new_array_col1 = np.concatenate(df['hand_location'].values)
        # for i in range(new_array_col2.shape[0]):
        #     print(i, array_col1[i], ' >> ', new_array_col1[i])
        
        # Normalize polar spatial features.
        array_col1 = np.concatenate(df['hand_location_polar'].values)
        array_col2 = np.stack(df['object_location_polar'].values)
        combined_array = np.vstack((array_col1, array_col2))
        mins = np.min(combined_array, axis=0)
        maxs = np.max(combined_array, axis=0)
        print('  Hand and object polar min/max:', mins, maxs)
        mins_byFrame['hand_location_polar'] = mins
        maxs_byFrame['hand_location_polar'] = maxs
        mins_byFrame['object_location_polar'] = mins
        maxs_byFrame['object_location_polar'] = maxs
        
        # Quaternions should be (essentially) unit quaternions and thus already normalized.
    
    for key in df:
        if key in mins_byFrame and key in maxs_byFrame:
            df[key] = [(arr - mins_byFrame[key]) / (maxs_byFrame[key] - mins_byFrame[key]) for arr in df[key]]
    # df['hand_location'] = [(arr - mins) / (maxs - mins) for arr in df['hand_location']]
    # df['object_location'] = [(arr - mins) / (maxs - mins) for arr in df['object_location']]
    # df['hand_location_polar'] = [(arr - mins) / (maxs - mins) for arr in df['hand_location_polar']]
    # df['object_location_polar'] = [(arr - mins) / (maxs - mins) for arr in df['object_location_polar']]
    
    
    # # Normalize joint angles.
    # array_col1 = np.concatenate(df['wrist_angles'].values)
    # array_col2 = np.concatenate(df['elbow_angles'].values)
    # array_col3 = np.concatenate(df['shoulder_angles'].values)
    # combined_array = np.concatenate((array_col1, array_col2, array_col3), axis=0).shape
    # mins = np.min(combined_array, axis=0)
    # maxs = np.max(combined_array, axis=0)
    # print('  Joint angles min/max [deg]:', np.degrees(mins), np.degrees(maxs))
    # df['wrist_angles'] = [(arr - mins) / (maxs - mins) for arr in df['wrist_angles']]
    # df['elbow_angles'] = [(arr - mins) / (maxs - mins) for arr in df['elbow_angles']]
    # df['shoulder_angles'] = [(arr - mins) / (maxs - mins) for arr in df['shoulder_angles']]
    # mins_byFrame['wrist_angles'] = mins
    # maxs_byFrame['wrist_angles'] = maxs
    # mins_byFrame['elbow_angles'] = mins
    # maxs_byFrame['elbow_angles'] = maxs
    # mins_byFrame['shoulder_angles'] = mins
    # maxs_byFrame['shoulder_angles'] = maxs
    
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
    return df, mins_byFrame, maxs_byFrame


def load_and_prep_df(normalize=False, train_set="S00", test_set=None, test_size=0.2, data_dir='./Data'):
    df_S00 = load_df(data_filepath=os.path.join(data_dir, 'pouring_trainingData_S00.hdf5'))
    df_S10 = load_df(data_filepath=os.path.join(data_dir, 'pouring_trainingData_S10.hdf5'))
    df_S11 = load_df(data_filepath=os.path.join(data_dir, 'pouring_trainingData_S11.hdf5'))
    dfs_all = []
    
    print('Loading datasets: train set [%s] test set [%s] test size [%s]' % (train_set, test_set, test_size))
    if train_set == 'all':
        train_set = 'S00,S10,S11'
    train_sets = train_set.split(',')
    dfs_train = []
    if "S00" in train_sets:
        dfs_train.append(copy.deepcopy(df_S00))
        dfs_all.append(copy.deepcopy(df_S00))
        # dfs.append(load_df(trajectories=os.path.join(data_dir, 'pouring_training_data_S00.hdf5'),
        #                    reference=os.path.join(data_dir, 'pouring_training_referenceObject_positions_S00.hdf5')))
    if "S10" in train_sets:
        dfs_train.append(copy.deepcopy(df_S10))
        dfs_all.append(copy.deepcopy(df_S10))
    if "S11" in train_sets:
        dfs_train.append(copy.deepcopy(df_S11))
        dfs_all.append(copy.deepcopy(df_S11))
    
    if test_set is None:
        df = pd.concat(dfs_train, ignore_index=True)
    
        if normalize:
            df, mins_byFrame, maxs_byFrame = normalise_data_frame(df)
        else:
            mins_byFrame = None
            maxs_byFrame = None
        
        print('Splitting the dataset with test_size=%g' % test_size)
        train, test = train_test_split(df, train_size=1-test_size, random_state=42)
        # train = pd.concat([df_S00,df_S11], ignore_index=True)
        # test = pd.concat([df_S10,], ignore_index=True)

        return train, test, mins_byFrame, maxs_byFrame
    else:
        if test_set == 'all':
            test_set = 'S00,S10,S11'
        test_sets = test_set.split(',')
        dfs_test = []
        if "S00" in test_sets:
            dfs_test.append(copy.deepcopy(df_S00))
            dfs_all.append(copy.deepcopy(df_S00))
            # dfs.append(load_df(trajectories=os.path.join(data_dir, 'pouring_training_data_S00.hdf5'),
            #                    reference=os.path.join(data_dir, 'pouring_training_referenceObject_positions_S00.hdf5')))
        if "S10" in test_sets:
            dfs_test.append(copy.deepcopy(df_S10))
            dfs_all.append(copy.deepcopy(df_S10))
        if "S11" in test_sets:
            dfs_test.append(copy.deepcopy(df_S11))
            dfs_all.append(copy.deepcopy(df_S11))
        df_test = pd.concat(dfs_test, ignore_index=True)
        df_train = pd.concat(dfs_train, ignore_index=True)
        df_all = pd.concat(dfs_all, ignore_index=True)
        if normalize:
            _, mins_byFrame, maxs_byFrame = normalise_data_frame(df_all)
            df_test, _, _ = normalise_data_frame(df_test, mins_byFrame=mins_byFrame, maxs_byFrame=maxs_byFrame)
            df_train, _, _ = normalise_data_frame(df_train, mins_byFrame=mins_byFrame, maxs_byFrame=maxs_byFrame)
        else:
            mins_byFrame = None
            maxs_byFrame = None
        return df_train, df_test, mins_byFrame, maxs_byFrame


def prepare_torch_datasets(device='cpu', normalize=False, train_set='S00', test_set=None, test_size=0.2, data_dir='./Data'):
    """
    :param predictions_len: The number of consecutive output predictions made by the neural network model.
    :return: train and eval datasets
    """
    print('Preparing the dataset')
    train, test, mins_byFrame, maxs_byFrame = load_and_prep_df(normalize=normalize,
                                                               train_set=train_set,
                                                               test_set=test_set, test_size=test_size,
                                                               data_dir=data_dir)
    train_x, train_y = split_sequences(train)
    test_x, test_y = split_sequences(test)

    print('train_x.shape:', train_x.shape)
    print('train_y.shape:', train_y.shape)

    ds_train = MotionData(train_x, train_y)
    ds_test = MotionData(test_x, test_y)
    return ds_train, ds_test, mins_byFrame, maxs_byFrame


if __name__ == "__main__":
    # main()
    print(prepare_torch_datasets(normalize=True))
