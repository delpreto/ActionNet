import argparse
from copy import deepcopy
from typing import Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import h5py
import os
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
actionsense_root_dir = script_dir
while os.path.split(actionsense_root_dir)[-1] != 'ActionSense':
  actionsense_root_dir = os.path.realpath(os.path.join(actionsense_root_dir, '..'))

from learning_trajectories.create_models_rnn.models import LSTMVanilla, GRUVanilla, NCPVanilla
from learning_trajectories.create_models_rnn.preprocess import prepare_torch_datasets, cartesian_to_polar
from learning_trajectories.create_models_rnn.utils import positional_encoding
from learning_trajectories.create_models_rnn.utils import one_dim_positional_encoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hide_figure_windows = False
if hide_figure_windows:
    try:
        matplotlib.use('Agg')
    except:
        pass

torch.manual_seed(422)
np.random.seed(422)
random.seed(422)

def add_args(_parser):
    _parser.add_argument("--model_name", type=str, default='GRU', help="Model to use (LSTM, GRU, NCP)")
    _parser.add_argument("--cpu", action='store_true', help="Force training on CPU, even if CUDA is available.")
    _parser.add_argument("--size", type=int, default=64,
                         help="Size of the model (could be used to set hidden layers sizes), default: 64")
    _parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs, default: 30")
    _parser.add_argument("--batch_size", type=int, default=16, help="Training batch size, default: 16")
    _parser.add_argument("--lr", type=float, default=0.001, help="Learning rate, default: 0.001")
    _parser.add_argument("--train_set", type=str, default='S00,S11', help="Training set to use, default: S00, options: "
                                                                      "S00, S10, S11, all")
    _parser.add_argument("--test_set", type=str, default=None,
                         help="Size of the test set, default: 0.2")
    _parser.add_argument("--test_set_size", type=float, default=0.2,
                         help="Size of the test set, default: 0.2")
    _parser.add_argument("--output_dir", type=str, default=None)
        # os.path.join(actionsense_root_dir, 'results', 'learning_trajectories', 'models', 'rnns'),
        #                  help="Folder to save the model checkpoint")
    _parser.add_argument("--data_dir", type=str, default=
        os.path.join(actionsense_root_dir, 'results', 'learning_trajectories', 'humans'),
                         help="Folder with training data HDF5 files, default: ./Data")
    _parser.add_argument("--output_name", type=str, default='model_train-all',
                         help="Name of the output file, default: model")
    _parser.add_argument("--loss_coefficient", type=float, default=1.0,
                         help="Coefficient to scale the loss, default: 1.0")
    _parser.add_argument("--loss_speed_regularization", type=bool, default=True,
                         help="use speed regularization in loss function, default: True")
    return _parser

def plot_points(input_features, points_true, points_predicted):
    print('Plotting all test set trajectories in one figure')
    n = points_predicted.shape[0]  # Number of plots
    # n = 4
    fig = plt.figure(figsize=(15, 15))  # Increase figure size for clarity
    # plt.get_current_fig_manager().window.showMaximized()
    
    # Determine grid size
    cols = np.floor(n ** 0.5).astype(int)
    rows = np.ceil(n/cols).astype(int)
    # distances = torch.norm(input_features[:, 0:3] - input_features[:, 3:6], dim=1)
    # print(ref.shape, distances.shape)
    
    def plot_trial(ax, trial_index):
        # Plot the human trajectory.
        if points_true is not None:
            x_human = points_true[trial_index, :, 0]
            y_human = points_true[trial_index, :, 1]
            z_human = points_true[trial_index, :, 2]
            ax.scatter(x_human, y_human, z_human,
                       c='g', marker='o', s=10, alpha=0.5)
        # Plot the model trajectory.
        x_model = points_predicted[trial_index, :, 0]
        y_model = points_predicted[trial_index, :, 1]
        z_model = points_predicted[trial_index, :, 2]
        ax.scatter(x_model, y_model, z_model,
                   c='b', marker='o', s=10, alpha=0.5)
        # Plot the human starting hand position.
        ax.scatter(input_features[trial_index, 9], input_features[trial_index, 10], input_features[trial_index, 11],
                   c='red', marker='o', s=30)
        # Plot the first model position.
        # Note: index 0 has been set to the target start, so the first generated timestep is index 1.
        ax.scatter(x_model[1], y_model[1], z_model[1],
                   c='m', marker='o', s=30, alpha=0.5)
        # Plot the reference object position.
        ax.scatter(input_features[trial_index, 0], input_features[trial_index, 1], input_features[trial_index, 2],
                   c='black', marker='o', s=30)
        # Format the plot.
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    for trial_index in range(n):
        # print(trial_index, distances[trial_index].item())
        ax = fig.add_subplot(rows, cols, trial_index + 1, projection='3d')
        plot_trial(ax, trial_index)
    plt.tight_layout()
    
    # Save the plot if desired.
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, args.output_name+'_test_set_trajectories.png'), dpi=300)
    
    # Plot in groups of 4 so the plots are bigger.
    rows = 2
    cols = 3
    figs = []
    for trial_index in range(n):
        if trial_index % (rows*cols) == 0:
            print('Plotting test set trials %d-%d in one figure' % (trial_index, trial_index+rows*cols-1))
            fig = plt.figure()
            if not hide_figure_windows:
                plt.get_current_fig_manager().window.showMaximized()
            figs.append(fig)
        ax = fig.add_subplot(rows, cols, (trial_index % (rows*cols)) + 1, projection='3d')
        plot_trial(ax, trial_index)
    # Save the plots if desired.
    for (window_index, fig) in enumerate(figs):
        plt.figure(fig)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, args.output_name+'_test_set_trajectories_figure%02d.png' % window_index), dpi=300)


def plot_trajectory(model: nn.Module, y_true, features, mins_byFrame, maxs_byFrame):
    features = features.to(device=device, dtype=torch.float)  # bring inputs to same device
    y_true = y_true.to(device=device, dtype=torch.float)
    
    # Convert to Numpy arrays.
    y_true_np = y_true.detach().cpu().numpy()
    features_np = features.detach().cpu().numpy()

    # Compute one-dimensional positional encoding
    distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
    pos_encoding = one_dim_positional_encoding(99, distances).to(
        device=device)  # Adjust the size according to your model's input
    # Predict a trajectory for each test scenario.
    y_pred = model(features, pos_encoding)
    y_pred_np = y_pred.detach().cpu().numpy()
    # Add the starting position.
    empty_entries = np.zeros(shape=(y_pred_np.shape[0], 1, y_pred_np.shape[2]), dtype=y_pred_np.dtype)
    y_pred_np = np.concatenate([empty_entries, y_pred_np], axis=1)
    for trial_index in range(y_pred_np.shape[0]):
        starting_hand_position = features_np[trial_index, 9:12]
        starting_hand_quaternion = features_np[trial_index, 12:16]
        y_pred_np[trial_index, 0, 0:3] = starting_hand_position
        y_pred_np[trial_index, 0, 3:7] = starting_hand_quaternion
        y_pred_np[trial_index, 0, 7:10] = cartesian_to_polar(starting_hand_position)
    # Generate a time vector.
    time_s_pred = [np.linspace(0, 7.7, y_pred_np.shape[1])[:,None] for trial_index in range(y_pred_np.shape[0])]

    print('features_np.shape', features_np.shape)
    print('y_pred_np.shape', y_pred_np.shape)
    print('y_true_np.shape', y_true_np.shape)
    
    # Denormalize.
    print("**************************", y_pred_np.shape, features.shape)
    old_features_np = features_np.copy()
    old_y_pred_np = y_pred_np.copy()
    if maxs_byFrame is not None and 'hand_location' in maxs_byFrame:
        y_pred_np[:, :, 0:3] = y_pred_np[:, :, 0:3] * (maxs_byFrame['hand_location'].reshape(1, 1, -1) - mins_byFrame['hand_location'].reshape(1, 1, -1)) + mins_byFrame['hand_location'].reshape(1, 1, -1)
        y_true_np[:, :, 0:3] = y_true_np[:, :, 0:3] * (maxs_byFrame['hand_location'].reshape(1, 1, -1) - mins_byFrame['hand_location'].reshape(1, 1, -1)) + mins_byFrame['hand_location'].reshape(1, 1, -1)
        features_np[:, 9:12] = features_np[:, 9:12] * (maxs_byFrame['hand_location'].reshape(1, 1, -1) - mins_byFrame['hand_location'].reshape(1, 1, -1)) + mins_byFrame['hand_location'].reshape(1, 1, -1)
    if maxs_byFrame is not None and 'hand_quaternion' in maxs_byFrame:
        y_pred_np[:, :, 3:7] = y_pred_np[:, :, 3:7] * (maxs_byFrame['hand_quaternion'].reshape(1, 1, -1) - mins_byFrame['hand_quaternion'].reshape(1, 1, -1)) + mins_byFrame['hand_quaternion'].reshape(1, 1, -1)
        y_true_np[:, :, 3:7] = y_true_np[:, :, 3:7] * (maxs_byFrame['hand_quaternion'].reshape(1, 1, -1) - mins_byFrame['hand_quaternion'].reshape(1, 1, -1)) + mins_byFrame['hand_quaternion'].reshape(1, 1, -1)
        features_np[:, 12:16] = features_np[:, 12:16] * (maxs_byFrame['hand_quaternion'].reshape(1, 1, -1) - mins_byFrame['hand_quaternion'].reshape(1, 1, -1)) + mins_byFrame['hand_quaternion'].reshape(1, 1, -1)
    if maxs_byFrame is not None and 'hand_location_polar' in maxs_byFrame:
        y_pred_np[:, :, 7:10] = y_pred_np[:, :, 7:10] * (maxs_byFrame['hand_location_polar'].reshape(1, 1, -1) - mins_byFrame['hand_location_polar'].reshape(1, 1, -1)) + mins_byFrame['hand_location_polar'].reshape(1, 1, -1)
        y_true_np[:, :, 7:10] = y_true_np[:, :, 7:10] * (maxs_byFrame['hand_location_polar'].reshape(1, 1, -1) - mins_byFrame['hand_location_polar'].reshape(1, 1, -1)) + mins_byFrame['hand_location_polar'].reshape(1, 1, -1)
        features_np[:, 6:9] = features_np[:, 6:9] * (maxs_byFrame['hand_location_polar'].reshape(1, 1, -1) - mins_byFrame['hand_location_polar'].reshape(1, 1, -1)) + mins_byFrame['hand_location_polar'].reshape(1, 1, -1)
    if maxs_byFrame is not None and 'object_location' in maxs_byFrame:
        features_np[:, 0:3] = features_np[:, 0:3] * (maxs_byFrame['object_location'].reshape(1, 1, -1) - mins_byFrame['object_location'].reshape(1, 1, -1)) + mins_byFrame['object_location'].reshape(1, 1, -1)
    if maxs_byFrame is not None and 'object_location_polar' in maxs_byFrame:
        features_np[:, 3:6] = features_np[:, 3:6] * (maxs_byFrame['object_location_polar'].reshape(1, 1, -1) - mins_byFrame['object_location_polar'].reshape(1, 1, -1)) + mins_byFrame['object_location_polar'].reshape(1, 1, -1)
    # print('-'*50)
    # print('Object location denormalizing')
    # for i in range(features_np.shape[0]):
    #     print(i, old_features_np[i, 0:3], ' >> ', features_np[i, 0:3])
    # print('-'*50)
    # print('Starting hand location denormalizing')
    # for i in range(y_pred_np.shape[0]):
    #     print(i, old_y_pred_np[i, 0:3], ' >> ', y_pred_np[i, 0:3])
    
    # Plot the points.
    plot_points(features_np, y_true_np, y_pred_np)
    
    # Save the output data as an HDF5 file.
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        fout = h5py.File(os.path.join(args.output_dir, args.output_name+'_output_data.hdf5'), 'w')
        fout.create_dataset('hand_position_m', data=y_pred_np[:,:,0:3])
        fout.create_dataset('hand_quaternion_wijk', data=y_pred_np[:,:,3:7])
        fout.create_dataset('hand_position_polar', data=y_pred_np[:,:,7:11])
        fout.create_dataset('referenceObject_position_m', data=features_np[:,0:3])
        fout.create_dataset('time_s', data=time_s_pred)
        # fout.create_dataset('wrist_angles', data=y_pred_np[:,:,7:10])
        # fout.create_dataset('elbow_angles', data=y_pred_np[:,:,10:13])
        # fout.create_dataset('shoulder_angles', data=y_pred_np[:,:,13:16])
        # fout.create_dataset('hand_location_polar', data=y_pred_np[:,:,16:19])
        truth_group = fout.create_group('truth')
        truth_group.create_dataset('referenceObject_position_m', data=features_np[:,0:3])
        truth_group.create_dataset('referenceObject_position_polar', data=features_np[:,3:6])
        truth_group.create_dataset('hand_position_polar', data=features_np[:,6:9])
        truth_group.create_dataset('hand_position_m', data=features_np[:,9:12])
        truth_group.create_dataset('hand_quaternion_wijk', data=features_np[:,12:16])
        # truth_group.create_dataset('object_location', data=features_np[:,0:3])
        # truth_group.create_dataset('object_location_polar', data=features_np[:,3:6])
        # truth_group.create_dataset('hand_location_polar', data=features_np[:,6:9])
        # truth_group.create_dataset('hand_location', data=features_np[:,9:12])
        # truth_group.create_dataset('hand_quaternion', data=features_np[:,12:16])
        # truth_group.create_dataset('wrist_angles', data=features_np[:,16:19])
        # truth_group.create_dataset('elbow_angles', data=features_np[:,19:22])
        # truth_group.create_dataset('shoulder_angles', data=features_np[:,22:25])
        fout.close()
        # fout = h5py.File(os.path.join(args.output_dir, args.output_name+'_referenceObject_positions.hdf5'), 'w')
        # fout.create_dataset('position_m', data=features_np[:, 0:3])
        # fout.close()

def train_step(batch: Tuple[torch.Tensor, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn,
               device: torch.device, loss_coefficient, loss_speed_regularization):
    features, y = batch
    features = features.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)

    # Compute one-dimensional positional encoding
    distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
    pos_encoding = one_dim_positional_encoding(99, distances).to(
        device=device)  # Adjust the size according to your model's input

    y_pred = model(features, pos_encoding)

    if loss_speed_regularization:
        movement_diffs = y[:, 1:, 0:3] - y[:, :-1, 0:3]

        # Calculate the Euclidean distances for these differences
        movement_distances = torch.sqrt(torch.sum(movement_diffs ** 2, dim=2))
        threshold = 0.004  # minimal movement threshold
        stationary_mask = (movement_distances < threshold)

        # Compute mean squared error loss without reduction
        loss_per_timestep = (y_pred - y) ** 2

        # Apply higher weights to stationary points
        weights = torch.ones_like(loss_per_timestep)
        weights[:, :-1][stationary_mask] = 15  # Increase weight where movement is minimal

        # Compute the weighted loss
        loss = (loss_per_timestep * weights).mean()
    else:
        loss = loss_fn(loss_coefficient * y_pred, loss_coefficient * y)


    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss


def evaluate(model: nn.Module, features, pos_encoding, y, loss_fn, device: torch.device, loss_coefficient):
    features = features.to(device=device, dtype=torch.float)  # bring inputs to same device
    y = y.to(device=device, dtype=torch.float)

    y_pred = model(features, pos_encoding)
    loss = loss_fn(y_pred, y)

    return {
        'loss': loss.cpu().item()
    }


def train(model: nn.Module, datasets: Tuple[TensorDataset, Optional[TensorDataset]],
          dataset_mins_byFrame, dataset_maxs_byFrame, num_epochs=20, batch_size=32,
          loss_coefficient=10, lr=0.001, checkpoint_dir: str = None, checkpoint_model_base_name: str = None,
          loss_speed_regularization=True):
    print('Preparing to train')
    train_set, test_set = datasets
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device, dtype=torch.float)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logs = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    best_model = model
    
    training_losses = []
    testing_losses = []
    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
        # if epoch % max(1, num_epochs//20) == 0 or epoch == num_epochs-1:
        #     print('  Training epoch %d/%d' % (epoch+1, num_epochs))
        model.train()  # Set model to training mode
        total_loss = 0

        for features, y in train_dataloader:
            features = features.to(device)
            y = y.to(device)

            # Compute one-dimensional positional encoding for each batch
            distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
            pos_encoding = one_dim_positional_encoding(99, distances).to(
                device)  # Adjust according to your model's input

            loss = train_step(
                batch=(features, y),
                model=model,
                optimizer=optimizer,
                loss_fn=criterion,
                device=device,
                loss_coefficient=loss_coefficient,
                loss_speed_regularization=loss_speed_regularization
            )
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        logs['train_loss'].append(avg_train_loss)

        if test_set is not None:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                test_dataloader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
                for features, y in test_dataloader:
                    features = features.to(device)
                    y = y.to(device)

                    # Compute one-dimensional positional encoding for the test set
                    distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
                    pos_encoding = one_dim_positional_encoding(99, distances).to(device)  # Adjust accordingly

                    evaluation = evaluate(
                        model=model,
                        features=features,
                        pos_encoding=pos_encoding,
                        y=y,
                        loss_fn=criterion,
                        device=device,
                        loss_coefficient=loss_coefficient
                    )

                val_loss = evaluation['loss']
                logs['val_loss'].append(val_loss)

                # if val_loss < best_loss:
                if epoch == num_epochs-1:
                    best_loss = val_loss
                    best_model = deepcopy(model)
                    if checkpoint_dir is not None:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        # print('  Saving checkpoint model')
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoint_dir, '%s_lowest_loss_epoch_%03d.pt' %
                                                (checkpoint_model_base_name, epoch)))

        # print(f'Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        training_losses.append(avg_train_loss)
        testing_losses.append(val_loss)
    
    plt.figure()
    if not hide_figure_windows:
        plt.get_current_fig_manager().window.showMaximized()
    plt.plot(training_losses, label='Training')
    plt.plot(testing_losses, label='Testing')
    plt.legend()
    plt.grid(True, color='lightgray')
    plt.title('Training and Testing Set Losses')
    plt.xlabel('Epoch Index')
    plt.ylabel('Loss')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, args.output_name+'_losses.png'), dpi=300)
      
    plot_trajectory(best_model, y, features, dataset_mins_byFrame, dataset_maxs_byFrame)
    
    if not hide_figure_windows:
        print('Close all plot windows to continue')
        plt.show()
    
    return logs, best_model


# Training logic remains mostly unchanged
# Ensure positional encoding is computed and passed where necessary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()

    # Dynamically select the model based on command line argument
    if args.model_name == 'LSTM':
        model = LSTMVanilla(input_size=17, hidden_size=args.size, output_size=10, num_layers=2)
    elif args.model_name == 'GRU':
        model = GRUVanilla(input_size=17, hidden_size=args.size, output_size=10, num_layers=2)
    elif args.model_name == 'NCP':
        model = NCPVanilla(input_size=17, hidden_size=args.size, output_size=10)
    # elif args.model_name == 'LEM':
    #     model = LEM(ninp=17, nhid=args.size, nout=19, dt=0.2371)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model = model.float().to(device)

    train_set, test_set, mins_byFrame, maxs_byFrame = prepare_torch_datasets(
        normalize=True, train_set=args.train_set, test_size=args.test_set_size, test_set=args.test_set, data_dir=args.data_dir)
    input('Press enter to start training the model ')
    logs, best_model = train(model=model, datasets=(train_set, test_set),
                             dataset_mins_byFrame=mins_byFrame, dataset_maxs_byFrame=maxs_byFrame,
                             checkpoint_dir=os.path.join(args.output_dir, 'models'),
                             checkpoint_model_base_name=args.model_name,
                             num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                             loss_coefficient=args.loss_coefficient,
                             loss_speed_regularization=args.loss_speed_regularization)
    print(logs)
