import argparse
from copy import deepcopy
import pickle
from typing import Optional, Tuple
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from models import LSTMVanilla, GRUVanilla, NCPVanilla
from preprocess import prepare_torch_datasets, min_max_decode
from utils import positional_encoding, \
    one_dim_positional_encoding

import h5py
import os
output_data_dir = 'C:/Users/jdelp/Desktop/ActionSense/code/parsing_data/learning_trajectories/creating_models/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    _parser.add_argument("--checkpoint_path", type=str, default=None,
                         help="Path to save the model checkpoint, default: None")
    _parser.add_argument("--loss_coefficient", type=float, default=1.0,
                         help="Coefficient to scale the loss, default: 1.0")
    return _parser


def plot_points(ref, points, preds):
    n = points.shape[0]  # Number of plots
    n = 16
    fig = plt.figure(figsize=(15, 15))  # Increase figure size for clarity

    # Determine grid size
    rows = int(n ** 0.5)
    cols = n // rows + (n % rows)
    distances = torch.norm(ref[:, 0:3] - ref[:, 3:6], dim=1)
    print(ref.shape, distances.shape)

    for j in range(n):
        print(j, distances[j].item())
        ax = fig.add_subplot(rows, cols, j + 1, projection='3d')
        ax.scatter(ref[j, 0], ref[j, 1], ref[j, 2], c='black', marker='o')
        x = points[j, :, 0]
        y = points[j, :, 1]
        z = points[j, :, 2]
        x2 = preds[j, :, 0]
        y2 = preds[j, :, 1]
        z2 = preds[j, :, 2]
        ax.scatter(x, y, z, c='r', marker='o')
        ax.scatter(x2, y2, z2, c='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.tight_layout()
    plt.show()


def plot_trajectory(model: nn.Module, labels, features):
    features = features.to(device=device, dtype=torch.float)  # bring inputs to same device
    labels = labels.to(device=device, dtype=torch.float)

    # Compute one-dimensional positional encoding
    distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
    pos_encoding = one_dim_positional_encoding(99, distances).to(
        device=device)  # Adjust the size according to your model's input

    y_pred = model(features, pos_encoding)

    # Convert to Numpy for plotting
    y_pred_np = y_pred.detach().cpu().numpy()
    y_np = labels.detach().cpu().numpy()
    
    # Save the output data as an HDF5 file.
    if output_data_dir is not None:
        fout = h5py.File(os.path.join(output_data_dir, 'model_output_data.hdf5'), 'w')
        fout.create_dataset('feature_matrices', data=y_pred_np)
        fout.close()
        fout = h5py.File(os.path.join(output_data_dir, 'model_referenceObject_positions.hdf5'), 'w')
        fout.create_dataset('position_m', data=features[:, 0:3])
        fout.close()
        plot_points(features.cpu(), labels.cpu(), y_pred_np)


def train_step(batch: Tuple[torch.Tensor, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn,
               device: torch.device, loss_coefficient):
    features, y = batch
    features = features.to(device=device, dtype=torch.float)
    y = y.to(device=device, dtype=torch.float)

    # Compute one-dimensional positional encoding
    distances = torch.norm(features[:, 0:3] - features[:, 3:6], dim=1)
    pos_encoding = one_dim_positional_encoding(99, distances).to(
        device=device)  # Adjust the size according to your model's input

    y_pred = model(features, pos_encoding)
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
    loss = loss_fn(loss_coefficient * y_pred, loss_coefficient * y)

    return {
        'loss': loss.cpu().item()
    }


def train(model: nn.Module, datasets: Tuple[TensorDataset, Optional[TensorDataset]], num_epochs=20, batch_size=32,
          loss_coefficient=10, lr=0.001, checkpoint: str = None):
    train_set, test_set = datasets
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device, dtype=torch.float)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logs = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    best_model = model

    for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
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
                loss_coefficient=loss_coefficient
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

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = deepcopy(model)
                    if checkpoint:
                        torch.save(model.state_dict(), checkpoint)

        print(f'Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
    plot_trajectory(best_model, y, features)

    return logs, best_model


# Training logic remains mostly unchanged
# Ensure positional encoding is computed and passed where necessary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser).parse_args()

    # Dynamically select the model based on command line argument
    if args.model_name == 'LSTM':
        model = LSTMVanilla(input_size=20, hidden_size=args.size, output_size=16, num_layers=2)
    elif args.model_name == 'GRU':
        model = GRUVanilla(input_size=20, hidden_size=args.size, output_size=16, num_layers=2)
    elif args.model_name == 'NCP':
        model = NCPVanilla(input_size=20, hidden_size=args.size, output_size=16)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model = model.float().to(device)

    train_set, test_set, mins, maxs = prepare_torch_datasets(normalize=True)
    logs, best_model = train(model=model, datasets=(train_set, test_set), checkpoint=f'models/{args.model_name}.pt',
                 num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, loss_coefficient=args.loss_coefficient)
    print(logs)
