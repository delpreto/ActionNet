import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch


def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return pos * angle_rates

    angle_rads = get_angles(torch.arange(position)[:, None],
                            torch.arange(d_model)[None, :],
                            d_model)

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    return angle_rads[None, ...]


def one_dim_positional_encoding(position, distances):
    gamma = 0.1
    v = 0.5
    list_of_linspaces = [torch.linspace(d * -1, d, position) for d in distances]
    final_tensor = torch.stack(list_of_linspaces).unsqueeze(2)
    final_tensor = torch.exp(-gamma * (((final_tensor ** 2) / v) ** 2))
    return final_tensor


def train(model,
          epochs=300,
          out_folder='../logs_wandb',
          logging=True,
          **kwargs):
    early_stop_callback = EarlyStopping(monitor='loss/val',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=False,
                                        mode='min')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='loss/val', mode='min', filename='checkpoint')
    trainer = pl.Trainer(max_epochs=epochs, detect_anomaly=True,
                         check_val_every_n_epoch=30,
                         devices=1,
                         accelerator='auto',
                         callbacks=[early_stop_callback, checkpoint_callback],
                         enable_checkpointing=logging,
                         logger=TensorBoardLogger(out_folder,
                                                  name=model.cell.__class__.__name__,
                                                  version=model.version_name) if logging else False)
    trainer.fit(model)
    return trainer.validate(model)
