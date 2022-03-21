# %%
from ast import Tuple
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import NeptuneLogger
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


# %%
class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, time_series: np.ndarray, seq_len: int, y_size: int):
        self.time_series = torch.tensor(time_series).float()
        self.seq_len = seq_len
        self.y_size = y_size

    def __len__(self):
        return self.time_series.__len__() - (self.seq_len + self.y_size - 1)

    def __getitem__(self, index):
        # return x, y
        return (
            self.time_series[index: index + self.seq_len],
            self.time_series[index + self.seq_len: index + self.seq_len + self.y_size],
        )


# %%
test_size = 0.2
raw_data = pd.read_csv("../datasets/external/Alcohol_Sales.csv", parse_dates=["DATE"])
raw_data = raw_data["S4248SM144NCEN"].to_numpy()
raw_data = np.expand_dims(raw_data, axis=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
raw_data_scaled = scaler.fit_transform(raw_data)

simple_data_train = raw_data_scaled[: -int(test_size * len(raw_data))]
simple_data_val = raw_data_scaled[-int(test_size * len(raw_data)):]
train_data = TimeseriesDataset(simple_data_train, seq_len=7, y_size=1)
val_data = TimeseriesDataset(simple_data_val, seq_len=7, y_size=1)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)

# %%
class CNN_AE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # -> Inn chanels, out chanels, kernel size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 3),
            nn.ReLU(True),
            nn.Conv1d(8, 16, 5)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 5),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3)
        )
        #self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        out = self.encoder(x)
        out = self.decoder(out)
        out = out.view(
            out.shape[0],
            out.shape[2],
            out.shape[1]
            )
        return out 

    def training_step(self, batch: Tuple(Tensor), batch_idx: int):
        x, y = batch
        x_out = self.forward(x)
        loss = self.loss(x_out, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_out = self.forward(x)
        loss = self.loss(x_out, x)
        return loss

    def test_step(self, batch: Tuple(Tensor), batch_idx: int):
        x, y = batch
        x_out = self.forward(x)
        loss = self.loss(x_out, x)
        return loss

# %%
trainer = pl.Trainer(
    max_epochs=50,
)
model = CNN_AE()
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# %%
trainer.test(model, dataloaders=val_loader)


# %%
x_targets = []
train_predictions = []
for x_batch, y_batch in train_loader:
    y_pred = model(x_batch)
    train_predictions.extend(y_pred.detach().numpy().flatten())
    x_targets.extend(x_batch.flatten())

# train_predictions = trainer.predict(model, dataloaders=train_loader)[0].flatten()
plt.plot(x_targets)
plt.plot(train_predictions)


# %%
y_val_targets = []
y_val_predictions = []
for x_batch, y_batch in val_loader:
    x_pred = model(x_batch)
    print("X")
    print(x_pred)
    x_batch = x_batch[:, 0, 0]
    x_pred = x_pred[:, 0, 0]
    y_val_predictions.extend(x_batch.detach().numpy().flatten())
    y_val_targets.extend(x_pred.detach().numpy().flatten())

# predictions = trainer.predict(model, dataloaders=val_loader)[0]
# predictions = predictions.flatten()

plt.plot(y_val_targets, label="y_val_targets")
plt.plot(y_val_predictions)
plt.show()


###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################


# %%

data = pd.read_csv("../datasets/raw/market_insights_overview_5p_full.csv")
data["date"] = pd.to_datetime(data["date"])

test_size = 0.2
raw_data = pd.read_csv("../datasets/raw/market_insights_overview_5p_full.csv", parse_dates=["date"])
print(raw_data.info())
raw_data = raw_data["hits"].to_numpy()
raw_data = np.expand_dims(raw_data, axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
raw_data_scaled = scaler.fit_transform(raw_data)

simple_data_train = raw_data_scaled[: -int(test_size * len(raw_data))]
simple_data_val = raw_data_scaled[-int(test_size * len(raw_data)):]
train_data = TimeseriesDataset(simple_data_train, seq_len=7, y_size=1)
val_data = TimeseriesDataset(simple_data_val, seq_len=7, y_size=1)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)


# %%
trainer = pl.Trainer(
    max_epochs=30,
)
model = CNN_AE()
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# %%
trainer.test(model, dataloaders=train_loader)

# %%
x_targets = []
train_predictions = []
for x_batch, y_batch in train_loader:
    y_pred = model(x_batch)
    train_predictions.extend(y_pred.detach().numpy().flatten())
    x_targets.extend(x_batch.flatten())

# train_predictions = trainer.predict(model, dataloaders=train_loader)[0].flatten()
plt.plot(x_targets)
plt.plot(train_predictions)

# %%
y_val_targets = []
y_val_predictions = []
for x_batch, y_batch in val_loader:
    x_pred = model(x_batch)
    print("X")
    print(x_pred)
    x_batch = x_batch[:, 0, 0]
    x_pred = x_pred[:, 0, 0]
    y_val_predictions.extend(x_pred.detach().numpy().flatten())
    y_val_targets.extend(x_batch.detach().numpy().flatten())

# predictions = trainer.predict(model, dataloaders=val_loader)[0]
# predictions = predictions.flatten()

plt.plot(y_val_targets, label="y_val_targets")
plt.plot(y_val_predictions)
plt.show()

