# %%
from ast import Tuple
from pytorch_lightning.loggers import NeptuneLogger
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, _test_serialization_subcmul
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# %%

class LSTM(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        learning_rate: float,
        batch_first: bool = True,
        dropout: float = 0.2,
        bidirectional: bool = False,
        optimizer_name: str = "Adam",
        #batch_size: int = 32,
    ):
        super().__init__()

        self.output_size = output_size  # shape of output
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # number of features in each sample
        self.hidden_size = hidden_size  # hidden state
        self.learning_rate = learning_rate  # learning rate
        self.dropout = nn.Dropout(p=dropout)
        #self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fully_conencted_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size * self.input_size
        )

    def reset_hidden_state(self, batch_size):

        self.h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        self.c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x):
        self.reset_hidden_state(batch_size=x.size(0))
        # Here you have defined the hidden state, and internal state first, initialized with zeros.

        # Propagate input through LSTM

        # output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
        # h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        # c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        # print("output", ula.shape)
        # Choose the hidden state from the last layer
        last_hidden_state_layer = h_out[-1]
        # h_out = h_out.view(-1, self.hidden_size)
        # out_squashed = ula.view(-1, self.hidden_size)

        out = self.fully_conencted_layer(last_hidden_state_layer)
        # out = self.fc(ula)
        # out = self.dropout(out)

        out_multi_feature = out.reshape(out.size(0), self.output_size, self.input_size)
        # return out
        return out_multi_feature

    def training_step(self, train_batch: Tuple(Tensor), batch_idx: int):
        x, y, scaler = train_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, scaler = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch: Tuple(Tensor), batch_idx: int):
        x, y, scaler = test_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def training_epoch_end(self, training_step_outputs: List[Dict]):
        avg_train_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)

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
            self.time_series[index : index + self.seq_len],
            self.time_series[index + self.seq_len : index + self.seq_len + self.y_size],
        )
class GlobalTimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, time_series: List[np.ndarray], seq_len: int, y_size: int, scalers):
        self.scalers = scalers
        time_series_combined = np.concatenate(time_series, axis=0)

        self.time_series = torch.tensor(time_series_combined).float()
        self.seq_len = seq_len
        self.y_size = y_size

    def __len__(self):
        return self.time_series.__len__() - (self.seq_len + self.y_size - 1)

    def __getitem__(self, index):
        # return x, y
        return (
            self.time_series[index : index + self.seq_len],
            self.time_series[index + self.seq_len : index + self.seq_len + self.y_size],
            self.scalers
        )

# %%

raw_data = pd.read_csv(
    "../datasets/interim/market_insight_preprocessed.csv",
    parse_dates=["date"],
)
raw_data.head()

# %%
input_window_size = 10
output_window_size = 7
raw_data.groupby("cat_id").count()
# id11573 has mose data with 902 entries
# data_chosen_cat = raw_data.loc[raw_data["cat_id"] == 11573][["hits","clicks", "date"]]
def process_data_for_cat_id(cat_ids: List[str]):
  datasets = []
  train_datasets = []
  val_datasets = []
  scalers = []
  for cat_id in cat_ids:
    data_chosen_cat = raw_data.loc[raw_data["cat_id"] == cat_id][["hits", "date"]]
    data_chosen_cat_filled_dates = data_chosen_cat.groupby(pd.Grouper(key="date", freq="D")).sum()

    dates = data_chosen_cat_filled_dates.index.tolist()

    data = np.array(data_chosen_cat_filled_dates)

    val_data_split_index = len(data) - output_window_size - input_window_size
    print("split index", val_data_split_index)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    data_scaled = scaler.fit_transform(data)

    data_train = data_scaled[:val_data_split_index]
    data_val = data_scaled[val_data_split_index:]
    print("data_train shape: ", data_train.shape)
    print("data_val shape: ", data_val.shape)
    train_datasets.append(data_train)
    val_datasets.append(data_val)
    scalers.append(scaler)
    
    train_data = TimeseriesDataset(data_train, seq_len=input_window_size, y_size=output_window_size)
    val_data = TimeseriesDataset(data_val, seq_len=input_window_size, y_size=output_window_size)
    print("length val data", len(data_val))
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    #return train_loader, val_loader

  combined_train_datasets = GlobalTimeseriesDataset(train_datasets, seq_len=input_window_size, y_size=output_window_size, scalers=scalers)
  combined_val_datasets = GlobalTimeseriesDataset(val_datasets, seq_len=input_window_size, y_size=output_window_size, scalers=scalers)

  print("combined_train_datasets", len(combined_train_datasets))
  train_loader = DataLoader(dataset=combined_train_datasets, batch_size=32, shuffle=False)
  val_loader= DataLoader(dataset=combined_val_datasets, batch_size=1, shuffle=False)

  return train_loader, val_loader

# %%
cat_ids = [11091, 11573]
train_loader, val_loader = process_data_for_cat_id(cat_ids)
# %%
def use_simple_data():
    raw_data = pd.read_csv("../datasets/external/Alcohol_Sales.csv", parse_dates=["DATE"])
    raw_data = raw_data["S4248SM144NCEN"].to_numpy()
    raw_data = np.expand_dims(raw_data, axis=1)

    input_window_size = 10
    output_window_size = 7
    scaler = MinMaxScaler(feature_range=(-1, 1))
    raw_data_scaled = scaler.fit_transform(raw_data)

    print("RAW_DATA SHAPE", raw_data.shape)
    #simple_data_train = raw_data_scaled[: -int(test_size * len(raw_data))]
    #simple_data_val = raw_data_scaled[-int(test_size * len(raw_data)) :]
    test_data_split_index = len(raw_data) - output_window_size - input_window_size
    print("split index", test_data_split_index)

    simple_data_train = raw_data_scaled[ : test_data_split_index]
    simple_data_val = raw_data_scaled[(test_data_split_index):]

    print("Train data shape: ", simple_data_train.shape)
    print("test_data shape: ", simple_data_val.shape)
    train_data = TimeseriesDataset(simple_data_train, seq_len=10, y_size=7)
    val_data = TimeseriesDataset(simple_data_val, seq_len=10, y_size=7)
    print("length val data", len(val_data))
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    return train_loader, val_loader
#train_loader, val_loader = use_simple_data()

# %%
trainer = pl.Trainer(
    max_epochs=10,
    #logger=neptune_logger,
    )
model = LSTM(
    input_size=1,
    hidden_size=20,
    output_size=7,
    num_layers=1,
    learning_rate=0.001,
    #batch_size=64,
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# %%
#trainer.validate(model, dataloaders=[dataset["train"] for dataset in datasets])
# %%
#trainer.test(model, dataloaders=[dataset["train"] for dataset in datasets])
# %%
y_val_targets = []
y_val_predictions = []
for x_batch, y_batch in val_loader:
    print("BATCH")
    y_pred = model(x_batch)
    print(y_pred.shape)
    print(y_batch.shape)
    y_val_predictions.extend(y_pred.detach().numpy().flatten())
    y_val_targets.extend(y_batch.flatten())

# predictions = trainer.predict(model, dataloaders=val_loader)[0]
# predictions = predictions.flatten()

plt.plot(y_val_targets, label="y_val_targets")
plt.plot(y_val_predictions)
plt.show()

# %%
x_targets = []
train_predictions = []
x_targets_first_step = []
train_predictions_first_step = []

for x_batch, y_batch in train_loader:
    y_pred = model(x_batch)
    print("y_pred", y_pred.shape)
    print("y_batch", y_batch.shape)
    y_pred_visualize = y_pred[::10, :, :].detach().numpy().flatten()
    y_pred_visualize_first_step = y_pred[:, 0, :].detach().numpy().flatten()

    y_target_visualize_first_step = y_batch[:,0, :].detach().numpy().flatten()
    y_target_visualize = y_batch[::10, :, :].detach().numpy().flatten()

    print("y_pred_visualize", y_pred_visualize.shape)
    train_predictions.extend(y_pred_visualize)
    x_targets.extend(y_target_visualize)
    train_predictions_first_step.extend(y_pred_visualize_first_step)
    x_targets_first_step.extend(y_target_visualize_first_step)

# train_predictions = trainer.predict(model, dataloaders=train_loader)[0].flatten()
plt.plot(x_targets)
plt.plot(train_predictions)
plt.show()
plt.plot(x_targets_first_step)
plt.plot(train_predictions_first_step)
