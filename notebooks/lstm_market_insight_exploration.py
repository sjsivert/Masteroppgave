# %%
import datetime
import os
import random
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastprogress import master_bar, progress_bar
from nptyping import NDArray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

# %%
#raw_data = pipeline.market_insight_pipeline().run()
raw_data = pd.read_csv(
  "../datasets/interim/market_insight_preprocessed.csv",
  parse_dates=["date"],
  )
raw_data.head()
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
raw_data.groupby("cat_id").count()
# id11573 has mose data with 902 entries
data_chosen_cat = raw_data.loc[raw_data["cat_id"] == 11573][["hits", "date"]]
data_chosen_cat_filled_dates = data_chosen_cat.groupby(pd.Grouper(key="date", freq="D")).sum()
dates = data_chosen_cat_filled_dates.index.tolist()
data_chosen_cat_filled_dates.tail()

# %%

data_chosen_cat_filled_dates.plot()
# %%
SEED = 1345
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

# %%
# Split into train data and validation data
data = np.array(data_chosen_cat_filled_dates)
test_size = 0.2
data_validation = data[-int(test_size * len(data)):]
data_train = data[:-int(test_size* len(data))]
data_train.shape
# %%
# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(data_train.reshape(-1, 1))
train_data_normalized.shape
# %%
window_size = 1
output_size = 1
def sliding_window_generate_tuples(input_data: np.ndarray, window_size: int, output_size: int) -> Tuple[ NDArray[(Any, window_size, 1)], NDArray[(Any, output_size, 1)] ]:
    x = []
    y = []
    for i in range(0, len(input_data) - window_size -1):
        _x = input_data[i : i + window_size]
        _y = input_data[i + window_size : i + window_size + output_size]

        x.append(_x)
        y.append(_y)


    # Reshape to remove the last dimension
    return np.array(x).reshape(len(x), len(x[0])), np.array(y).reshape(len(y), len(y[0]))
    #return np.array(x), np.array(y)
x, y = sliding_window_generate_tuples(train_data_normalized, window_size=window_size, output_size=output_size)
print(x.shape, y.shape)
print(x[:5])
print(y[:5])
# %%
# Split train and test data
X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.25, shuffle=False
)
# %%
class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])

# Wait, is this a CPU tensor now? Why? Where is .to(device)?

train_data = TimeseriesDataset(X_train, y_train)
test_data = TimeseriesDataset(X_val, y_val)
print(train_data[0])
print(train_data.__getitem__(0)[0].shape)
print(train_data.__getitem__(0)[1].shape)

# %%
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=False)
train_loader_no_batch = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
val_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
val_loader_no_batch = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
# %%
# Plot distribution
fig, axs = plt.subplots(2)
 
fig.suptitle('Data Distribution Before and After Normalization ',fontsize = 19)
pd.DataFrame(data_train).plot(kind='hist',ax = axs[0] , alpha=.4 , figsize=[12,6], legend = False,title = ' Before Normalization',color ='red') 
pd.DataFrame(train_data_normalized).plot(kind='hist', ax = axs[1] ,figsize=[12,6], alpha=.4 , legend = False,title = ' After Normalization'\
                                         ,color = 'blue')

# %%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.output_size = 1  # number of classes
        self.num_layers = 1  # number of layers
        self.input_size = 1  # input size of baches?
        self.hidden_size = 300  # hidden state
        self.learning_rate = 0.001  # learning rate
        self.dropout = nn.Dropout(p=0.2)

        self.criterion = torch.nn.MSELoss()

        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

    def forward(self, x):
        # Here you have defined the hidden state, and internal state first, initialized with zeros. 
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        out = self.dropout(out)
       
        return out

    # Builds function that performs a step in the train loop
    def train_step(self, x, y):
        # Sets model to TRAIN mode
        self.train()
        # Makes predictions
        yhat = self(x)
        # Computes loss
        loss = self.criterion(y, yhat)
 
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    

    def validation_step(self, x, y):
        val_losses = []
        with torch.no_grad():
            self.eval()

            yhat = self(x)
            val_loss = self.criterion(y, yhat)
            val_losses.append(val_loss.item())

        return val_losses

    def train_network(self, train_loader, val_loader, n_epochs=100, verbose=True):
        self.train()
        losses = []
        val_losses = []
        for epoch in progress_bar(range(n_epochs)):
            for x_batch, y_batch in train_loader:
                # the dataset "lives" in the CPU, so do our mini-batches
                # therefore, we need to send those mini-batches to the
                # device where the model "lives"
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                loss = self.train_step(x_batch, y_batch)
                losses.append(loss)
            
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                val_loss = self.validation_step(x_val, y_val)
                val_losses.append(val_loss)
                
            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss}. Validation losses: {val_loss}" )
        return losses, val_losses
# %%
# Train network
lstm = LSTM()
print(lstm)
losses, val_losses = lstm.train_network(train_loader, val_loader, n_epochs=500, verbose=True)

# %%
plt.plot(losses)
# %%
# Predict training
predictions = []
for x_batch, y_batch in train_loader_no_batch:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    yhat = lstm(x_batch)
    predictions.append(yhat.detach().cpu().numpy().flatten())
print(predictions[0].shape)
plt.plot(y_train)
plt.plot(predictions)

# %%
# Predict test
predictions_val = []
for x_batch, y_batch in val_loader_no_batch:
    x_batch = x_batch.to(device)
    yhat = lstm(x_batch)
    predictions_val.append(yhat.detach().cpu().numpy().flatten())
predictions_val_renormalize = scaler.inverse_transform(predictions_val)
y_val_renormalize = scaler.inverse_transform(y_val)
print(predictions_val_renormalize[:10])
plt.plot(y_val_renormalize)
plt.plot(predictions_val_renormalize)

# %%

writer = SummaryWriter(f'logs/net')
writer.add_graph(lstm, next(iter(train_loader))[0])
