# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fastprogress import master_bar, progress_bar
from pandas import DataFrame
from torch.autograd import Variable


# %%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.output_size = 2  # number of classes
        self.num_layers = 1  # number of layers
        self.input_size = 1  # input size of baches?
        self.hidden_size = 64  # hidden state
        # self.seq_length = 20  # sequence length
        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
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


# %%
# Generate 100 examples of sine waves. Each sine wave consists of 1000 points
N = 100  # number of samples
L = 1000  # length of each sample (number of values for each sine wave)
T = 20  # width of the wave
x = np.empty((N, L), np.float32)  # instantiate empty array
x[:] = np.arange(L) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
y = np.sin(x / 1.0 / T).astype(np.float32)
plt.plot(x[0], y[0])

# %%
# Split into train data and test data
train_data = y[0][: int(0.8 * L)]
test_data = y[0][int(0.8 * L) :]
plt.plot(train_data)
plt.plot(test_data)

# %%
def sliding_window_generate_tuples(input_data, window_size, output_size):
    x = []
    y = []
    for i in range(0, len(input_data) - window_size -1):
        _x = input_data[i : i + window_size]
        _y = input_data[i + window_size : i + window_size + output_size]

        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

# %%

window_size = 2
output_size = 1

x, y = sliding_window_generate_tuples(train_data, window_size=window_size, output_size=output_size)
x = Variable(torch.Tensor(x))
y = Variable(torch.Tensor(y))
x_tensor = torch.reshape(x, (x.shape[0], x.shape[1], 1))
y_tensor = torch.reshape(y, (y.shape[0], y.shape[1]))
print(x_tensor.shape)
print(y.shape)
x = x_tensor
y = y_tensor

# %%
train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
print("train shape is:",trainX.size())
print("train label shape is:",trainY.size())
print("test shape is:",testX.size())
print("test label shape is:",testY.size())


# %%
#####Init the Model #######################
lstm = LSTM()

print("hei")
learning_rate = 0.001

##### Set Criterion Optimzer and scheduler ####################
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=1e-5)
# What is this
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
# Train the model

num_epochs = 500

training_outputs = []

for epoch in progress_bar(range(num_epochs)): 
    lstm.train()
    outputs = lstm(trainX)
    training_outputs.append(outputs)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    
    optimizer.step()
    
    #Evaluate on test     
    lstm.eval()
    valid = lstm(testX)
    vall_loss = criterion(valid, testY)
    scheduler.step(vall_loss)
    
    if epoch % 50 == 0:
      print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(),vall_loss.cpu().item()))


# %%
plt.plot(training_outputs[-1].detach().numpy())
len(training_outputs[3])
# %%
predict = lstm(testX)
print(predict.detach().numpy()[:, 1])
plt.plot(predict.detach().numpy()[:,0])
plt.plot(testY.detach().numpy()[:,0])
