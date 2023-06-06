import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_length, num_layers, batch_size):
    super(LSTM, self).__init__()

    # self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=hidden_size)
    self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers = num_layers, batch_first = True)
    self.classify = nn.Linear(hidden_dim, output_length*batch_size)
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

  def forward(self, x):
    self.lstm.flatten_parameters()
    self.lstm.to(self.device)
    _, (hidden, _) = self.lstm(x)
    
    out = hidden[-1]
    self.classify.to(self.device)
    out = self.classify(out) 
    out = torch.reshape(out, (self.batch_size, -1))
    return out

class RNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_length, num_layers, batch_size):
    super(RNN, self).__init__()

    # self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=hidden_size)
    self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers = num_layers, batch_first = True)
    self.classify = nn.Linear(hidden_dim, output_length*batch_size)
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

  def forward(self, x):
    self.rnn.flatten_parameters()
    self.rnn.to(self.device)
    _, hidden = self.rnn(x)
    
    out = hidden[-1]
    self.classify.to(self.device)
    out = self.classify(out) 
    out = torch.reshape(out, (self.batch_size, -1))
    return out
  
class GRU(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_length, num_layers, batch_size):
    super(GRU, self).__init__()

    # self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=hidden_size)
    self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers = num_layers, batch_first = True)
    self.classify = nn.Linear(hidden_dim, output_length*batch_size)
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

  def forward(self, x):
    self.gru.flatten_parameters()
    self.gru.to(self.device)
    _, hidden = self.gru(x)
    
    out = hidden[-1]
    self.classify.to(self.device)
    out = self.classify(out) 
    out = torch.reshape(out, (self.batch_size, -1))
    return out