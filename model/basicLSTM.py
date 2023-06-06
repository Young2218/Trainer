import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class BasicLSTM(nn.Module):
  def __init__(self, embedding_length, hidden_size, output_length, batch_size):
    super(BasicLSTM, self).__init__()

    self.lstm = nn.LSTM(input_size=embedding_length, hidden_size=hidden_size)
    self.classify = nn.Linear(hidden_size, output_length)
    
    self.batch_size = batch_size

  def forward(self, X):
    h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
    c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())

    output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
    
    return self.classify(final_hidden_state[-1]) 
