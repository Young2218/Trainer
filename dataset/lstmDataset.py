from torch.utils.data import Dataset
import torch
import numpy as np

class LSTMDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output
        
    def __getitem__(self, index):
        emb = self.input[index]
        gt = self.output[index]
        
        emb = np.array(emb)
        gt = np.array(gt)
        
        emb = torch.from_numpy(emb).float()
        gt = torch.from_numpy(gt).float()
        
        return emb, gt
    
    def __len__(self):
        return len(self.input)