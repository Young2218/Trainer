from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output
        
    def __getitem__(self, index):
        emb = self.input[index]
        gt = self.output[index]
        
        return emb, gt
    
    def __len__(self):
        return len(self.input)