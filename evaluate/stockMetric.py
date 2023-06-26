import torch

class AccuracyMeter():
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, pred, gt):
        self.sum += (pred == gt).sum().item()
        self.count += len(gt)
        self.avg = self.sum / self.count
    
    def compute(self):
        return self.avg
    
    def __str__(self):
        return f"Accuracy: [{self.avg}]"

class StockMeter():
    def __init__(self):
        self.pred_money = 1
        self.real_money = 1
    
    def reset(self):
        self.pred_money = 1
        self.real_money = 1
    
    def update(self, pred, gt):
        argmax_dim = torch.max(pred, dim = 1)
        
        for p in argmax_dim[0]:
            self.pred_money *= (1+p)
        for e, i in enumerate(argmax_dim[1]):
            self.real_money *= (1+gt[e][i])
    
    def compute(self):
        return  self.real_money.item()
    
    def __str__(self):
        return f": p[{self.pred_money}], r[{self.real_money}]"
    

class StocAverageRevenuekMeter():
    def __init__(self, topk=3):
        self.sum = 0.
        self.count = 0.
        self.topk = topk
    
    def reset(self):
        self.sum = 0.
        self.count = 0.
    
    def update(self, pred, gt):
        values, indexs = torch.topk(pred,  self.topk, dim = 1)
        
        for e, i in enumerate(indexs):
            self.sum += gt[e][i].sum()
            self.count += self.topk
    
    def compute(self):
        return  self.sum / self.count
    
    def __str__(self):
        return f": AVG{self.get_value()}"

