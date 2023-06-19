import torch
from torchmetrics import Dice

class DiceCoef():
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, pred, gt):
        pred = torch.nn.functional.sigmoid(pred)
        pred = pred > 0.5
     
        d = self.dice_coefficient(pred, gt)
        
        self.sum += d
        self.count += len(gt)
        self.avg = self.sum / self.count
    
    def get_value(self):
        return self.avg
    
    def __str__(self):
        return f"DiceCoef: [{self.avg}]"
    
    def dice_coefficient(self, mask1, mask2):
        intersect = torch.sum(mask1*mask2)
        fsum = torch.sum(mask1)
        ssum = torch.sum(mask2)
        dice = (2 * intersect ) / (fsum + ssum)
        dice = torch.mean(dice)
        return dice 

class meanIOU():
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, pred, gt):
        pred = torch.nn.functional.sigmoid(pred)
        pred = pred > 0.5
        
        self.sum += self.get_mean_IOU(pred, gt).item()
        self.count += len(gt)
        self.avg = self.sum / self.count
    
    def get_value(self):
        return self.avg
    
    def __str__(self):
        return f"mIOU: [{self.avg}]"
    
    def get_mean_IOU(self, mask1, mask2):
        intersect = torch.sum(mask1*mask2)
        union = torch.count_nonzero(mask1+mask2)
        IOU = intersect / union
        IOU = torch.mean(IOU)
        return IOU

    


