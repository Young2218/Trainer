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
    
    def get_value(self):
        return self.avg
    
    def __str__(self):
        return f"Accuracy: [{self.avg}]"
    