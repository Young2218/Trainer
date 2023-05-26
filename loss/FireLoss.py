import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional

class AreaCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, input: Tensor, target: Tensor, area) -> Tensor:
        celoss = []
        for pr, gt, ar in zip(input, target, area):
            if ar == 0:
                m = 1
            else:
                m = -(ar) + 1.5
                 
            celoss.append(super().forward(pr, gt) * m)
            
        return sum(celoss)/len(celoss)
    
class RatioCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, class_ratio: dict = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.class_ratio = {}
        for k, v in class_ratio.items():
            self.class_ratio[k] = float(-v + 1.5)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        celoss = []
        for pr, gt in zip(input, target):
            celoss.append(super().forward(pr, gt) * self.class_ratio[gt.item()])
           
        return sum(celoss)/len(celoss)

class AreaRatioCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight: Optional[Tensor] = None, class_ratio: dict = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.class_ratio = {}
        for k, v in class_ratio.items():
            self.class_ratio[k] = float(-v + 1.5)
    
    def forward(self, input: Tensor, target: Tensor, area) -> Tensor:
        celoss = []
        for pr, gt, ar in zip(input, target, area):
            if ar == 0:
                m = 1
            else:
                m = -(ar) + 1.5
            celoss.append(super().forward(pr, gt) * self.class_ratio[gt.item()] * m)
           
        return sum(celoss)/len(celoss)