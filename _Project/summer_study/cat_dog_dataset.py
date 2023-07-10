from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, resize, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        
        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(resize, resize),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        if image is None:
            print(img_path)
        
        image = self.transforms(image=image)['image']
        
        
        if self.labels is not None: # train mode
            label = self.labels[index]
            return image, label
        else: # inference mode
            return image
        
    def __len__(self):
        return len(self.img_paths)