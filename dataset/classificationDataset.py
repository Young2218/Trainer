from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch

class ClassificationDataset(Dataset):
    def __init__(self, img_paths, labels, resize, transforms=None, feature_extractor=None):
        self.img_paths = img_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        
        if transforms is None and feature_extractor is None:
            self.transforms = A.Compose([
                A.Resize(resize, resize),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ])
        elif transforms is None:
            self.transforms = A.Compose([
                A.Resize(resize, resize),
                A.HorizontalFlip(p=0.5),
                A.Normalize()
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        if image is None:
            print(img_path)
        
        image = self.transforms(image=image)['image']
        
        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(image, return_tensors="pt")
            image = encoded_inputs['pixel_values'].squeeze_()
                
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_paths)