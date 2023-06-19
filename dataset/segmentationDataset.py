import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, img_size, mask_threshold = 200, transforms=None, feature_extractor=None):
        self.feature_extractor = feature_extractor
        
        self.images = images
        self.masks = masks
        self.feature_extractor = feature_extractor
        self.mask_threshold = mask_threshold
        
        if transforms is None:
            transforms = A.Compose([
                A.Resize(img_size, img_size)
            ])
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path)
        
        if self.masks is not None:
            mask_path = self.masks[index]
            mask = cv2.imread(mask_path, 0)
            mask = (mask > self.mask_threshold)*1.0
        else:
            mask = np.zeros((image.shape[0], image[1], 1))
        
       
        transformed = self.transforms(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        transformed_mask = torch.from_numpy(transformed_mask)
        transformed_mask = torch.unsqueeze(transformed_mask,0)
        
        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(transformed_image, return_tensors="pt")
            
            for k,v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
            transformed_image =  encoded_inputs['pixel_values']
        else:
            transformed_image = ToTensorV2(transformed_image)
            transformed_mask = ToTensorV2(transformed_mask)
        
        if self.masks is not None:
            return transformed_image, transformed_mask
        else:
            return transformed_image
        
        
    def __len__(self):
        return len(self.images)