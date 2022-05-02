import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transforms=None):
        self.dictionary = dictionary
        self.transforms = transforms

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        objs = self.dictionary[key]
        # open the file as a PIL image
        img = Image.open(key).convert("RGB")

        #create the target dict that is required from the tutorial
        num_objs = len(objs)
        
        boxes = []
        labels = []

        for i in range(num_objs):
            boxes.append([objs[i][0],objs[i][2],objs[i][1],objs[i][3]])
            labels.append(int(objs[i][-1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)  
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # apply the necessary transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target