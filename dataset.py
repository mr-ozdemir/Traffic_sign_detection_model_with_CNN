from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.dataframe['class'] -= 1
        


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Existing code to load image and normalize bounding box coordinates
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['file'])
        image = Image.open(img_name).convert('RGB')

        xmin, xmax, ymin, ymax = self.dataframe.iloc[idx][['xmin', 'xmax', 'ymin', 'ymax']].values
        
        # Crop and resize the image
        image = image.crop((xmin, ymin, xmax, ymax))
     
        # Apply the transform after cropping
        if self.transform:
            image = self.transform(image)

        class_id = self.dataframe.iloc[idx]['class']  # This will now be correctly zero-indexed
        label = torch.tensor(class_id, dtype=torch.long)

        return {'image': image, 'label': label}
