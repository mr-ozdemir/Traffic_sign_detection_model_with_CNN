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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Existing code to load image and normalize bounding box coordinates
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['file'])
        image = Image.open(img_name).convert('RGB')

        xmin, xmax, ymin, ymax = self.dataframe.iloc[idx][['xmin', 'xmax', 'ymin', 'ymax']].values
        original_size = image.size  # Original size (width, height)

        scale_width = 720 / original_size[0]
        scale_height = 720 / original_size[1]

        xmin_norm = int(xmin * scale_width)
        xmax_norm = int(xmax * scale_width)
        ymin_norm = int(ymin * scale_height)
        ymax_norm = int(ymax * scale_height)

        # Crop and resize the image
        image = image.crop((xmin_norm, ymin_norm, xmax_norm, ymax_norm))
        
        # Apply the transform after cropping
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none specified
            resize_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the cropped region to a fixed size
                transforms.ToTensor(),
            ])
            image = resize_transform(image)

        class_id = self.dataframe.iloc[idx]['class']
        label = torch.tensor(class_id, dtype=torch.long)

        return {'image': image, 'label': label}
