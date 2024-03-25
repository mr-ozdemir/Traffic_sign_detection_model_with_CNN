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
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['file'])
        image = Image.open(img_name).convert('RGB')
        original_size = image.size
        class_id = self.dataframe.iloc[idx]['class']
        label = torch.tensor(class_id, dtype=torch.long)

        xmin, xmax, ymin, ymax = self.dataframe.iloc[idx][['xmin', 'xmax', 'ymin', 'ymax']]
        scale_width = 720 / original_size[0]
        scale_height = 720 / original_size[1]
        xmin_norm = min(xmin * scale_width, 720)
        xmax_norm = min(xmax * scale_width, 720)
        ymin_norm = min(ymin * scale_height, 720)
        ymax_norm = min(ymax * scale_height, 720)
        bbox = torch.tensor([xmin_norm, xmax_norm, ymin_norm, ymax_norm], dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'bbox': bbox}
