import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import CustomDataset
from model import SimpleCNN
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR



data = pd.read_csv("<path_to_your_csv>")
img_dir = "<path_to_your_image_directory>"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomDataset(data, img_dir=img_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize the model

model = SimpleCNN(num_classes=41)  

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 



scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(dataloader, 0):
        inputs, labels = batch['image'], batch['label']
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch} - Loss: {running_loss}')
