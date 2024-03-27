import torch
from dataset import CustomDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd


data = pd.read_csv("<path_to_your_test_csv>/test_data1.csv")
img_dir = "<path_to_your_test_image_directory>"

transform = transforms.Compose([
    transforms.Resize((720, 720)),
    transforms.ToTensor(),
])

test_dataset = CustomDataset(data, img_dir=img_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = SimpleCNN(num_classes=40)
model.load_state_dict(torch.load("<path_to_your_model>/model.pth"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        images, labels = batch['image'], batch['label']
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')
