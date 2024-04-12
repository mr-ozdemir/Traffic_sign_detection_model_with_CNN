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

def torch_accuracy_score(y_true, y_pred):
    correct_predictions = torch.eq(y_true, y_pred).sum().item()
    total_predictions = y_true.shape[0]
    return correct_predictions / total_predictions

def torch_precision_recall_f1_score(y_true, y_pred, num_classes):
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        true_positive = ((y_pred == cls) & (y_true == cls)).sum().item()
        false_positive = ((y_pred == cls) & (y_true != cls)).sum().item()
        false_negative = ((y_pred != cls) & (y_true == cls)).sum().item()
        
        precision[cls] = true_positive / (true_positive + false_positive + 1e-8)
        recall[cls] = true_positive / (true_positive + false_negative + 1e-8)
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls] + 1e-8)
    
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    return macro_precision, macro_recall, macro_f1

def evaluate_model_pytorch(model, dataloader, device, num_classes):
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            true_labels.append(labels)
            predictions.append(predicted)
    
    true_labels = torch.cat(true_labels)
    predictions = torch.cat(predictions)
    
    accuracy = torch_accuracy_score(true_labels, predictions)
    precision, recall, f1 = torch_precision_recall_f1_score(true_labels, predictions, num_classes)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, precision, recall, f1
