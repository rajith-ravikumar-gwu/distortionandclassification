import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models.resnet import ResNet101_Weights
import requests

# Define paths
image_path1 = r'C:\Users\rajit\Pictures\tank.jpg'
image_path2 = r'C:\Users\rajit\Pictures\golden.jpg'
output_dir = r'C:\Users\rajit\Pictures\merged_images'
os.makedirs(output_dir, exist_ok=True)

# Define transparency levels
transparency_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Generate merged images with different transparency levels
for transparency in transparency_levels:
    output_path = os.path.join(output_dir, f'goldentank_{transparency:.1f}resnet.jpg')
    subprocess.run(['python', 'merge.py', image_path1, image_path2, str(transparency), output_path])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the pre-trained ResNet101 model
resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
resnet.eval()

# Check if a GPU is available and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Function to get top-1 accuracy
def get_top1_accuracy(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    
    with torch.no_grad():
        out = resnet(batch_t)
    
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    top1_idx = indices[0][0].item()
    top1_label = labels[top1_idx]
    top1_percentage = percentage[top1_idx].item()
    
    return top1_label, top1_percentage

# Analyze accuracy for different transparency levels
accuracies = []
for transparency in transparency_levels:
    image_path = os.path.join(output_dir, f'goldentank_{transparency:.1f}resnet.jpg')
    label, accuracy = get_top1_accuracy(image_path)
    accuracies.append(accuracy)
    print(f"Transparency {transparency:.1f}: {label} - {accuracy:.2f}%")

# Plot the accuracy levels
plt.figure(figsize=(10, 6))
plt.plot(transparency_levels, accuracies, marker='o')
plt.title('Top-1 Accuracy vs. Transparency Level')
plt.xlabel('Transparency Level')
plt.ylabel('Top-1 Accuracy (%)')
plt.grid(True)
plt.show()