from torchvision import models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
import torch
from torchvision import transforms
from PIL import Image
import requests

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

# Load the pre-trained EfficientNet-B0 model
efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet.eval()

# Check if a GPU is available and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet.to(device)

def get_top1_accuracy(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # Get the model output
    with torch.no_grad():
        out = efficientnet(batch_t)

    # Get the top predictions
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    top1_idx = indices[0][0].item()
    top1_label = labels[top1_idx]
    top1_percentage = percentage[top1_idx].item()

    return top1_label, top1_percentage

if __name__ == "__main__":
    image_path = r'C:\Users\rajit\Pictures\goldentank_0.5.jpg'
    label, accuracy = get_top1_accuracy(image_path)
    print(f"{label}: {accuracy:.2f}%")