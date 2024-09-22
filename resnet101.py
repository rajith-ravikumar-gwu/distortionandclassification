from torchvision import models
from torchvision.models.resnet import ResNet101_Weights
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

# Load the image
img = Image.open(r'C:\Users\rajit\Pictures\goldentank_0.5.jpg')
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Load the pre-trained ResNet101 model
resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
resnet.eval()

# Check if a GPU is available and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)
batch_t = batch_t.to(device)

# Get the model output
with torch.no_grad():
    out = resnet(batch_t)

# Get the top predictions
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Print the top 10 predictions
for idx in indices[0][:10]:
    label = labels[int(idx.item())]
    print(f"{label}: {percentage[idx].item():.2f}%")