import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
from model import CNNModel

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model = CNNModel().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/cnn_cifar10.pth",
        map_location=device
    )
)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)

print(f"\nAccuracy: {accuracy*100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n")
print(cm)

# Confusion Matrix Image
plt.figure(figsize=(10,8))
plt.imshow(cm)
plt.colorbar()

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("results/confusion_matrix.png")
plt.show()

print("\nConfusion Matrix Saved -> results/confusion_matrix.png")