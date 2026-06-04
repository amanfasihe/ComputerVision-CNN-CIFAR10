import os
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

transform = transforms.ToTensor()

dataset = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    download=False,
    transform=transform
)

classes = (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

count = 0

plt.figure(figsize=(12, 6))

for image, label in dataset:

    # Cat = 3, Dog = 5
    if label == 3 or label == 5:

        count += 1

        plt.subplot(2, 5, count)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(classes[label])
        plt.axis("off")

        if count == 10:
            break

plt.suptitle("Sample Cat and Dog Images from CIFAR-10")

plt.tight_layout()

plt.savefig(
    "results/cat_dog_samples.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Image Saved -> results/cat_dog_samples.png"
)

plt.show()