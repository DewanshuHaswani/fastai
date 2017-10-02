import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from random import choice

BATCH_SIZE=64

# Load the mnist dataset
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data", 
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE
    )

train_data = train_loader.dataset.train_data

char = choice(train_data)

print(char)

plt.imshow(char.numpy())
plt.show()

