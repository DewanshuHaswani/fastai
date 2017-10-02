import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

BATCH_SIZE=64
LR=0.01
MOMENTUM=0.5
EPOCHS = 5

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

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data", 
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE
    )

# The neural network
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        
        # Create a convolutional layer, 1 input and 10 outputs
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # Randomly zeroes whole channels of the tensor
        self.conv2_drop = nn.Dropout2d()

        # Linear layer with 320 inputs and 50 outputs
        # We choose 320 because we reshape it with the 'view' function later
        self.fc1 = nn.Linear(320,  50)

        # Another fully connected layer
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Relu activation function with max pooling layer with kernel size 2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Relu activation function with max pooling after dropout
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Returns a new tensor with the dimensions of [x, 320]
        # Where x is the original dimension / 320
        x = x.view(-1, 320)

        # Activate on the FC layer
        x = F.relu(self.fc1(x))

        # Dropout
        x = F.dropout(x, training=self.training)

        # A fully connected layer
        x = self.fc2(x)

        # log(softmax)
        return F.log_softmax(x)

# The model instance
model = Net()

# A Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

# Train the model
def train(epoch):
    # Set the model to training mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to variables (tensors)
        data, target = Variable(data), Variable(target)

        # Resets the gradient
        optimizer.zero_grad()

        output = model(data)

        # Negative log likelihood loss
        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    # Set model to evaluation mode
    model.eval()

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # Sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]

        # Get the index of the max log probability
        pred = output.data.max(1, keepdim=True)[1]

        # Element wize equality
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test()
