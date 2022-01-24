import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F


def intro_tensor():
    """Examples of tensors operation"""
    v1 = torch.tensor(1)
    v2 = torch.tensor([1, 1])
    v3 = torch.tensor((3, 3))
    v4 = torch.tensor(3, 3, 3)

    print(f"Tensor 1:\n {v1}")
    print(f"Tensor 2:\n {v2}")
    print(f"Tensor 3:\n {v3}")
    print(f"Tensor 4:\n {v4}")


def simple_linear_regression():
    """Builds a simple linear regression with sampling data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Initializes two random tensor variables
    a = torch.randn(1, dtype=torch.float).to(device)
    b = torch.randn(1, dtype=torch.float).to(device)

    # Sets them as requiring gradients
    a.requires_grad_()
    b.requires_grad_()

    # Makes sampling data
    x_train = np.random.rand(100, 1)  # features
    y_train = 1 + 2 * x_train + 0.1 * np.random.randn(100, 1)  # calculated labels

    # Converts variables to tensor
    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)

    # Initializes the learning rates and number of epochs
    lr = 1e-1
    n_epochs = 20
    for epoch in range(n_epochs):
        # Note that a and b are the only 2 variables that we learn
        yhat = a + b * x_train_tensor  # prediction
        error = y_train_tensor - yhat  # difference
        loss = (error ** 2).mean()  # MSE
        loss.backward()  # gradient descent

        # Updates gradients - ball rolling down
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad

    # Reinitializes gradients for the next time train
    a.grad.zero_()
    b.grad.zero_()

    print("\nResults after updating:")
    print(f"Updated a: {a}")
    print(f"Updated b: {b}")


class Net(nn.Module):
    def __init__(self):
        """Constructors"""
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def fully_connected_network():
    """Build a fully connected layer on MNIST dataset"""
    # Loads dataset
    train = torchvision.datasets.MNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    test = torchvision.datasets.MNIST(
        "",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    # Initializes device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializes neural networks
    net = Net().to(device)

    # Initializes loss
    loss_criterion = nn.CrossEntropyLoss()

    # Initializes optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    # Trains
    for epoch in range(5):
        for data in trainset:  # go through each data
            X, y = data
            X = X.to(device)
            y = y.to(device)
            net.zero_grad()  # initialize the gradient at each batch to 0
            output = net(
                X.view(-1, 784)
            )  # view convert image to different shape and input to network
            loss = loss_criterion(output, y)
            loss.backward()  # Gradient Descent
            optimizer.step()  # update gradient
        print(loss)

    # Evaluates scores
    correct = 0
    total = 0

    with torch.no_grad():  # don't change the gradient
        for data in testset:
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output = net(X.view(-1, 784))

            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct / total, 2))


def main():
    # intro_tensor()
    # simple_linear_regression()
    fully_connected_network()


if __name__ == "__main__":
    main()
