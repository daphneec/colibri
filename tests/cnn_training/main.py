#!/usr/bin/env python3
# MAIN.py
#   by Lut99
#
# Created:
#   28 Nov 2023, 15:02:08
# Last edited:
#   28 Nov 2023, 16:50:03
# Auto updated?
#   Yes
#
# Description:
#   File to try our hand at handwriting recognition using pytorch.
# 
#   This file is a trial run for [CrypTen](https://crypten.ai/)'s unencrypted training functionality.
# 
#   Based on: https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
#

import argparse
import itertools
import typing
from datetime import datetime
from time import time

import crypten
import crypten.nn as cnn
# import crypten.optim as coptim
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms


##### HELPER FUNCTIONS #####
def assert_data(dataset_path: str) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
        Asserts that the handwriting dataset exists at the given location.

        # Arguments
        - `dataset_path`: The path to download the MNIST dataset to.
    """

    # Define the transform on all the sets (why? check the tutorial)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Get the data as two different sets
    trainset = datasets.MNIST(dataset_path, download=True, train=True, transform=transform)
    valset = datasets.MNIST(dataset_path, download=True, train=False, transform=transform)

    # Write loaders around them
    return (
        torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True),
        torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    )





##### MODELS #####
class DigitRecognizer(cnn.Module):
    """
        Network for doing digit recognition, dope.
    """

    layers: cnn.ModuleList

    def __init__(self, input_size: int, hidden_sizes: typing.List[int], output_size: int):
        super(DigitRecognizer, self).__init__()

        self.layers = cnn.ModuleList()
        for (i, (prev_size, next_size)) in enumerate(itertools.pairwise([input_size] + hidden_sizes + [output_size])):
            # Add the neural layer
            print(f"[DigitRecognizer] Adding cnn.Linear({prev_size}, {next_size})")
            self.layers.append(cnn.Linear(prev_size, next_size))
            # Add the ReLU if not the last
            if i < 2 + len(hidden_sizes) - 1 - 1:
                print(f"[DigitRecognizer] Adding cnn.ReLU()")
                self.layers.append(cnn.ReLU())
        print(f"[DigitRecognizer] Adding cnn.LogSoftmax(dim=1)")
        self.layers.append(cnn.LogSoftmax(dim=1))

    def forward(self, x):
        # Simply iterator over all layers
        for layer in self.layers:
            x = layer.forward(x)
        return x





##### ENTRYPOINT #####
def main(dataset_path: str) -> int:
    """
        Main function/entrypoint of the script.

        # Arguments
        - `dataset_path`: The path to download the MNIST dataset to.

        # Returns
        An exit code for the script, where `0` is OK and non-zero is bad.
    """

    # Assert the dataset exists
    print(f"Asserting data exists at '{dataset_path}'...")
    (train, val) = assert_data(dataset_path)

    # Build the model
    print("Building model...")
    model = DigitRecognizer(784, [128, 64], 10)

    # Build the optimizer
    print("Building optimizer...")
    # criterion = cnn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # optimizer = coptim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Train the model
    time0 = time()
    epochs = 15
    print(f"Training for {epochs} epochs")
    for e in range(epochs):
        running_loss = 0
        for images, labels in train:
            # Flatten the MNIST images
            images = images.view(images.shape[0], -1)

            # Forward pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f" > Epoch {e} - Training loss: {running_loss/len(train)}")
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    # A smashing success!
    return 0



# Actual entrypoint
if __name__ == "__main__":
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="/tmp/mnist", help="The path to store the MNIST dataset in.")

    # Parse the defined arguments
    args = parser.parse_args()

    # Call main
    exit(main(args.dataset))
