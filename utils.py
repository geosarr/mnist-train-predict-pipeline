import os
from typing import Tuple
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from config import N_CLASSES, DATA_DIR, MODEL_DIR, RESULTS_DIR

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset(
    n_train: int, batch_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Builds the training and dataloaders
    Args:
        n_train: number of elements in the training set
        batch_size: number of elements in a training batch

    Returns:
        tuple of training and validation dataloaders

    """

    print("\n")
    print(" Loading training and validation sets ".center(100, "#"))
    
    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor()
    )

    print(f"Number of training instances {n_train}")
    print(f"Number of validation instances {len(train_set)-n_train}")

    print("\n")
    print(" Building dataloaders ".center(100, "#"))
    train_loader = torch.utils.data.DataLoader(
        [d for i, d in enumerate(train_set) if i < n_train],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        [d for i, d in enumerate(train_set) if i >= n_train],
        batch_size=len(train_set) - n_train,
        shuffle=True,
        num_workers=2,
    )
    print(" OK ".center(10, "*"))

    return train_loader, val_loader


def run_train(
    n_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    batch_size: int,
    n_train: int,
    n_early_stop: int,
    save_losses: bool,
) -> Tuple[np.array, np.array]:
    """
    Training the model
    Args:
        n_epochs: number of epochs
        train_loader: the training dataloader
        batch_size: number of elements in a training batch
        n_train: number of elements in the training set
        n_early_stop: minimal number of consecutive training times before early stopping
        save_losses: whether or not to save the training and validation losses plots

    Returns:
        training and validation losses per epoch

    """
    val_losses = []
    train_losses = []
    x_val, y_val = next(iter(val_loader))

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, N_CLASSES),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("\n")
    print(" INIT TRAINING MLP ".center(100, "#"))
    min_val_loss = np.inf
    n_stop = 0
    nb_epoch_eff = 0

    for epoch in range(n_epochs):
        if n_stop >= n_early_stop:
            print(
                f"\nStopped training due to early stopping, the model's "
                + f"performance did not improve after {n_early_stop} consecutive epochs."
            )
            break
        print(f"\n[Epoch {epoch + 1}]")
        train_loss = []
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # forward and backward pass on a batch of training set
            inputs, labels = data
            optimizer.zero_grad()  # puts the gradients of parameters to 0
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # tells to the optimizer that an optimization step is achieved
            train_loss.append(loss.item())
        train_losses.append(train_loss)
        val_loss = criterion(net(x_val), y_val).item()
        val_losses.append(val_loss)

        # For Early Stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            n_stop = 0
            save_path = os.path.join(MODEL_DIR, "mnist_nn.pth")
            torch.save(net, save_path)
            print(
                f"\nSaved the current best model to "
                + f"{os.path.abspath(save_path)} for [Epoch {epoch + 1}]"
            )
        else:
            n_stop += 1

        nb_epoch_eff += 1

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)

    if save_losses:
        plt.plot(range(1, nb_epoch_eff + 1), val_losses, label="validation set loss")
        plt.plot(
            range(1, nb_epoch_eff + 1),
            np.sum(batch_size * train_losses, axis=1) / n_train,
            label="training set loss",
        )
        plt.title("Cross Entropy Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(RESULTS_DIR, "losses.png")
        plt.savefig(save_path)
        print(f"Saved the losses to {os.path.abspath(save_path)}")

    return train_losses, val_losses


def run_predict() -> None:
    """
    Testing the model
    Args:
        None
    Returns:
        None
    """
    print("\n")
    print(" Loading the test set and the final model ".center(100, "#"))
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    n_test = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=4
    )
    best_net = torch.load(os.path.join(MODEL_DIR, "mnist_nn.pth"))
    print(" OK ".center(10, "*"))

    print("\n")
    print(" Predicting the test set ".center(100, "#"))

    predictions = [0] * n_test
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        x_test, __ = data
        outputs = best_net(x_test)
        predicted_digit = np.argmax(torch.softmax(outputs, 1).detach().numpy())
        predictions[i] = int(predicted_digit)

    print("\n")
    print(" Writing the predictions ".center(100, "#"))
    with open(
        os.path.join(RESULTS_DIR, "predictions.txt"), "w", encoding="utf8"
    ) as file:
        for i, pred in enumerate(predictions):
            if i <= len(predictions) - 2:
                file.write(f"{i+1},{pred}" + "\n")
            else:
                file.write(f"{i+1},{pred}")
    print(" OK ".center(10, "*"))
