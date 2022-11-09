#!/usr/bin/env python

from tap import Tap
from utils import run_train, load_dataset


class ArgumentParser(Tap):
    n_train: int  # size of the training set
    batch_size: int=64  # size of a batch
    n_epochs: int=10  # number of epochs to consider for training
    n_early_stop: int=3  # maximum number of consecutive epochs to consider for early stopping
    save_losses: bool=True  # whether or not to save the validation and training set losses


def train():
    args = ArgumentParser().parse_args()
    n_train = args.n_train
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_early_stop = args.n_early_stop
    save_losses = args.save_losses

    print(f"\nCode launched with these arguments {args}")

    train_loader, val_loader = load_dataset(n_train, batch_size)
    train_losses, val_losses = run_train(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        n_train=n_train,
        n_early_stop=n_early_stop,
        save_losses=save_losses,
    )
    return train_losses, val_losses


if __name__ == "__main__":
    results = train()
