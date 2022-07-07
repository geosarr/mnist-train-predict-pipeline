import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from config import N_CLASSES
import os


def load_dataset(n_train, batch_size):
    print('\n')
    print(" Loading training and validation sets ".center(100, '#'))
    
    os.makedirs("./data", exist_ok=True)
    train_set=datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    print(f"Number of training instances {n_train}")
    print(f"Number of validation instances {len(train_set)-n_train}")
   
    print('\n')
    print(" Building dataloaders ".center(100, '#'))
    train_loader = torch.utils.data.DataLoader([d for i, d in enumerate(train_set) if i<n_train], batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader([d for i, d in enumerate(train_set) if i>=n_train], batch_size=len(train_set)-n_train, shuffle=True, num_workers=2)
    print(" OK ".center(10, "*"))

    return train_loader, val_loader



def run_train(n_epochs, train_loader, val_loader, batch_size, n_train, n_early_stop,  save_losses):
    val_losses=[]
    train_losses=[]
    x_val, y_val=next(iter(val_loader))

    net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,32),
    nn.ReLU(),
    nn.Linear(32,N_CLASSES)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print('\n')
    print(" INIT TRAINING MLP ".center(100, "#")) 
    min_val_loss=np.inf
    n_stop=0
    nb_epoch_eff=0

    for epoch in range(n_epochs):
        if n_stop>=n_early_stop: 
            print(f"\nStopped training due to early stopping, the model's performance did not improve after {n_early_stop} consecutive epochs.")
            break 
        print(f"\n[Epoch {epoch + 1}]")
        tl=[]
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # forward and backward pass on a batch of training set
            inputs, labels = data
            optimizer.zero_grad() # puts the gradients of parameters to 0
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # tells to the optimizer that an optimization step is achieved
            tl.append(loss.item())
        train_losses.append(tl)   
        val_loss = criterion(net(x_val), y_val).item() 
        val_losses.append(val_loss)

        # For Early Stopping
        if val_loss < min_val_loss:
            min_val_loss=val_loss
            n_stop=0
            os.makedirs("./model", exist_ok=True)
            save_relative_path='./model/mnist_nn.pth'
            torch.save(net, save_relative_path)
            print(f"\nSaved the current best model to {os.path.abspath(save_relative_path)} for [Epoch {epoch + 1}]")
        else:
            n_stop+=1
    
        nb_epoch_eff+=1

    train_losses, val_losses = np.array(train_losses), np.array(val_losses)

    if save_losses:
        os.makedirs("./results", exist_ok=True)
        plt.plot(range(1, nb_epoch_eff+1), val_losses, label="validation set loss")
        plt.plot(range(1, nb_epoch_eff+1), np.sum(batch_size * train_losses, axis=1)/n_train, label="training set loss")
        plt.title('Cross Entropy Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        save_relative_path="./results/losses.png"
        plt.savefig(save_relative_path)
        print(f"Saved the losses to {os.path.abspath(save_relative_path)}")

    return train_losses, val_losses


def run_predict():

    print('\n')
    print(' Loading the test set and the final model '.center(100, "#"))
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    n_test = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    best_net = torch.load('./model/mnist_nn.pth')
    print(" OK ".center(10, "*"))

    print('\n')
    print(" Predicting the test set ".center(100, "#"))

    predictions=[0]*n_test
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)): 
        x_test, __ = data
        outputs=best_net(x_test)
        predicted_digit=np.argmax(torch.softmax(outputs,1).detach().numpy())
        predictions[i]=predicted_digit

    print('\n')
    print(" Writing the predictions ".center(100, '#'))
    with open("./results/predictions.txt", "w") as f:
        for i, pred in enumerate(predictions):
            if i<=len(predictions)-2:
                f.write(f"{i+1},{pred}"+'\n')
            else:
                f.write(f"{i+1},{pred}")
    print(" OK ".center(10, "*"))

    return predictions

