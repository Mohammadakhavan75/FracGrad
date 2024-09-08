import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from tqdm import tqdm

from grads import grads
from models.resnet import ResNet18
from operators import operators
from pytorch_optim import SGD, AdaGrad, RMSProp, Adam
from torch.optim import SGD as psgd



# Define model
def init_model(args):
    if args.model == 'fc1':
        model = Net(3072, 128, 10)
    if args.model == 'resnet18':
        model = ResNet18(10)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    
    if args.grad == 'grad':
        G = torch.autograd.grad
    elif args.grad == 'Ggamma':
        G = grads.Ggamma
    elif args.grad == 'Glearning_rate':
        G = grads.Glearning_rate
    elif args.grad == 'Reimann_Liouville':
        G = grads.Reimann_Liouville
    elif args.grad == 'Caputo':
        G = grads.Caputo
    elif args.grad == 'Reimann_Liouville_fromG':
        G = grads.Reimann_Liouville_fromG
    elif args.grad == 'Caputo_fromG':
        G = grads.Caputo_fromG
    else:
        raise ValueError(f"Unknown gradient function: {args.grad}")

    OPT = operators(G, alpha1=args.alphas[0], alpha2=args.alphas[1])

    if args.operator == "integer":
        OPT = None
    elif args.operator == "fractional":
        OPT = OPT.fractional
    elif args.operator == "multi_fractional":
        OPT = OPT.multi_fractional
    elif args.operator == "distributed_fractional":
        OPT = OPT.distributed_fractional
    else:
        raise ValueError(f"Unknown operator: {args.operator}")

    if args.optimizer == "sgd":
        OPTIM = SGD(model.parameters(), OPT, lr=args.lr)
    elif args.optimizer == "adagrad":
        OPTIM = AdaGrad(model.parameters(), OPT, lr=args.lr)
    elif args.optimizer == "rmsprop":
        OPTIM = RMSProp(model.parameters(), OPT, lr=args.lr)
    elif args.optimizer == "adam":
        OPTIM = Adam(model.parameters(), OPT, lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return OPTIM, model, criterion

# Define the neural network model
class Net(nn.Module):
    def __init__(self, layer_sizes):
        super(Net, self).__init__()
        # Validate that the list has at least two elements
        assert len(layer_sizes) >= 2, "The layer_sizes list must contain at least input and output layer sizes."
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Don't add ReLU after the last layer
                layers.append(nn.ReLU())

        # Use nn.Sequential to stack all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


# Load and preprocess the MNIST dataset
# def load_mnist():
#     # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
#     test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    
#     train_size = int(0.8 * len(train_dataset))
#     val_size = len(train_dataset) - train_size
#     train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
#     train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
#     return train_loader, val_loader, test_loader


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = CIFAR10(root='D:/Datasets/data', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='D:/Datasets/data', train=False, transform=transform, download=True)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Train the model

def create_save_path(args, epochs):
    save_path = f'./run/exp_cifar10_{args.model}_{args.lr}_{args.optimizer}_epochs_{epochs}_{args.operator}_alpha1_{args.alphas[0]}_alpha2_{args.alphas[1]}/'
    model_save_path = os.path.join(save_path, 'models')
    os.makedirs(model_save_path, exist_ok=True)
    return save_path, model_save_path

def train(model, train_loader, criterion, optimizer, args):
    model.train()
    train_loss = []
    batch_time = []
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        s = time.time()
        loss.backward(create_graph=True)
        optimizer.step()
        e = time.time()
        batch_time.append(e-s)
        train_loss.append(loss.item())

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return np.mean(train_loss), accuracy, np.mean(batch_time), np.sum(batch_time), batch_time, train_loss

def train_loop(model, train_loader, val_loader, criterion, optimizer, args, epochs=5):
    pickle_saver = {}
    save_path, model_save_path = create_save_path(args, epochs)

    for epoch in range(epochs):
        mean_train_loss, accuracy, mean_batch_time, epoch_optimization_time, batch_time, train_loss = train(
            model, train_loader, criterion, optimizer, args
        )

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.16f}, Accuracy: {accuracy:.2f}%, '
              f'batch mean time: {mean_batch_time}, epoch optimization time: {epoch_optimization_time}')

        pickle_saver[epoch+1] = {
            'maen_batch_time': mean_batch_time,
            'mean_train_loss': mean_train_loss,
            'accuracy': accuracy,
            'batch_time': batch_time,
            'train_loss': train_loss
        }

    save_results(pickle_saver, filename=os.path.join(save_path, 'training_results.pkl'))

def save_results(results, filename):
    with open(filename, 'wb') as file:
        pickle.dump(results, file)



# Validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, 28*28)
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
    
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

# Evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    te_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            te_loss += loss.item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return te_loss/len(test_loader)


def display_loss(train_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', linestyle='-', color='b')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(range(1, len(train_loss) + 1))  # To show all epochs on x-axis
    plt.show()


    
def main():
       
    grad_funcs = ['grad', 'Ggamma', 'Glearning_rate', 'Reimann_Liouville', 'Caputo', 'Reimann_Liouville_fromG', 'Caputo_fromG']
    opers = ['integer', 'fractional', 'multi_fractional', 'distributed_fractional']
    optims = ['sgd', 'adagrad', 'rmsprop', 'adam']
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds',default=MNIST, choices= datasets)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--grad', default='grad', choices=grad_funcs)
    parser.add_argument('--operator', default='multi_fractional', choices=opers)
    parser.add_argument('--optimizer', default='sgd', choices=optims)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model', default='fc1')
    parser.add_argument('--alphas', type=str, default="[0.9, 1.1]")
    args = parser.parse_args()

    args.alphas = [int(x) for x in eval(args.alphas)]
    my_seed = 1
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)
    train_loader, val_loader, test_loader = load_cifar10()
    
    input_size = 32 * 32 * 3  # CIFAR-10 image size (32x32) with 3 color channels
    hidden_size = 256
    output_size = 10
    optimizer, model, criterion = init_model(args)
    train_loss = train_loop(model, train_loader, val_loader, criterion, optimizer, args, epochs=50)
    
    
main()