
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
from grads import grads
from operators import operators
from pytorch_optim import SGD, AdaGrad, RMSProp, Adam


# Define model
def init_model(args):
    model = Net(784, 128, 10)
    criterion = nn.CrossEntropyLoss()
    
    if args.grad == 'grad':
        G = grads.grad
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

    if args.operator == "integer":
        OPT = operators(G)
        OPT = OPT.integer
    elif args.operator == "fractional":
        OPT = operators(G)
        OPT = operators.fractional(G)
    elif args.operator == "multi_fractional":
        OPT = operators(G)
        OPT = operators.multi_fractional(G)
    elif args.operator == "distributed_fractional":
        OPT = operators(G)
        OPT = operators.distributed_fractional(G)
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
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc3(out)
        # out = self.softmax(out)
        return out

# Load and preprocess the MNIST dataset
def load_mnist():
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = []
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(-1, 28*28)
            # labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            running_loss += loss.item()
        train_loss.append(running_loss/len(train_loader))
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        # validate_model(model, val_loader, criterion)
    return train_loss

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
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--grad', default='grad', choices=grad_funcs)
    parser.add_argument('--operator', default='integer', choices=opers)
    parser.add_argument('--optimizer', default='sgd', choices=optims)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_mnist()
    
    input_size = 28 * 28
    hidden_size = 256
    output_size = 10
    
    # model = Net(input_size, hidden_size, output_size)
    
    # criterion = nn.CrossEntropyLoss()
    optimizer, model, criterion = init_model(args)
    train_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)
    test_loss = evaluate_model(model, test_loader)

    # diaplying model train_loss
    # display_loss(train_loss)
    print(f'Total train loss is {train_loss}')
    print(f'Total test loss is {test_loss}')
    # diaplying model val_loss
    # display_loss(val_loss)
    
    
    
main()


# pca = PCA(10)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)

# print("Starting GD")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_int = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.optimizer(model.categorical_cross_entropy, model.w_flatten, model,lr=0.03, max_iter=10)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_int.append(temp_loss)
#         print(f"GD, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_GD")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# sns.lineplot(hist_int, label="int")
# plt.savefig("Int_Loss.png", dpi=500)
# plt.close()

# with open('hist_int.pkl', 'wb') as f:
#     pickle.dump(hist_int, f)

# print("Starting Fractional")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_frac = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha=0.9, max_iter=10)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_frac.append(temp_loss)
#         print(f"Fractional, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")

#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Frac")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# sns.lineplot(hist_int, label="int")
# sns.lineplot(hist_frac, label="frac")
# plt.savefig("Int_Frac.png", dpi=500)
# plt.close()

# with open('hist_frac.pkl', 'wb') as f:
#     pickle.dump(hist_frac, f)

# print("Starting Multi Fractional")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_multi = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.multi_frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha1=0.9, alpha2=1.1, max_iter=10)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_multi.append(temp_loss)
#         print(f"Multi, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Frac_Multi")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# sns.lineplot(hist_int, label="int")
# sns.lineplot(hist_frac, label="frac")
# sns.lineplot(hist_multi, label="multi")
# plt.savefig("Int_Frac_Multi.png", dpi=500)
# plt.close()

# with open('hist_multi.pkl', 'wb') as f:
#     pickle.dump(hist_multi, f)

# print("Starting Distribute Fractional")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_dist = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.dist_frac_optimizer(model.categorical_cross_entropy, model.w_flatten, model, lr=0.03, alpha1=0.9, alpha2=1.1, max_iter=10, N=10)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_dist.append(temp_loss)
#         print(f"Distribute, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Frac_dist")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# with open('hist_dist.pkl', 'wb') as f:
#     pickle.dump(hist_dist, f)

# sns.lineplot(data=hist_int, label="int")
# sns.lineplot(data=hist_frac, label="frac")
# sns.lineplot(data=hist_multi, label="multi")
# sns.lineplot(data=hist_dist, label="dist")
# plt.savefig("Int_Frac_Multi_Dist.png", dpi=500)
# plt.close()

# ##########################################

# print("Reimann_Liouville")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_RL = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.gen_frac_opt(model.categorical_cross_entropy, model.w_flatten, model, D=opt.Reimann_Liouville)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_RL.append(temp_loss)
#         print(f"Reimann_Liouville, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Reimann_Liouville")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# with open('hist_Reimann_Liouville.pkl', 'wb') as f:
#     pickle.dump(hist_RL, f)

# sns.lineplot(data=hist_int, label="int")
# sns.lineplot(data=hist_frac, label="frac")
# sns.lineplot(data=hist_multi, label="multi")
# sns.lineplot(data=hist_dist, label="dist")
# sns.lineplot(data=hist_RL, label="RL")
# plt.savefig("Int_Frac_Multi_Dist_RL.png", dpi=500)
# plt.close()

# print("Caputo")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_Caputo = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.gen_frac_opt(model.categorical_cross_entropy, model.w_flatten, model, D=opt.Caputo)
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_Caputo.append(temp_loss)
#         print(f"Caputo, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Caputo")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# with open('hist_Caputo', 'wb') as f:
#     pickle.dump(hist_Caputo, f)

# sns.lineplot(data=hist_int, label="int")
# sns.lineplot(data=hist_frac, label="frac")
# sns.lineplot(data=hist_multi, label="multi")
# sns.lineplot(data=hist_dist, label="dist")
# sns.lineplot(data=hist_RL, label="RL")
# sns.lineplot(data=hist_Caputo, label="Cap")
# plt.savefig("Int_Frac_Multi_Dist_RL_Cap.png", dpi=500)
# plt.close()


# print("Reimann_Liouville_GLR")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_RL_GLR = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.gen_frac_opt(model.categorical_cross_entropy, model.w_flatten, model, D=opt.Reimann_Liouville, lr=opt.Glearning_rate(model.w_flatten))
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_RL_GLR.append(temp_loss)
#         print(f"Reimann_Liouville_GLR, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Reimann_Liouville_GLR")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# with open('hist_Reimann_Liouville_GLR.pkl', 'wb') as f:
#     pickle.dump(hist_RL_GLR, f)

# sns.lineplot(data=hist_int, label="int")
# sns.lineplot(data=hist_frac, label="frac")
# sns.lineplot(data=hist_multi, label="multi")
# sns.lineplot(data=hist_dist, label="dist")
# sns.lineplot(data=hist_RL, label="RL")
# sns.lineplot(data=hist_Caputo, label="Cap")
# sns.lineplot(data=hist_Caputo, label="RL_GLR")
# plt.savefig("Int_Frac_Multi_Dist_RL_Cap_RLGLR.png", dpi=500)
# plt.close()


# print("Caputo_GLR")
# s=time.time()
# epoch = 5
# model = Net(x_train, y_train, [10, 10], batch_size=64)
# hist_Caputo_GLR = []
# for ep in range(epoch):
#     print("EPOCH: ", ep)
#     for b in range(int(x_test.shape[0]/model.batch_size)):
#         opt.gen_frac_opt(model.categorical_cross_entropy, model.w_flatten, model, D=opt.Caputo, lr=opt.Glearning_rate(model.w_flatten))
#         temp_loss = model.categorical_cross_entropy(model.w_flatten)
#         hist_Caputo_GLR.append(temp_loss)
#         print(f"Caputo_GLR, EPOCH: {ep}, Batch step: {b}, Loss: {temp_loss}")
        
#         model.batch_counter += model.batch_size

# d=time.time()
# model.save_model(name="model_Caputo_GLR")
# print(f"time is: {d-s}")
# print(model.eval(w=model.w_flatten, x=x_test, y=y_test))

# with open('hist_Caputo_GLR.pkl', 'wb') as f:
#     pickle.dump(hist_Caputo_GLR, f)


# sns.lineplot(data=hist_int, label="int")
# sns.lineplot(data=hist_frac, label="frac")
# sns.lineplot(data=hist_multi, label="multi")
# sns.lineplot(data=hist_dist, label="dist")
# sns.lineplot(data=hist_RL, label="RL")
# sns.lineplot(data=hist_Caputo, label="Cap")
# sns.lineplot(data=hist_Caputo, label="RL_GLR")
# sns.lineplot(data=hist_Caputo, label="RL_GLR")
# plt.savefig("Int_Frac_Multi_Dist_RL_Cap_RLGLR_CapGLR.png", dpi=500)
# plt.close()