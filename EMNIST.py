import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_dataset = datasets.EMNIST(root = './data',train = True,download = True,split = 'byclass')
train_dataset.data = train_dataset.data.permute(0,2,1).flip(dims = (2,))
train_dataset.transform = transform
test_dataset = datasets.EMNIST(root = './data',train = False,download = True,split = 'byclass')
test_dataset.data = test_dataset.data.permute(0,2,1).flip(dims = (2,))
test_dataset.transform = transform

batch_size = 64
train_loader = DataLoader(train_dataset,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False)

print(f"训练集样本数：{len(train_dataset)}")
print(f"测试集样本数：{len(test_dataset)}")
print(f"类别数：{len(train_dataset.classes)}")



class EMNISTNet(nn.Module):
    def __init__(self,input_dim = 784,hidden1_dim = 512,hidden2_dim = 256,output_dim = 62):
        super(EMNISTNet,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden1_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_dim,hidden2_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_dim,output_dim)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMNISTNet().to(device)
print(f"模型运行设备：{device}")



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)

def train(model,train_loader,criterion,optimizer,epoch):
    model.train()
    train_loss = 0.0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch[{epoch+1}],Train Loss:{train_loss:.4f}")
    return train_loss

def test(model,test_loader,criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()*data.size(0)       ##
            pred = output.argmax(dim = 1,keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)*100
    print(f"Test Loss:{test_loss:.4f},Test Accuracy:{test_acc:.2f}%\n")
    return test_loss,test_acc



num_epochs = 20
train_losses = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    train_loss = train(model,train_loader,criterion,optimizer,epoch)
    train_losses.append(train_loss)
    test_loss,test_acc = test(model,test_loader,criterion)
    test_losses.append(test_loss)    
    test_accs.append(test_acc)
    print(f"最终测试准确率：{test_accs[-1]:.2f}%")