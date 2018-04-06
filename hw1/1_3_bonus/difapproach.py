import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import sys
import random
#Hyper Parameters
input_size = 784
hidden_size1 = 500
hidden_size2 = 500
num_classes = 10
batch_size = int(sys.argv[1])
num_epochs = min(batch_size*10,100)
learning_rate = 0.01
batch_num = 60000//batch_size

#get dataset
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              transform=transforms.ToTensor())

#load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           shuffle=True,
                                           num_workers=0,
                                           batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          shuffle=False,
                                          num_workers=0,
                                          batch_size=batch_size)

#NN Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()
net.cuda()
#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

inputs = []
labels = []
for i,(inp,lab) in enumerate(train_loader):
    inputs.append(Variable(inp.view(-1,input_size).cuda()))
    labels.append(lab.cuda())

final_grad = 0.0
#train
my_loss = 0.0
for epoch in range(num_epochs):
    correct = 0
    cnt = 0.0
    total = 0
    for i  in range(batch_num):
        inp = inputs[i]
        lab = labels[i]
        optimizer.zero_grad()
        outputs = net(inp)
        _, predicted = torch.max(outputs.data,1)
        total+=lab.size(0)
        correct+=int((predicted==lab).sum())
        loss = criterion(outputs,Variable(lab))
        loss.backward()
        optimizer.step()
        cnt+=loss.data[0]
        grad = (list(net.parameters())[-1].cpu().data.numpy()**2).sum()
        final_grad+=grad**0.5
        if (i+1)%1==0:
            print('Epoch [%d/%d], Step [%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, loss))
    my_loss = cnt/batch_num
sen = final_grad*batch_size/total
whole_loss = []

origi_f1 = net.fc1.weight
origin_f1 = origi_f1.cpu().data.numpy()
origi_b1 = net.fc1.bias
origin_b1 = origi_b1.cpu().data.numpy()
origi_f2 = net.fc2.weight
origin_f2 = origi_f2.cpu().data.numpy()
origi_b2 = net.fc2.bias
origin_b2 = origi_b2.cpu().data.numpy()
#print(origin_f1,origin_b1,origin_f2,origin_b2)
ran = 0.01
r_f1 = [[i-ran,i+ran] for i in origin_f1]
r_b1 = [[i-ran,i+ran] for i in origin_b1]
r_f2 = [[i-ran,i+ran] for i in origin_f2]
r_b2 = [[i-ran,i+ran] for i in origin_b2]

tf1,tb1,tf2,tb2 = origin_f1,origin_b1,origin_f2,origin_b2

for ind in range(150):
    tf1,tb1,tf2,tb2 = origin_f1,origin_b1,origin_f2,origin_b2
    for i,txt in enumerate(r_f1):
        tf1[i] = random.uniform(txt[0],txt[1])
    for i,txt in enumerate(r_b1):
        tb1[i] = random.uniform(txt[0],txt[1])
    for i,txt in enumerate(r_f2):
        tf2[i] = random.uniform(txt[0],txt[1])
    for i,txt in enumerate(r_b2):
        tb2[i] = random.uniform(txt[0],txt[1])
    par = torch.nn.Parameter(torch.from_numpy(tf1))
    net.fc1.weight = par
    net.fc1.weight.cuda()

    par = torch.nn.Parameter(torch.from_numpy(tb1))
    net.fc1.bias = par
    net.fc1.bias.cuda()
    par = torch.nn.Parameter(torch.from_numpy(tf2))
    net.fc2.weight = par
    net.fc2.weight.cuda()
    par = torch.nn.Parameter(torch.from_numpy(tb2))
    net.fc2.bias = par
    net.fc2.bias.cuda()
    cnt = 0.0
    for i  in range(batch_num):
        inp = inputs[i]
        lab = labels[i]
        outputs = net(inp.cpu())
        loss = criterion(outputs,Variable(lab.cpu()))
        cnt+=loss.data[0]
#        print(i)
    whole_loss.append(cnt/batch_num)
whole_loss = np.array(whole_loss)
with open('result','a') as f:
    print(batch_size,sen,np.mean(whole_loss),my_loss,np.mean(whole_loss)-my_loss,file = f)
