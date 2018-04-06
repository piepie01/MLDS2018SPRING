import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from random import shuffle
import random

shuffle_trainacc = open('shuffle_trainacc','w')
shuffle_trainloss = open('shuffle_trainloss','w')
shuffle_testacc = open('shuffle_testacc','w')
shuffle_testloss = open('shuffle_testloss','w')

#Hyper Parameters
input_size = 784
hidden_size1 = 700
hidden_size2 = 700
hidden_size3 = 700
num_classes = 10
num_epochs = 50000
learning_rate = 0.0001
batch_size = 100

#get dataset
train_dataset = datasets.MNIST(root='../data', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='../data', 
                              train=False, 
                              transform=transforms.ToTensor())

#load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          shuffle=False,
                                          batch_size=batch_size,
                                          num_workers=0)
#NN Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,hidden_size3)
        self.fc4 = nn.Linear(hidden_size3,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

net = Net(input_size,hidden_size1,hidden_size2,hidden_size3,num_classes)
net.cuda()

#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

inputs = []
labels = []
testinputs = []
testlabels = []

for i,(inp,lab) in enumerate(train_loader):
    inputs.append(Variable(inp.view(-1,input_size).cuda()))
    labels.append(lab.cuda())
shuffle(labels)
for i,(inp,lab) in enumerate(test_loader):
    testinputs.append(Variable(inp.view(-1,input_size).cuda()))
    testlabels.append(lab.cuda())

#train
for epoch in range(num_epochs):
    correct = 0
    total = 0
    cnt = 0
    for i in range(600):
        inp = inputs[i]
        lab = labels[i]
        optimizer.zero_grad()
        outputs = net(inp)
        _,predicted = torch.max(outputs.data,1)
        total+=lab.size(0)
        correct+=int((predicted==lab).sum())
        loss = criterion(outputs,Variable(lab))
        loss.backward()
        optimizer.step()
        cnt+=float(loss)
        if (i+1)%100==0:
            print('Epoch [%d/%d], Step [%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, loss))
    print(correct/total,file=shuffle_trainacc)
    print(cnt/600,file=shuffle_trainloss)
    print(correct/total)
    shuffle_trainacc.flush()
    shuffle_trainloss.flush()
    correct = 0
    total = 0
    cnt = 0
    for i in range(100):
        inp = testinputs[i]
        lab = testlabels[i]
        outputs = net(inp)
        _,predicted = torch.max(outputs.data,1)
        correct+=int((predicted==lab).sum())
        total+=lab.size(0)
        loss = criterion(outputs,Variable(lab))
        cnt+=float(loss)
    print(correct/10000)
    print(correct/total,file=shuffle_testacc)
    print(cnt/100,file=shuffle_testloss)
    shuffle_testacc.flush()
    shuffle_testloss.flush()
    
