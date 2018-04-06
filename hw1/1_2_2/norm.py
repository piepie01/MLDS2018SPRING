import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

#Hyper Parameters
input_size = 1
hidden_size1 = 15
hidden_size2 = 15
hidden_size3 = 15
hidden_size4 = 15
output_size = 1
num_epochs = 700
learning_rate = 0.001

#####Deal traindata#####
fetch = open('data/train2','r')
traindata = fetch.read().split('\n')
length = len(traindata)-1
xtrain = []
ytrain = []
for i in range(length):
    traindata[i] = traindata[i].split(' ')
    xtrain.append(float(traindata[i][0]))
    ytrain.append(float(traindata[i][1]))
xtrain = torch.FloatTensor(xtrain)
ytrain = torch.FloatTensor(ytrain)

#####Deal testdata#####
fetch = open('data/test2','r')
testdata = fetch.read().split('\n')
length = len(testdata)-1
xtest = []
ytest = []
for i in range(length):
    testdata[i] = testdata[i].split(' ')
    xtest.append(float(testdata[i][0]))
    ytest.append(float(testdata[i][1]))
xtest = torch.FloatTensor(xtest)
ytest = torch.FloatTensor(ytest)

#NN Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,hidden_size3)
        self.fc4 = nn.Linear(hidden_size3,hidden_size4)
        self.fc5 = nn.Linear(hidden_size4,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
        out = functional.relu(out)
        out = self.fc3(out)
        out = functional.relu(out)
        out = self.fc4(out)
        out = functional.relu(out)
        out = self.fc5(out)
        return out

net = Net(input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,output_size)
#net.cuda()

#loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
L2optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-5)
inputs = Variable(xtrain.view(-1,1))#.cuda())
labels = Variable(ytrain.view(-1,1)) #.cuda())

f_norm = open('gradient_norm/norm','w')
f_acc = open('gradient_norm/loss','w')

#train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    print('Epoch [%d/%d], Step [%d], Loss: %f'
        %(epoch+1, num_epochs, i+1, loss.data[0]),end = '')
    ###this time's work###
    #fc2weight = net.fc2.weight.data.numpy()
    #pca = PCA(n_components=2)
    #pca.fit(fc2weight)
    #tsne = TSNE(n_components=2)
    #tsne.fit(fc2weight)
    #print(pca.components_)
    #print(pca.singular_values_[0],pca.singular_values_[1],file = f)

    inputs1 = Variable(xtest.view(-1,1))
    labels1 = Variable(ytest.view(-1,1))
    correct1 = 0
    total1 = 0
    outputs1 = net(inputs1)
    loss1 = criterion(outputs1,labels1)

    print('%.5f'%(loss),file = f_acc)
    #print(100*correct1/total1)


    #print(tsne.embedding_)
    grad_all = 0.0
    for p in net.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy()**2).sum()
        grad_all += grad
    grad_norm = grad_all**0.5
    print(grad_norm,file = f_norm)
    print('norm :',grad_norm)
    ###







#save
torch.save(net.state_dict(),'model.plk')

#test
inputs = Variable(xtest.view(-1,1))
labels = Variable(ytest.view(-1,1))
correct = 0
total = 0
outputs = net(inputs)
loss = criterion(outputs,labels)
#print('loss=%f'%(loss))
#print('Accuracy of the network on the 10000 test images: %d %%' %(100*correct/total))

