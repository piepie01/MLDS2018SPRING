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
hidden_size1 = 100
output_size = 1
num_epochs = 1000
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
    def __init__(self, input_size, hidden_size1, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
        return out

whole_weight = []
layer2_weight = []
loss_data = []

for filename in range(8):
    net = Net(input_size,hidden_size1,output_size)
    #net.cuda()

    #loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    L2optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-5)
    inputs = Variable(xtrain.view(-1,1))#.cuda())
    labels = Variable(ytrain.view(-1,1)) #.cuda())
    #train
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        print('%d Epoch [%d/%d], Step [%d], Loss: %f'
            %(filename,epoch+1, num_epochs, i+1, loss.data[0]))
        ###this time's work###
        fc1w = net.fc1.weight.data.numpy()
        fc1b = net.fc1.bias.data.numpy()
        fc2w = net.fc2.weight.data.numpy()
        fc2b = net.fc2.bias.data.numpy()
        t_whole_weight = np.concatenate((fc1w.reshape(100,),fc1b),axis = 0)
        t_whole_weight = np.concatenate((t_whole_weight,fc2w.reshape(100,)),axis = 0)
        t_whole_weight = np.concatenate((t_whole_weight,fc2b),axis = 0)
        t_whole_weight = t_whole_weight.reshape(1,-1)

        t_layer2_weight = np.concatenate((fc1w.reshape(100,),fc1b),axis = 0)
        t_layer2_weight = t_layer2_weight.reshape(1,-1)

        if filename == 0 and epoch == 0:
            whole_weight = t_whole_weight
            layer2_weight = t_layer2_weight
        else:
            whole_weight = np.concatenate((whole_weight,t_whole_weight),axis = 0)
            layer2_weight = np.concatenate((layer2_weight,t_layer2_weight),axis = 0)
        loss_data.append(loss.data[0])

whole_pca = PCA(n_components = 2)
layer2_pca = PCA(n_components = 2)
whole_da = whole_pca.fit_transform(whole_weight)
layer2_da = layer2_pca.fit_transform(layer2_weight)

for i in range(8):
    f_w = open('weight/'+str(i),'w')
    f_2 = open('weight/layer_'+str(i),'w')
    f_a = open('acc/'+str(i),'w')
    for j in range(i*num_epochs,(i+1)*num_epochs):
        print(whole_da[j][0],whole_da[j][1],file = f_w)
        print(layer2_da[j][0],layer2_da[j][1],file = f_2)
        print(loss_data[j],file = f_a)
    f_w.close()
    f_2.close()
    f_a.close()
