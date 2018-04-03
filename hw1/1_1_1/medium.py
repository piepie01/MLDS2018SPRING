import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import math

#file object
loss_file = open('result_data/medium_loss.txt','w')
predict_file = open('result_data/medium_predict.txt','w')


#Hyper Parameters
input_size = 1
hidden_size1 = 20
hidden_size2 = 45
hidden_size3 = 20
output_size = 1
num_epochs = 10000
learning_rate = 0.01

#####Deal traindata#####
fetch = open('./data/train2','r')
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
fetch = open('./data/test2','r')
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
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,hidden_size3)
        self.fc4 = nn.Linear(hidden_size3,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.tanh(out)
        out = self.fc2(out)
        out = functional.tanh(out)
        out = self.fc3(out)
        out = functional.tanh(out)
        out = self.fc4(out)
        return out

net = Net(input_size,hidden_size1,output_size)
#net.cuda()

#loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
inputs = Variable(xtrain.view(-1,1))
labels = Variable(ytrain.view(-1,1))
#labels = Variable(labels)
#labels = Variable(labels.cuda())

#train
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    print('%d %f'
        %(epoch+1,loss.data[0]),file = loss_file)
    #print('%d %f'
    #    %(epoch+1,loss.data[0]))

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
for x,y_p in zip(inputs,outputs):
    print(float(x),float(y_p),file=predict_file)
#print('Accuracy of the network on the 10000 test images: %d %%' %(100*correct/total))
#close file
loss_file.close()
predict_file.close()

