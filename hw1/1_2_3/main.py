import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numdifftools as nd
import numpy as np
import math
import random
#file object
#loss_file = open('pic_data/shallow_loss.txt','w')
#predict_file = open('pic_data/shallow_predict.txt','w')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def hessian(loss, model):
    var = model.parameters()
    temp = []
    grads = torch.autograd.grad(loss, var, create_graph=True)
    grads = torch.cat([g.view(-1) for g in grads])
    for grad in grads:
        var = model.parameters()
        grad2 = torch.autograd.grad(grad, var, create_graph=True)
        fla = []
        for g in grad2:
            fla.append(g.data.numpy().reshape(-1,))
        con = np.concatenate((fla[0],fla[1]),axis = 0)
        con = np.concatenate((con,fla[2]),axis = 0)
        con = np.concatenate((con,fla[3]),axis = 0)
        temp.append(con)
    temp = np.array(temp)
    for i in range(31):
        for j in range(31):
            if 0<=i<=9 and 0<=j<=9 or 10<=i<=19 and 10<=j<=19 or 20<=i<=29 and 20<=j<=29 or i==30 and j==30:
                pass
            else:
                temp[i][j] = 0.0
    return temp
#Hyper Parameters
input_size = 1
hidden_size1 = 10
output_size = 1
num_epochs = 1000
learning_rate = 0.008

#####Deal traindata#####
fetch = open('./data/train','r')
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
fetch = open('./data/test','r')
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
        out = functional.tanh(out)
        out = self.fc2(out)
        return out

result = open('loss_ratio','w')
for _ in range(200):
    net = Net(input_size,hidden_size1,output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    inputs = Variable(xtrain.view(-1,1))
    labels = Variable(ytrain.view(-1,1))

    #train
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        print('[ %d epoch %d] loss : %f'%(_,epoch+1, loss.data[0]),end = ' ')
        grad_all = 0.0
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
        grad_norm = grad_all**0.5
        print('norm :',grad_norm)



    optimizer = torch.optim.LBFGS(net.parameters(),lr=learning_rate)
    for epoch in range(num_epochs+1,num_epochs+1001):
        def closure():
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            return loss
        optimizer.step(closure) 
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        print('[ %d epoch %d] loss : %f'%(_,epoch+1, loss.data[0]),end = ' ')
        grad_all = 0.0
        for p in net.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
        grad_norm = grad_all**0.5
        print('norm :',grad_norm)
        if grad_norm < 0.1:
            hes = hessian(loss,net)
            w,v = np.linalg.eig(hes)
            cnt = 0
            for i in range(len(w)):
                if w[i]>=0:
                    cnt+=1
            print(loss.data[0],cnt/len(w),file = result)
            break


#save
torch.save(net.state_dict(),'model.plk')
