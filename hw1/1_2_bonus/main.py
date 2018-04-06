import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import math
import random
#file object
#loss_file = open('pic_data/shallow_loss.txt','w')
#predict_file = open('pic_data/shallow_predict.txt','w')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#Hyper Parameters
input_size = 1
hidden_size1 = 10
output_size = 1
num_epochs = 1000
learning_rate = 0.005

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

net = Net(input_size,hidden_size1,output_size)
#net.cuda()
#print(count_parameters(net))
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
    print('[epoch %d] loss : %f'
        %(epoch+1, loss.data[0]))
    #for p in net.parameters():
    #    if p.grad is not None:
    #        print(p.grad)
    #net.fc1.weight = torch.nn.Parameter(torch.zeros())
    #print(net.fc1.weight)
    #t = net.fc1.weight
    #print(t.data.numpy())
    #par = torch.nn.Parameter(torch.from_numpy(t.data.numpy())) 
    #net.fc1.weight = par
    #exit()

#save
final_weight = None
final_loss = None
whole_weight = []
whole_loss = []
for _ in range(8):

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    print('[epoch %d] loss : %f'
        %(epoch+1, loss.data[0]))
    torch.save(net.state_dict(),'model.plk')

    origi_f1 = net.fc1.weight
    origin_f1 = origi_f1.data.numpy()
    origi_b1 = net.fc1.bias
    origin_b1 = origi_b1.data.numpy()
    origi_f2 = net.fc2.weight
    origin_f2 = origi_f2.data.numpy()
    origi_b2 = net.fc2.bias
    origin_b2 = origi_b2.data.numpy()
    #print(origin_f1,origin_b1,origin_f2,origin_b2)
    ran = 0.01
    r_f1 = [[i-ran,i+ran] for i in origin_f1]
    r_b1 = [[i-ran,i+ran] for i in origin_b1]
    r_f2 = [[i-ran,i+ran] for i in origin_f2]
    r_b2 = [[i-ran,i+ran] for i in origin_b2]

    tf1,tb1,tf2,tb2 = origin_f1,origin_b1,origin_f2,origin_b2
    weight = np.concatenate((tf1.reshape(10,),tb1),axis=0)
    weight = np.concatenate((weight,tf2.reshape(10,)),axis=0)
    weight = np.concatenate((weight,tb2),axis=0)
    weight = weight.reshape(1,-1)
    if _ == 0:
        whole_loss = [0.0]
        whole_weight = weight
    else:
        whole_loss.append(0.0)
        #print(whole_weight.shape,weight.shape)
        whole_weight = np.concatenate((whole_weight,weight),axis = 0)

    for ind in range(4000):
        tf1,tb1,tf2,tb2 = origin_f1,origin_b1,origin_f2,origin_b2
        for i,txt in enumerate(r_f1):
            tf1[i] = random.uniform(txt[0],txt[1])
        for i,txt in enumerate(r_b1):
            tb1[i] = random.uniform(txt[0],txt[1])
        for i,txt in enumerate(r_f2):
            tf2[i] = random.uniform(txt[0],txt[1])
        for i,txt in enumerate(r_b2):
            tb2[i] = random.uniform(txt[0],txt[1])
        weight = np.concatenate((tf1.reshape(10,),tb1),axis=0)
        weight = np.concatenate((weight,tf2.reshape(10,)),axis=0)
        weight = np.concatenate((weight,tb2),axis=0)

        weight = weight.reshape(1,-1)
        #weight = np.concatenate((weight,weight),axis = 0)
        whole_weight = np.concatenate((whole_weight,weight),axis = 0)
        #print(weight)
        #print(da)
        #data_pca_tsne = TSNE(n_components=2)
        #data_pca_tsne.fit_transform(weight)
        #print(da)
        #print(data_pca.explained_variance_)
        #print(data_pca.explained_variance_ratio_)
        #tsne = TSNE(n_components=2)
        #value = tsne.get_params(weight) 
        #tsne.fit(weight)
        #print(value)

        ##分行！！！！！
        par = torch.nn.Parameter(torch.from_numpy(tf1))
        net.fc1.weight = par
        par = torch.nn.Parameter(torch.from_numpy(tb1))
        net.fc1.bias = par
        par = torch.nn.Parameter(torch.from_numpy(tf2))
        net.fc2.weight = par
        par = torch.nn.Parameter(torch.from_numpy(tb2))
        net.fc2.bias = par

        inputs = Variable(xtest.view(-1,1))
        labels = Variable(ytest.view(-1,1))
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        whole_loss.append(loss.data[0])
    net.load_state_dict(torch.load('model.plk'))
    '''
        par = origi_f1
        net.fc1.weight = par
        par = origi_b1
        net.fc1.bias = par
        par = origi_f2
        net.fc2.weight = par
        par = origi_b2
        net.fc2.bias = par
        '''
data_pca = PCA(n_components=10)
da2 = data_pca.fit_transform(whole_weight)
data_tsne = TSNE(n_components=2)
da = data_tsne.fit_transform(da2)
#print(da)
#print(da)
with open('0','w') as fi:
    for i,j in zip(da,whole_loss):
        print(i[0],i[1],j,file = fi)
    #print(whole_weight)
    #print(da)
    #print(whole_loss)
