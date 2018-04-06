import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

trainloss = open('interpolar_trainloss','w')
testloss = open('interpolar_testloss','w')
trainacc = open('interpolar_trainacc','w')
testacc = open('interpolar_testacc','w')
#Hyper Parameters
input_size = 784
hidden_size1 = 500
hidden_size2 = 500
num_classes = 10
num_epochs = 50
learning_rate = 0.001
batch_size = 10000
batch_num = 60000//batch_size
inserts = 100
gap = 0.03

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
#        self.fc3 = nn.Linear(hidden_size2,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
#        out = self.fc3(out)
        return out

net1 = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()
net2 = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()
net = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()

#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

inputs = []
labels = []
for i,(inp,lab) in enumerate(train_loader):
    inputs.append(Variable(inp.view(-1,input_size).cuda()))
    labels.append(lab.cuda())

net1.load_state_dict(torch.load('batch50.plk'))
net2.load_state_dict(torch.load('batch1000.plk'))

n1f1 = net1.fc1.weight.data.cpu().numpy()
n1b1 = net1.fc1.bias.data.cpu().numpy()
n1f2 = net1.fc2.weight.data.cpu().numpy()
n1b2 = net1.fc2.bias.data.cpu().numpy()
n2f1 = net2.fc1.weight.data.cpu().numpy()
n2b1 = net2.fc1.bias.data.cpu().numpy()
n2f2 = net2.fc2.weight.data.cpu().numpy()
n2b2 = net2.fc2.bias.data.cpu().numpy()

#train
for __ in range(inserts):
    _ = -2+gap*__
    nf1 = (1-_)*n1f1+_*n2f1
    nb1 = (1-_)*n1b1+_*n2b1
    nf2 = (1-_)*n1f2+_*n2f2
    nb2 = (1-_)*n1b2+_*n2b2
    parf1 = nn.Parameter(torch.from_numpy(nf1).cuda())
    parb1 = nn.Parameter(torch.from_numpy(nb1).cuda())
    parf2 = nn.Parameter(torch.from_numpy(nf2).cuda())
    parb2 = nn.Parameter(torch.from_numpy(nb2).cuda())
    net.fc1.weight = parf1
    net.fc1.bias = parb1
    net.fc2.weight = parf2
    net.fc2.bias = parb2
    correct = 0
    cnt = 0
    total = 0
    for i  in range(batch_num):
        inp = inputs[i]
        lab = Variable(labels[i])
        optimizer.zero_grad()
        outputs = net(inp)
        a, predicted = torch.max(outputs.data,1)
        total+=lab.size(0)
        correct+=int((predicted==labels[i]).sum())
        loss = criterion(outputs,lab)
        cnt+=loss
    print(__)
    print('%f %f'%(_,cnt/batch_num))
    print('%f %f'%(_,correct/total))
    print('%f %f'%(_,cnt/batch_num),file=trainloss)
    print('%f %f'%(_,correct/total),file=trainacc)
#test
    correct = 0
    total = 0
    cnt = 0
    final_grad = 0
    for inp,lab in test_loader:
        inp = Variable(inp.view(-1,input_size).cuda())
        lab = lab.cuda()
        #images = Variable(images.view(-1,28*28)).cuda()
        outputs = net(inp)
        a, predicted = torch.max(outputs.data,1)
        correct+=int((predicted==lab).sum())
        total+=int(lab.size(0))
        loss = criterion(outputs,Variable(lab))
        cnt+=loss
    print(final_grad*batch_size/total)
    print(correct)
    print(total)
    print('Accuracy of the network on the 10000 test images: %f' %(correct/total))
    print('%f %f'%(_,cnt/total*batch_size),file=testloss)
    print('%f %f'%(_,correct/total),file=testacc)

trainloss.close()
trainacc.close()
testloss.close()
testacc.close()
