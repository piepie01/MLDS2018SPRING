import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

trainloss = open('difparam_trainloss','a')
testloss = open('difparam_testloss','a')
trainacc = open('difparam_trainacc','a')
testacc = open('difparam_testacc','a')
#Hyper Parameters
input_size = 28*28
hidden_size1 = int(sys.argv[1])
hidden_size2 = int(sys.argv[1])*2
num_classes = 10
num_epochs = 50
learning_rate = 0.001
batch_size = 100

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
        self.fc3 = nn.Linear(hidden_size2,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
#        out = self.fc3(out)
        return out

net = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()
#params = (input_size+1)*hidden_size1+(hidden_size1+1)*hidden_size2+(hidden_size2+1)*10
params = (input_size+1)*hidden_size1+(hidden_size1+1)*10

#loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

inputs = []
labels = []
for i,(inp,lab) in enumerate(train_loader):
    inputs.append(Variable(inp.view(-1,input_size).cuda()))
    labels.append(lab.cuda())


#train
for epoch in range(num_epochs):
    correct = 0
    cnt = 0
    total = 0
    for i  in range(600):
        inp = inputs[i]
        lab = labels[i]
        optimizer.zero_grad()
        outputs = net(inp)
        _, predicted = torch.max(outputs.data,1)
        total+=lab.size(0)
        #print((predicted==lab).sum())
        correct+=int((predicted==lab).sum())
        loss = criterion(outputs,lab)
        loss.backward()
        optimizer.step()
        cnt+=loss
        if (i+1)%100==0:
            print('Epoch [%d/%d], Step [%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, loss))
    if epoch==num_epochs-1:
        print('%d %f'%(params,cnt/600),file=trainloss)
        print('%d %f'%(params,correct/total),file=trainacc)
#test
correct = 0
total = 0
cnt = 0
for inp,lab in test_loader:
    inp = Variable(inp.view(-1,input_size).cuda())
    lab = Variable(lab.cuda())
    #images = Variable(images.view(-1,28*28)).cuda()
    outputs = net(inp)
    _, predicted = torch.max(outputs.data,1)
    correct+=int((predicted==lab).sum())
    total+=int(lab.size(0))
    loss = criterion(outputs,lab)
    cnt+=loss
    #correct+=(predicted.cpu()==labels).sum()
print(correct)
print(total)
print('Accuracy of the network on the 10000 test images: %f' %(correct/total))
print('%d %f'%(params,cnt/100),file=testloss)
print('%d %f'%(params,correct/total),file=testacc)

trainloss.close()
trainacc.close()
testloss.close()
testacc.close()
