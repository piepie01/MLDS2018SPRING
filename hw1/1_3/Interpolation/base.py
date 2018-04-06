import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys

trainloss = open('difapproach_trainloss','a')
testloss = open('difapproach_testloss','a')
trainacc = open('difapproach_trainacc','a')
testacc = open('difapproach_testacc','a')
sensitivity = open('difapproach_sensitivity','a')
#Hyper Parameters
input_size = 784
hidden_size1 = 500
hidden_size2 = 500
num_classes = 10
num_epochs = 50
learning_rate = 0.001
batch_size = 10000
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
#        self.fc3 = nn.Linear(hidden_size2,num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = functional.relu(out)
        out = self.fc2(out)
#        out = self.fc3(out)
        return out

net = Net(input_size,hidden_size1,hidden_size2,num_classes).cuda()

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
    for i  in range(batch_num):
        inp = inputs[i]
        lab = Variable(labels[i])
        optimizer.zero_grad()
        outputs = net(inp)
        _, predicted = torch.max(outputs.data,1)
        total+=lab.size(0)
        correct+=int((predicted==labels[i]).sum())
        loss = criterion(outputs,lab)
        loss.backward()
        optimizer.step()
        cnt+=loss
        if (i+1)%1==0:
            print('Epoch [%d/%d], Step [%d], Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, loss))
    if epoch==num_epochs-1:
        print('%d %f'%(batch_size,cnt/batch_num),file=trainloss)
        print('%d %f'%(batch_size,correct/total),file=trainacc)
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
    _, predicted = torch.max(outputs.data,1)
    correct+=int((predicted==lab).sum())
    total+=int(lab.size(0))
    loss = criterion(outputs,Variable(lab))
    loss.backward()
    grad_all = 0
    grad = (list(net.parameters())[-1].cpu().data.numpy()**2).sum()
    final_grad+=grad**0.5
#    for p in net.parameters():
#        grad = 0
#        if p.grad is not None:
#            grad = (p.grad.cpu().data.numpy()**2).sum()
#        grad_all+=grad
#    final_grad+=grad_all**0.5
    cnt+=loss
    #correct+=(predicted.cpu()==labels).sum()
print(final_grad*batch_size/total)
print(correct)
print(total)
print('Accuracy of the network on the 10000 test images: %f' %(correct/total))
print('%d %f'%(batch_size,cnt/total*batch_size),file=testloss)
print('%d %f'%(batch_size,correct/total),file=testacc)
print('%d %f'%(batch_size,final_grad*batch_size/total),file=sensitivity)

trainloss.close()
trainacc.close()
testloss.close()
testacc.close()
sensitivity.close()
torch.save(net.state_dict(),'batch1000.plk')
