import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import pad
from torch.nn.functional import avg_pool3d
import torch.optim as optim
import torchvision
from torchvision import transforms
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import random
import sys
import os

#get dataset
resizetensor = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
totensor = transforms.ToTensor()

class Dataset(torch.utils.data.Dataset):
    def __init__(self,imagedir,labels,loader=torchvision.datasets.folder.default_loader):
        self.loader = loader
        self.listfiles = [os.path.join(imagedir,str(img)+'.jpg') for img in range(36740)]
        self.df = []
        for image in self.listfiles:
            img = Image.open(image)
            fp = img.fp
            img.load()
            #self.df.append(resizetensor(img))
            self.df.append(totensor(img)*2-1)
        self.label = open(labels).read().split('\n')[:-1]
        self.hairencoding = {}
        self.hairencoding_rev = {}
        self.eyesencoding = {}
        self.eyesencoding_rev = {}
        length = len(self.label)
        for i in range(length):
            self.label[i] = [self.label[i].split(',')[1].split(' ')[j] for j in [0,2]]
            if self.label[i][0] not in self.hairencoding.keys():
                self.hairencoding_rev[len(self.hairencoding.keys())] = self.label[i][0]
                self.hairencoding[self.label[i][0]] = len(self.hairencoding.keys())
            if self.label[i][1] not in self.eyesencoding.keys():
                self.eyesencoding_rev[len(self.eyesencoding.keys())] = self.label[i][1]
                self.eyesencoding[self.label[i][1]] = len(self.eyesencoding.keys())

    def __getitem__(self,index):
        label = np.zeros(22)
        label[self.hairencoding[self.label[index][0]]] = 1
        label[self.eyesencoding[self.label[index][1]]+12] = 1
        return self.df[index],label

    def __len__(self):
        n=len(self.df)
        return n

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(122,5*5*512)
        self.batchnorm1 = nn.BatchNorm1d(5*5*512,momentum=0.9)
        self.conv1 = nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=5,
                               stride=2)
        self.batchnorm2 = nn.BatchNorm2d(256,momentum=0.9)
        self.conv2 = nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=5,
                               stride=2,
                               output_padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128,momentum=0.9)
        self.conv3 = nn.ConvTranspose2d(in_channels=128,
                               out_channels=3,
                               kernel_size=5,
                               stride=2,
                               output_padding=1)
        self.rrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
    def forward(self,x):
        out = self.fc1(x)
        out = self.batchnorm1(out)
        out = out.view(-1,512,5,5)
        out = self.conv1(out)
        out = self.batchnorm2(out)
        out = self.rrelu(out)
        out = self.conv2(out)
        out = self.batchnorm3(out)
        out = self.rrelu(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=128,
                               kernel_size=5,
                               stride=2)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=5,
                               stride=2)
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=5,
                               stride=2)
        self.conv4 = nn.Conv2d(in_channels=512+22,
                               out_channels=1024,
                               kernel_size=1,
                               stride=1)
        self.fcaux = nn.Linear(1024*5*5,22)
        self.fcdis = nn.Linear(1024*5*5,2)
        self.rrelu = nn.LeakyReLU(0.2)

    def forward(self,x,label):
        out = self.conv1(x)
        out = self.rrelu(out)
        out = self.conv2(out)
        out = self.rrelu(out)
        out = self.conv3(out)
        out = self.rrelu(out)
        label = label.repeat(1,25).view(-1,25,22).transpose(1,2).contiguous().view(-1,22,5,5)
        out = torch.cat((out,label),1)
        out = self.conv4(out)
        out = self.rrelu(out)
        out = out.view(-1,1024*5*5)
        out1 = self.fcaux(out)
        out1 = functional.sigmoid(out1)
        out2 = self.fcdis(out)
        out2 = functional.softmax(out2,dim=1)
        return out1,out2

class GAN(nn.Module):
    def __init__(self):
        super(GAN,self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def genforward(self,x):
        out = self.generator(x)
        return out

    def disforward(self,x,label):
        out = self.discriminator(x,label)
        return out

    def forward(self,x,label):
        out = self.genforward(x)
        out = self.disforward(x,label)
        return out

class Frame():
    def __init__(self,traindata=None,trainlabel=None,testlabel=None,batch_size=100,num_epochs=20,num_steps=100,learning_rate=1e-4,cuda=False,Log=None,loaddictpath=None,savedictpath=None,resultpath=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.cuda = cuda
        if Log is None:
            self.Log = sys.stdout
        else:
            self.Log = open(Log,'w')
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testlabel = testlabel
        self.savedictpath = savedictpath
        self.loaddictpath = loaddictpath
        if resultpath is None:
            raise Exception('Error : Resultpath not specified')
        self.resultpath = resultpath

    def loaddata(self):
        if self.traindata is None:
            raise Exception('Error : Train Datapath not specified')
        self.traindataset = Dataset(imagedir=self.traindata,labels=self.trainlabel)
        self.trainloader = torch.utils.data.DataLoader(dataset=self.traindataset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=0)

    def init_model(self):
        self.model = GAN()
        if self.loaddictpath is not None:
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.model.cuda()
        self.criterionS = nn.CrossEntropyLoss()
        self.criterionC = nn.BCELoss()
        self.genoptimizer = torch.optim.Adam(self.model.generator.parameters(),lr=self.learning_rate)
        self.disoptimizer = torch.optim.Adam(self.model.discriminator.parameters(),lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            gentotal, genavgloss, gencacc, gendacc, distotal, disavgloss, discacc, disdacc = self.train_util()
            print('Dis || Epoch [%d/%d], Loss: %.4f, CAcc: %.4f, DAcc: %.4f'%(epoch+1,self.num_epochs,disavgloss/distotal,discacc/distotal,disdacc/distotal),file=self.Log)
            print('Gen || Epoch [%d/%d], Loss: %.4f, CAcc: %.4f, DAcc: %.4f'%(epoch+1,self.num_epochs,genavgloss/gentotal,gencacc/gentotal,gendacc/gentotal),file=self.Log)
            self.Log.flush()
            if epoch%20==19 or epoch==self.num_epochs-1:
                self.checkpoint()

    def train_util(self):

        gentotal = 0
        genavgloss = 0
        gencacc = 0
        gendacc = 0
        distotal = 0
        disavgloss = 0
        discacc = 0
        disdacc = 0

        self.model.generator.eval()
        self.model.discriminator.train()
        for i,(inputs,labels) in enumerate(self.trainloader):
            if i>=self.num_steps:
                break
            self.disoptimizer.zero_grad()
            noise = torch.FloatTensor(np.random.normal(0,1,(inputs.size(0),100)))
            randomlabels = torch.zeros(inputs.size(0),22).type(torch.FloatTensor)
            for j in range(inputs.size(0)):
                randomlabels[j][random.randint(0,11)] = 1.0
                randomlabels[j][random.randint(12,21)] = 1.0
            labels = torch.cat((labels.type(torch.FloatTensor),randomlabels),0)
            noise = torch.cat((noise,randomlabels),1)
            fake = torch.cat((torch.ones(inputs.size(0)),torch.zeros(inputs.size(0))),0).long()
            if self.cuda is True:
                inputs = inputs.cuda()
                labels = labels.cuda()
                noise = noise.cuda()
                fake = fake.cuda()
            noiseinputs = self.model.genforward(Variable(noise))
            noiseinputs = noiseinputs.data
            inputs = torch.cat((inputs,noiseinputs),0)
            inputs = Variable(inputs)
            aux, dis = self.model.disforward(inputs,Variable(labels))
            labels[-noiseinputs.size(0):,:] = 0.0
            #print(aux[-1].data.cpu().numpy())
            #print(labels[-1].cpu().numpy())
            #exit()
            loss = self.criterionC(aux,Variable(labels))+self.criterionS(dis,Variable(fake))
            #loss = self.criterionS(dis,Variable(fake))
            loss.backward()
            self.disoptimizer.step()

            aux = aux.data
            aux[aux>0.5] = 1.0
            aux[aux<=0.5] = 0.0
            correct1 = int(sum((pre.all() for pre in labels==aux)))
            _, predicted = torch.max(dis.data,1)
            correct2 = int((fake==predicted).sum())
            discacc+=correct1
            disdacc+=correct2
            distotal+=labels.size(0)
            disavgloss+=float(loss.data.cpu())*labels.size(0)
            if i%1==0:
                print('Dis || STEP %d, Loss: %.4f, CAcc: %.4f, DAcc: %.4f'%(i+1,loss.data.cpu(),correct1/labels.size(0),correct2/labels.size(0)),file=self.Log)
                self.Log.flush()
            if correct1/labels.size(0)==1 and correct2/labels.size(0)==1:
                break

        self.model.generator.train()
        self.model.discriminator.eval()
        for i in range(self.num_steps):
            self.genoptimizer.zero_grad()
            noise = torch.FloatTensor(np.random.normal(0,1,(self.batch_size,100)))
            labels = torch.zeros(self.batch_size,22).type(torch.FloatTensor)
            for j in range(self.batch_size):
                labels[j][random.randint(0,11)] = 1.0
                labels[j][random.randint(12,21)] = 1.0
            noise = torch.cat((noise,labels),1)
            fake = torch.ones(self.batch_size).long()
            if self.cuda is True:
                labels = labels.cuda()
                noise = noise.cuda()
                fake = fake.cuda()
            noise = Variable(noise)
            inputs = self.model.genforward(noise)
            aux, dis = self.model.disforward(inputs,Variable(labels))
            loss = self.criterionC(aux,Variable(labels))+self.criterionS(dis,Variable(fake))
            #loss = self.criterionS(dis,Variable(fake))
            loss.backward()
            self.genoptimizer.step()

            aux = aux.data
            aux[aux>0.5] = 1.0
            aux[aux<=0.5] = 0.0
            correct1 = int(sum((pre.all() for pre in labels==aux)))
            _, predicted = torch.max(dis.data,1)
            correct2 = int((fake==predicted).sum())
            gencacc+=correct1
            gendacc+=correct2
            gentotal+=labels.size(0)
            genavgloss+=float(loss.data.cpu())*labels.size(0)
            if i%1==0:
                print('Gen || STEP %d, Loss: %.4f, CAcc: %.4f, DAcc: %.4f'%(i+1,loss.data.cpu(),correct1/labels.size(0),correct2/labels.size(0)),file=self.Log)
                self.Log.flush()
            if correct1/labels.size(0)==1 and correct2/labels.size(0)==1:
                break

        return gentotal,genavgloss,gencacc,gendacc,distotal,disavgloss,discacc,disdacc

    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.model.state_dict(),self.savedictpath)

    def test(self):
        self.model.eval()
        tag = list(filter(None,open(self.testlabel).read().split('\n')))
        staticinp = np.load('staticinput.npy')
        self.hairdict = {'green':2,'purple':11,'red':4,'white':5,'brown':9,'pink':10,'aqua':0,'blonde':7,'black':6,'orange':3,'blue':8,'gray':1}
        self.eyesdict = {'yellow':9,'green':4,'purple':7,'red':8,'blue':2,'orange':5,'brown':3,'pink':6,'aqua':0,'black':1}
        for i in range(25):
            tag[i] = [tag[i].split(',')[1].split(' ')[j] for j in [0,2]]
            tag[i][0] = self.hairdict[tag[i][0]]
            tag[i][1] = self.eyesdict[tag[i][1]]
            #noise = torch.FloatTensor(np.random.normal(0,0.8,(1,100)))
            noise = torch.FloatTensor(staticinp[i])
            labels = torch.zeros(1,22).type(torch.FloatTensor)
            labels[0][tag[i][0]] = 1.0
            labels[0][tag[i][1]+12] = 1.0
            noise = torch.cat((noise,labels),1)
            if self.cuda is True:
                noise = noise.cuda()
            noise = Variable(noise)
            img = self.model.genforward(noise)
            img = img.data.cpu().numpy()[0].transpose(1,2,0)
            np.save(self.resultpath+str(i)+'.npy',img)

    def showinterpolation(self):
        self.model.eval()
        noise1 = torch.FloatTensor(np.random.normal(0,0.8,(1,100)))
        label1 = torch.zeros(1,22)
        label1[0][random.randint(0,11)] = 1.0
        label1[0][random.randint(12,21)] = 1.0
        noise1 = torch.cat((noise1,label1),1)
        for types in range(5):
            noise2 = torch.FloatTensor(np.random.normal(0,0.8,(1,100)))
            label2 = torch.zeros(1,22)
            label2[0][random.randint(0,11)] = 1.0
            label2[0][random.randint(12,21)] = 1.0
            noise2 = torch.cat((noise2,label2),1)
            diff = (noise2-noise1)/4
            for i in range(5):
                noise = noise1+diff*i
                if self.cuda is True:
                    noise = noise.cuda()
                noise = Variable(noise)
                img = self.model.genforward(noise)
                img = img.data.cpu().numpy()[0].transpose(1,2,0)
                np.save(self.resultpath+str(types*5+i)+'.npy',img)
            noise1 = noise2.clone()

def main():
    Model = Frame(traindata=None, #'../extra_data/images',
                  trainlabel=None, #'../extra_data/tags.csv',
                  testlabel=sys.argv[1],
                  #testlabel='../AnimeDataset/testing_tags.txt',
                  #testlabel='testing_tags.txt',
                  batch_size=64,
                  num_epochs=500,
                  num_steps=40,
                  learning_rate=1e-4,
                  cuda=True,
                  Log=None,
                  loaddictpath='3_2.plk',
                  savedictpath=None,
                  resultpath='samples/Cans')
    #Model.loaddata()
    Model.init_model()
    #Model.train()
    Model.test()
    #Model.showinterpolation()

main()
