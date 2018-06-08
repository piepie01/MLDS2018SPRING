import torch
from torch.autograd import grad
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
import sys
import os


#get dataset
resizetensor = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])
totensor = transforms.ToTensor()

class Dataset(torch.utils.data.Dataset):
    def __init__(self,imagedir,loader=torchvision.datasets.folder.default_loader):
        self.loader = loader
        self.listfiles = [os.path.join(imagedir,img) for img in os.listdir(imagedir)]
        self.df = []
        for image in self.listfiles:
            img = Image.open(image)
            fp = img.fp
            img.load()
            #self.df.append(resizetensor(img))
            self.df.append(totensor(img)*2-1)

    def __getitem__(self,index):
        return self.df[index]

    def __len__(self):
        n=len(self.df)
        return n

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(100,5*5*512)
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
        self.conv4 = nn.Conv2d(in_channels=512,
                               out_channels=1024,
                               kernel_size=1,
                               stride=1)
        self.fc1 = nn.Linear(1024*5*5,1)
        self.rrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        out = self.conv1(x)
        out = self.rrelu(out)
        out = self.conv2(out)
        out = self.rrelu(out)
        out = self.conv3(out)
        out = self.rrelu(out)
        out = self.conv4(out)
        out = self.rrelu(out)
        out = out.view(-1,1024*5*5)
        out = self.fc1(out)
        return out

class GAN(nn.Module):
    def __init__(self):
        super(GAN,self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def genforward(self,x):
        out = self.generator(x)
        return out

    def disforward(self,x):
        out = self.discriminator(x)
        return out

    def forward(self,x):
        out = self.genforward(x)
        out = self.disforward(x)
        return out

class Frame():
    def __init__(self,traindata=None,batch_size=100,num_epochs=20,num_steps=100,learning_rate=1e-4,lambd=10,alpha=0.5,cuda=False,Log=None,loaddictpath=None,savedictpath=None,resultpath=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.alpha = alpha
        self.cuda = cuda
        if Log is None:
            self.Log = sys.stdout
        else:
            self.Log = open(Log,'w')
        self.traindata = traindata
        self.savedictpath = savedictpath
        self.loaddictpath = loaddictpath
        if resultpath is None:
            raise Exception('Error : Resultpath not specified')
        self.resultpath = resultpath

    def loaddata(self):
        if self.traindata is None:
            raise Exception('Error : Train Datapath not specified')
        self.traindataset = Dataset(imagedir=self.traindata)
        self.trainloader = torch.utils.data.DataLoader(dataset=self.traindataset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=0)

    def interpolation(self,real,fake):
        omega = torch.acos(torch.mul(torch.div(real,torch.norm(real,p=2,dim=1).view(-1,1)),torch.div(fake,torch.norm(fake,p=2,dim=1).view(-1,1))))
        so = torch.sin(omega)
        alpha = torch.rand(omega.size(0)).view(-1,1)
        if self.cuda is True:
            alpha = alpha.cuda()
        return torch.sin((1-alpha)*omega)/so*real+torch.sin((alpha)/so*fake)

    def grad_penalty(self,real,fake):
        real = real.view(self.batch_size,-1)
        fake = fake.view(self.batch_size,-1)
        z = self.interpolation(real,fake)
        z = Variable(z,requires_grad=True)
        z = z.view(self.batch_size,3,64,64)
        o = self.model.disforward(z)
        grad_out = torch.ones(o.size())
        if self.cuda is True:
            grad_out = grad_out.cuda()
        g = grad(o,z,grad_outputs=grad_out,create_graph=True)[0].view(z.size(0),-1)
        gp = self.lambd*torch.mean(torch.pow(torch.norm(g)-1,2))
        return gp
        
    def DLoss(self,real,fake,rimg,fimg):
        return -(torch.mean(real)-torch.mean(fake))+self.grad_penalty(rimg,fimg)

    def GLoss(self,gen):
        return -torch.mean(gen)

    def init_model(self):
        self.model = GAN()
        if self.loaddictpath is not None:
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.model.cuda()
        self.genoptimizer = torch.optim.Adam(self.model.generator.parameters(),lr=self.learning_rate)
        self.disoptimizer = torch.optim.Adam(self.model.discriminator.parameters(),lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            gentotal, genavgloss, distotal, disavgloss = self.train_util()
            print('Dis || Epoch [%d/%d], Loss: %.4f'%(epoch+1,self.num_epochs,disavgloss/distotal),file=self.Log)
            print('Gen || Epoch [%d/%d], Loss: %.4f'%(epoch+1,self.num_epochs,genavgloss/gentotal),file=self.Log)
            self.Log.flush()
            if epoch%20==19 or epoch==self.num_epochs-1:
                self.checkpoint()

    def train_util(self):

        gentotal = 0
        genavgloss = 0
        distotal = 0
        disavgloss = 0

        self.model.generator.eval()
        self.model.discriminator.train()
        for i,inputs in enumerate(self.trainloader):
            if i>=self.num_steps*5:
                break
            noise = torch.FloatTensor(np.random.normal(0,1,(inputs.size(0),100)))
            if self.cuda is True:
                inputs = inputs.cuda()
                noise = noise.cuda()
            noise = Variable(noise)
            inputs = Variable(inputs)
            noiseinputs = self.model.genforward(noise)
            outputs = self.model.disforward(inputs)
            noiseoutputs = self.model.disforward(noiseinputs)

            loss = self.DLoss(outputs,noiseoutputs,inputs.data,noiseinputs.data)
            self.disoptimizer.zero_grad()
            loss.backward()
            self.disoptimizer.step()

            _, predicted = torch.max(outputs.data,1)
            disloss = float(loss.data.cpu())
            distotal+=inputs.size(0)
            disavgloss+=disloss*inputs.size(0)
            if i%1==0:
                print('Dis || STEP %d, Loss: %.4f'%(i+1,disloss),file=self.Log)
            self.Log.flush()

        self.model.generator.train()
        self.model.discriminator.eval()
        for i in range(self.num_steps):
            noise = torch.FloatTensor(np.random.normal(0,1,(self.batch_size,100)))
            if self.cuda is True:
                noise = noise.cuda()
            noise = Variable(noise)
            inputs = self.model.genforward(noise)
            outputs = self.model.disforward(inputs)

            loss = self.GLoss(outputs)
            self.genoptimizer.zero_grad()
            loss.backward()
            self.genoptimizer.step()

            gentotal+=inputs.size(0)
            genavgloss+=float(loss.data.cpu())*inputs.size(0)
            if i%1==0:
                print('Gen || STEP %d, Loss: %.4f'%(i+1,loss.data.cpu()),file=self.Log)
            self.Log.flush()

        return gentotal,genavgloss,distotal,disavgloss

    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.model.state_dict(),self.savedictpath)

    def test(self,gennum):
        self.model.eval()
        noises = np.load('staticinput.npy')
        for i in range(gennum):
            #noise = torch.FloatTensor(np.random.normal(0,1,(1,100)))
            noise = torch.FloatTensor(noises[i])
            #noise = torch.FloatTensor(-np.ones((1,100)))
            #noise[0] = -1+i*0.08
            if self.cuda is True:
                noise = noise.cuda()
            noise = Variable(noise)
            img = self.model.genforward(noise)
            img = img.data.cpu().numpy()[0].transpose(1,2,0)
            np.save(self.resultpath+str(i)+'.npy',img)


def main():
    Model = Frame(traindata=None, #'../extra_data/images',
                  batch_size=64,
                  num_epochs=10000,
                  num_steps=1,
                  learning_rate=1e-4,
                  lambd=10,
                  alpha=0.5,
                  cuda=True,
                  Log=None,
                  loaddictpath='3_1.plk',
                  savedictpath=None,
                  resultpath='samples/ans')
    #Model.loaddata()
    Model.init_model()
    #Model.train()
    Model.test(25)

main()
