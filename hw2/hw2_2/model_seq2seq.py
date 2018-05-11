import dataset
import sys
import copy
import numpy as np
import random
import pickle

import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
from model import Encoder
from model import Decoder

train_file = sys.argv[1]
batch_size = 16
enco_input_size = 10
enco_hidden_size = 500
embed_size = 1500
deco_hidden_size = 1000
deco_input_size = 0
sent_limit = 12
learning_rate = 1e-3
epochs = 100
times = '9'
model_name = 'model'+times+'.th'


online = 1
post = sys.stdout
if online == 1:
    post = open('/home/piepie01/public_html/123/output1','w')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class clr_conv(Dataset):
    def __init__(self,train_X,train_Y,word2index):
        self.X = copy.copy(train_X)
        self.Y = copy.copy(train_Y)
        self.word2index = word2index
    def __len__(self):
        return len(self.Y)
    def __getitem__(self,index):
        x = [0 for i in range(sent_limit)]
        y = [0 for i in range(sent_limit)]
        x[0] = 1
        y[0] = 1
        for i,txt in enumerate(self.X[index].split()):
            if txt in self.word2index.keys():
                x[i+1] = self.word2index[txt]
            else:
                x[i+1] = 3
            x[i+2] = 2
        for i,txt in enumerate(self.Y[index].split()):
            if txt in self.word2index.keys():
                y[i+1] = self.word2index[txt]
            else:
                y[i+1] = 3
            y[i+2] = 2
        x = torch.LongTensor(x).cuda()
        y = torch.LongTensor(y).cuda()
        return x,y
        
def index2sent(sent,index2word):
    l = [index2word[i] for i in sent]
    return ' '.join(l)
def get_weight(length):
    w = np.array([1 for _ in range(length)])


def main(argv):
    train_X, train_Y = dataset.read_data(train_file,post)
    word2index, index2word = dataset.word2vec(train_X,post)
    #print(word2index[:10])
    with open('save/word'+times+'.pickle','wb') as f:
        pickle.dump(word2index,f)
    with open('save/index'+times+'.pickle1','wb') as f:
        pickle.dump(index2word,f)
    #exit()
    conv = clr_conv(train_X,train_Y,word2index)
    data_loader = DataLoader(conv, shuffle = True, batch_size = batch_size)

    encoder = Encoder(enco_input_size, enco_hidden_size, len(word2index),embed_size).cuda()
    decoder = Decoder(deco_input_size, deco_hidden_size, len(word2index), sent_limit,embed_size).cuda()


    print("Encoder's parameters :",count_parameters(encoder))
    print("Decoder's parameters :",count_parameters(decoder))
    weight = get_weight(len(word2index))
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.NLLLoss().cuda()
    enco_opt = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    deco_opt = torch.optim.Adam(decoder.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        total = 0
        step_count = 0
        max_step = 10000
        for x,y in data_loader:
            if step_count > max_step:
                break
                #continue
            enco_opt.zero_grad()
            deco_opt.zero_grad()
            inp = Variable(x)
            oup = Variable(y)
            x_size = x.size()
            y_size = y.size()
            #print('x_size :',x_size)
            #print('y_size :',y_size)
            deco_padding = Variable(torch.zeros(1,x_size[0],deco_hidden_size).cuda())
            #enco_hid = Variable(torch.zeros(1,batch_size,enco_hidden_size).cuda()), Variable(torch.zeros(1,batch_size,enco_hidden_size).cuda())
            deco_hid = Variable(torch.zeros(1,batch_size,deco_hidden_size).cuda())
            enco_hid = None
            #deco_hid = None

            enco_out1, enco_hid = encoder(inp, enco_hid)
            #print(deco_padding.size())
            #print('enco_out1 :',enco_out1.size())
            #print('deco_hid :',deco_hid.size())
            #print('enco_hid :',enco_hid[0].size())
            #deco_out, deco_emb, deco_hid = decoder(enco_out1,deco_padding,deco_hid)
            #print('deco_out :',deco_out.size())
            #print('deco_emb :',deco_emb.size())
            #print('deco_hid :',deco_hid[0].size())
            embed = Variable(torch.zeros(x_size[0],1,embed_size).cuda())

            loss = 0.0
            #enco_padding = Variable(torch.zeros(x_size[0],enco_input_size).long().cuda())
            enco_padding = Variable(torch.zeros(x_size[0],sent_limit).long().cuda())
            sentence = []
            sent_ind = random.randint(0,batch_size-1)
            if random.random() < 0.5:
                for i in range(y_size[1]):
                    #enco_out2, enco_hid = encoder(enco_padding, enco_hid)
                    #print('enco_out2 :',enco_out2.size())
                    #print('embed :',embed.size())
                    #exit()
                    #print(enco_out1.size())
                    #print('embed_size',embed.size())
                    deco_out2, embed, deco_hid = decoder(enco_out1,embed.view(1,x_size[0],embed_size),deco_hid)
                    
                    embed = decoder.embed(oup[:,i]).view(1,-1,deco_hidden_size)
                    sentence.append(torch.max(deco_out2[sent_ind].view(1,-1),1)[1].data[0])
                    loss+=criterion(deco_out2,oup[:,i])
            else:
                for i in range(y_size[1]):
                    #enco_out2, enco_hid = encoder(enco_padding, enco_hid)
                    #print('enco_out2 :',enco_out2.size())
                    #print('embed :',embed.size())
                    #exit()
                    deco_out2, embed, deco_hid = decoder(enco_out1,embed.view(1,x_size[0],embed_size),deco_hid)
                    sentence.append(torch.max(deco_out2[sent_ind].view(1,-1),1)[1].data[0])
                    loss+=criterion(deco_out2,oup[:,i])
            #print(sentence)
            #exit()
            total+=loss.data[0]
            if step_count % 100 == 0:
                print('[epoch:{}], [step:{}] : {}'.format(epoch, step_count,loss.data[0]/y_size[1]),file = post)
                print("Q mao's output :",index2sent(sentence, index2word),file = post)
                print("Sample input :",index2sent(inp.cpu().data[sent_ind].numpy(), index2word),file = post)
                print("Sample output :",index2sent(oup.cpu().data[sent_ind].numpy(), index2word),file = post)
                post.flush()
            step_count+=1
            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.25)
            enco_opt.step()
            deco_opt.step()
            #exit()
        if (epoch+1) % 1 == 0:
            torch.save({'encoder': encoder.state_dict(),'decoder': decoder.state_dict()},'save/'+model_name)
if __name__ == "__main__":
    main(sys.argv)
