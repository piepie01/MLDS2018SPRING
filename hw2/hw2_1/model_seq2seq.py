import sys
import os
import numpy as np
import copy
import dataset

import random
import pickle

import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
from ultra_model import Encoder, Decoder


online = 1
post = sys.stdout
if online == 1:
    post = open('/home/piepie01/public_html/123/output1','w')

sent_limit = 50
batch_size = 48
enco_input_size = 4096
enco_hidden_size = 256
deco_input_size = 1024
deco_hidden_size = 512
embed_size = 512
learning_rate = 1e-3
epochs = 30
model_name = 'model8.th'
class clr_video(Dataset):
    def __init__(self,train_X, train_Y, video_name, word2index):
        self.X = copy.copy(train_X)
        self.Y = copy.copy(train_Y)
        self.v_name = video_name
        self.word2index = word2index
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        x = np.load(self.X[index]+'.npy')
        x = torch.FloatTensor(x).cuda()
        name = self.v_name[index]
        if name != self.Y[index]['id']:
            raise IndexError('11')
        
        ind = random.randint(0,len(self.Y[index]['caption'])-1)
        y = [0 for i in range(sent_limit)]
        y[0] = 1
        #print(len(self.Y), index)
        #print(self.Y[index]['caption'][ind])
        for i,txt in enumerate(self.Y[index]['caption'][ind].replace(',','').replace('.','').replace('!','').split()[:sent_limit-2]):
            if txt in self.word2index.keys():
                y[i+1] = self.word2index[txt]
            else:
                y[i+1] = 3
            y[i+2] = 2
        y = torch.LongTensor(y).cuda()
        return x,y
class test_clr_video(Dataset):
    def __init__(self,train_X, video_name, word2index):
        self.X = copy.copy(train_X)
        self.v_name = video_name
        self.word2index = word2index
    def __len__(self):
        return len(self.X)
    def __getitem__(self,index):
        x = np.load(self.X[index]+'.npy')
        x = torch.FloatTensor(x).cuda()
        return x, self.v_name[index]
def index2sent(sent,index2word):
    l = []
    for i in sent:
        if i > 2:
            l.append(index2word[i])
    if len(l) == 0:
        l.append('a')
    return ' '.join(l)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main(argv):
    train_X, video_name = dataset.read_X(argv[1])
    test_X, test_video_name = dataset.read_test_X(argv[1])
    train_Y = dataset.read_Y(argv[1])
    word2index, index2word = dataset.word2vec(train_Y, post)
    with open('save/word2index.pickle','wb') as f:
        pickle.dump(word2index,f)
    with open('save/index2word.pickle','wb') as f:
        pickle.dump(index2word,f)
    #exit()
    clv = clr_video(train_X, train_Y, video_name, word2index)
    test_clv = test_clr_video(test_X, test_video_name, word2index)
    data_loader = DataLoader(clv, shuffle = True, batch_size = batch_size)
    test_data_loader = DataLoader(test_clv, batch_size = 1)
    encoder = Encoder(enco_input_size, enco_hidden_size).cuda()
    decoder = Decoder(deco_input_size, deco_hidden_size, len(word2index)).cuda()

    print("Encoder's parameters :",count_parameters(encoder),file = post)
    print("Decoder's parameters :",count_parameters(decoder), file = post)
    criterion = nn.NLLLoss().cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    enco_opt = torch.optim.Adam(encoder.parameters(), lr = learning_rate)
    deco_opt = torch.optim.Adam(decoder.parameters(), lr = learning_rate)
    
    for epoch in range(epochs):
        step_count = 0
        for x,y in data_loader:
            #print(x.size())
            enco_opt.zero_grad()
            deco_opt.zero_grad()
            inp = Variable(x)
            oup = Variable(y)
            x_size = inp.size()
            y_size = oup.size()
            enco_hid = None
            deco_hid = None
            enco_out1, enco_hid = encoder(inp, enco_hid)
            
            deco_padding = Variable(torch.zeros(x_size[0],80,deco_hidden_size).cuda())
            deco_out1, _, deco_hid = decoder(torch.cat((enco_out1, deco_padding),2),deco_hid)
            sentence = []
            sent_ind = random.randint(0,x_size[0]-1)
            embed = Variable(torch.zeros(x_size[0],1,deco_hidden_size).cuda())
            loss = 0.0
            enco_padding = Variable(torch.zeros(x_size[0],1,4096).cuda())
            for i in range(y_size[1]):
                enco_out2, enco_hid = encoder(enco_padding, enco_hid)
                deco_out2, embed, deco_hid = decoder(torch.cat((enco_out2, embed),2),deco_hid)
                if random.uniform(0,1) < 0.6:
                    embed = decoder.embed(oup[:,i]).view(-1,1,deco_hidden_size)
                sentence.append(torch.max(deco_out2[sent_ind].cpu().view(1,-1),1)[1].data[0])
                loss+=criterion(deco_out2,oup[:,i])
            if step_count % 10 == 0:
                print('[epoch:{}], [step:{}] : {}'.format(epoch, step_count,loss.data[0]/y_size[1]),file = post)
            loss.backward()
            torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.25)
            enco_opt.step()
            deco_opt.step()
            if step_count%10==0:
                print("Q mao's output :",index2sent(sentence, index2word),file = post)
                #print("Sample input :",index2sent(inp.cpu().data[sent_ind].numpy(), index2word),file = post)
                print("Sample output :",index2sent(oup.cpu().data[sent_ind].numpy(), index2word),file = post)
                post.flush()
            step_count+=1
        '''
        predict_file = open('ans.txt','w')
        for x,name in test_data_loader:
            #print(name)
            inp = Variable(x)
            x_size = inp.size()
            enco_hid = None
            deco_hid = None
            enco_out1, enco_hid = encoder(inp, enco_hid)
            
            deco_padding = Variable(torch.zeros(x_size[0],80,deco_hidden_size).cuda())
            deco_out1, _, deco_hid = decoder(torch.cat((enco_out1, deco_padding),2),deco_hid)
            sentence = []
            sent_ind = 0
            embed = Variable(torch.zeros(x_size[0],1,deco_hidden_size).cuda())
            enco_padding = Variable(torch.zeros(x_size[0],1,4096).cuda())
            for i in range(sent_limit):
                enco_out2, enco_hid = encoder(enco_padding, enco_hid)
                deco_out2, embed, deco_hid = decoder(torch.cat((enco_out2, embed),2),deco_hid)
                sentence.append(torch.max(deco_out2[sent_ind].cpu().view(1,-1),1)[1].data[0])
            print(name[0],index2sent(sentence, index2word),sep = ',',file = predict_file)
        predict_file.flush()
        predict_file.close()
        import subprocess
        bl = subprocess.check_output('python3 bleu_eval.py ../ans.txt', shell=True, cwd = 'MLDS_hw2_1_data/')
        print('----------------',float(bl.decode('ascii')[22:-1]),'----------------', file = post)
        score = float(bl.decode('ascii')[23:-1])
        if score > 0.66 and epoch > 20:
        '''
    torch.save({'encoder': encoder.state_dict(),'decoder': decoder.state_dict()},'save/'+model_name)

if __name__ == "__main__":
    main(sys.argv)
