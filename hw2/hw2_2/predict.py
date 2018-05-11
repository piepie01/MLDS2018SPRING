import sys
import copy
import numpy as np
import random

import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
import pickle
from model import Encoder
from model import Decoder
enco_input_size = 10
enco_hidden_size = 500
embed_size = 1500
deco_hidden_size = 1000
deco_input_size = 0
sent_limit = 12
batch_size = 1
def index2sent(sent,index2word):
    sent1 = []
    for j in range(1,len(sent)):
        if j == 1:
            sent1.append(sent[0])
        if sent[j] != sent[j-1]:
            sent1.append(sent[j])
    l = []
    for i in sent1:
        if i != 0 and i != 1 and i != 2 and i != 3:
            l.append(index2word[i])
    return ' '.join(l)
class test_clr_conv(Dataset):
    def __init__(self,train_Y,word2index):
        self.Y = copy.copy(train_Y)
        self.word2index = word2index
    def __len__(self):
        return len(self.Y)
    def __getitem__(self,index):
        y = [0 for i in range(sent_limit)]
        y[0] = 1
        tmp = self.Y[index].split()[:10]
        for i,txt in enumerate(tmp):
            if txt in self.word2index.keys():
                y[i+1] = self.word2index[txt]
            else:
                y[i+1] = 3
            y[i+2] = 2
        y = torch.LongTensor(y).cuda()
        return y
def predict(inp, encoder, decoder, index2word):
    #deco_padding = Variable(torch.zeros(1,inp.size()[0],deco_hidden_size).cuda())
    enco_hid = None
    deco_hid = Variable(torch.zeros(1,1,deco_hidden_size).cuda())

    enco_out1, enco_hid = encoder(inp, enco_hid)
    #deco_out, deco_emb, deco_hid = decoder(enco_out1, deco_padding,deco_hid)
    
    embed = Variable(torch.zeros(1,1,embed_size).cuda())
    enco_padding = Variable(torch.zeros(1,12).long().cuda())
    sentence = []
    for i in range(sent_limit):
        #enco_out2, enco_hid = encoder(enco_padding, enco_hid)
        deco_out2, embed, deco_hid = decoder(enco_out1, embed,deco_hid)
        sentence.append(torch.max(deco_out2[0].view(1,-1),1)[1].cpu().data[0])
        if sentence[-1] == 2:
            break
    response = index2sent(sentence, index2word)
    return response
def ultra_predict(inp, encoder, decoder, index2word):
    #deco_padding = Variable(torch.zeros(1,inp.size()[0],deco_hidden_size).cuda())
    enco_hid = None
    deco_hid = Variable(torch.zeros(1,1,deco_hidden_size).cuda())

    enco_out1, enco_hid = encoder(inp, enco_hid)
    #deco_out, deco_emb, deco_hid = decoder(enco_out1, deco_padding,deco_hid)
    
    embed = Variable(torch.zeros(1,1,embed_size).cuda())
    enco_padding = Variable(torch.zeros(1,12).long().cuda())
    sentence = []
    for i in range(sent_limit):
        #enco_out2, enco_hid = encoder(enco_padding, enco_hid)
        deco_out2, embed, deco_hid = decoder(enco_out1, embed,deco_hid)
        if i == 1:
            sentence.append(inp[0][1].cpu().data[0])
            embed = decoder.embed(inp[:,i]).view(1,-1,embed_size)
        else:
            sentence.append(torch.max(deco_out2[0].view(1,-1),1)[1].cpu().data[0])
        if sentence[-1] == 2:
            break
    response = index2sent(sentence, index2word)
    return response
def main(argv):
    p_file = open(argv[2],'w')
    with open('save/word7.pickle','rb') as f:
        word2index = pickle.load(f)
    with open('save/index7.pickle','rb') as f:
        index2word = pickle.load(f)
    model = torch.load('save/model7.th')
    encoder = Encoder(enco_input_size, enco_hidden_size, len(word2index),embed_size).cuda()
    decoder = Decoder(deco_input_size, deco_hidden_size, len(word2index), sent_limit,embed_size).cuda()
    encoder.load_state_dict(model['encoder'])
    decoder.load_state_dict(model['decoder'])
    with open(argv[1],'r') as f:
        txt = f.read().split('\n')[:-1]
    conv = test_clr_conv(txt,word2index)
    data_loader = DataLoader(conv, batch_size = batch_size)
    cnt = 0
    for x,q in zip(data_loader,txt):
        inp = Variable(x.view(1,-1))
        response = predict(inp,encoder, decoder, index2word)
        if len(response) == 0 or len(response.split()) == 1 or len(response.split()) == 2 and response.split()[1] == '的' or len(response.split()) == 2 and response.split()[1] == '!':
            response = ultra_predict(inp,encoder, decoder, index2word)
        if len(response) == 0:
            print('我',file = p_file)
        else:
            print(response,file = p_file)
        print(cnt,end = '\r')
        cnt+=1
    print()
    p_file.close()

if __name__ == "__main__":
    main(sys.argv)
