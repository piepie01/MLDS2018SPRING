import numpy as np
import random
import pickle

import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
import sys

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_len,embed_size):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(word_len, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, 1, batch_first = True,bidirectional=True)
    def forward(self, seq, hid):
        out = self.embed(seq)
        out, hid = self.gru(out, hid)
        return out, hid

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_len, sent_limit,embed_size):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(word_len, embed_size)
        self.lstm = nn.GRU(hidden_size//2, hidden_size, 1, batch_first = True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, word_len)

        self.attn = nn.Linear(hidden_size + embed_size, sent_limit)
        self.attn_com = nn.Linear(hidden_size + embed_size, hidden_size//2)

        self.sent_limit = sent_limit
        self.softmax = nn.LogSoftmax(dim = 1)
        self.word_len = word_len
        self.hidden_size = hidden_size
        self.embed_size = embed_size
    def forward(self, enco_out, emb, hid):
        attn_weight = torch.cat((emb, hid),2) #[1, 40, 800]
        #print(attn_weight.size())
        attn_weight = self.attn(attn_weight)
        attn_weight = functional.softmax(attn_weight,dim = 2) #[1, 40, 12]

#        print(attn_weight.size())
#        print(enco_out.size())
        attn_applied = torch.bmm(attn_weight.view(-1,1,self.sent_limit), enco_out) #torch.Size([40, 1, 400])

        #print('in decoder :',attn_applied.size())
        #exit()
        out = torch.cat((emb.view(-1,1,self.embed_size), attn_applied),2)
        out = self.attn_com(out)
        out = functional.relu(out) #[40,1,400]
        #print('in decoder :',out.size())


        out, hid = self.lstm(out, hid)
        #out = self.fc1(out)
        #out = functional.relu(out)
        out = self.fc2(out)
        out = out.view(-1, self.word_len)
        out = self.softmax(out)
        emb = self.embed(torch.max(out, 1)[1])
        emb = emb.view(-1,1,self.embed_size)
        #print(emb.size())
        #exit()
        return out, emb, hid

