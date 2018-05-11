import numpy as np
import random
import pickle

import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, 2, batch_first = True,bidirectional=True)
    def forward(self, seq, hid):
        out, hid = self.gru(seq, hid)
        return out, hid

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_len):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(word_len, hidden_size)
        self.lstm = nn.GRU(input_size, hidden_size, 2, batch_first = True)
        self.fc2 = nn.Linear(hidden_size, word_len)
        self.softmax = nn.LogSoftmax(dim = 1)
        self.word_len = word_len
        self.hidden_size = hidden_size
    def forward(self, seq, hid):
        out, hid = self.lstm(seq, hid)
        out = self.fc2(out)
        out = out.view(-1, self.word_len)
        out = self.softmax(out)
        emb = self.embed(torch.max(out, 1)[1])
        emb = emb.view(-1,1,self.hidden_size)
        return out, emb, hid
