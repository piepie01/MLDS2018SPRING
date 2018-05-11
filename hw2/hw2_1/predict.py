import sys
import os
import numpy as np
import copy

import random
import pickle

import torch.nn.functional as functional
import torch
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.nn as nn
from ultra_model import Encoder, Decoder


online = 0
post = sys.stdout

sent_limit = 50
batch_size = 16
embed_size = 512
enco_input_size = 4096
enco_hidden_size = 256
deco_input_size = 1024
deco_hidden_size = 512
learning_rate = 1e-3
model_name = 'model8.th'
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
def read_test_X(path):
    with open(os.path.join(path,'id.txt')) as f:
        name = f.read().split('\n')[:-1]
        video_name = [item for item in name]
        train_X = [os.path.join(path,'feat',item) for item in name]
    return train_X, video_name
def main(argv):
    test_X, test_video_name = read_test_X(argv[1])
    with open('save/word2index.pickle','rb') as f:
        word2index = pickle.load(f)
    with open('save/index2word.pickle','rb') as f:
        index2word = pickle.load(f)
    #exit()
    test_clv = test_clr_video(test_X, test_video_name, word2index)
    test_data_loader = DataLoader(test_clv, batch_size = 1)
    model = torch.load('save/'+model_name)
    encoder = Encoder(enco_input_size, enco_hidden_size).cuda()
    decoder = Decoder(deco_input_size, deco_hidden_size, len(word2index)).cuda()
    encoder.load_state_dict(model['encoder'])
    decoder.load_state_dict(model['decoder'])
    predict_file = open(argv[2],'w') 
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
    predict_file.close()

if __name__ == "__main__":
    main(sys.argv)
