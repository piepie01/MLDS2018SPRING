import sys
import os
import json
from termcolor import cprint
def read_X(path):
    with open(os.path.join(path,'training_data','training_id.txt')) as f:
        name = f.read().split('\n')[:-1]
        video_name = [item for item in name]
        train_X = [os.path.join(path,'training_data','feat',item) for item in name]
    return train_X, video_name
def read_test_X(path):
    with open(os.path.join(path,'testing_data','id.txt')) as f:
        name = f.read().split('\n')[:-1]
        video_name = [item for item in name]
        train_X = [os.path.join(path,'testing_data','feat',item) for item in name]
    return train_X, video_name
def read_Y(path):
    with open(os.path.join(path,'training_label.json')) as f:
        label = json.load(f)
    return label
def word2vec(data,post):
    word2index = {'<PAD>':0,'<BOS>':1, '<EOS>':2, '<UNK>':3}
    index2word = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
    count = {}
    data1 = []
    for dic in data:
        for line in dic['caption']:
            data1.append(line.replace(',','').replace('.','').replace('!',''))
    for line in data1:
        for word in line.strip().split():
            if word not in count.keys():
                count[word] = 0
            count[word]+=1
    tmp = sorted(count.items(), key=lambda x: x[1],reverse = True)
    collect_words = 5000
    cprint('Origin kinds of word : {}, Collected kinds of word : {}'.format(len(tmp),collect_words),'cyan',attrs = ['bold'])
    for item,ind in zip(tmp[:collect_words],list(range(4,collect_words+4))):
        word2index[item[0]] = ind
        index2word[ind] = item[0]
    #print(word2index)
    #print(index2word)

    #print(tmp[:50])
    return word2index, index2word
if __name__ == "__main__":
    train_X, video_name = read_X(sys.argv[1])
    train_Y = read_Y(sys.argv[1])
    print(train_Y)
    w2i, i2w = word2vec(train_Y, sys.stdout)

