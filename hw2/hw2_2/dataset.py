import sys
from termcolor import cprint
def read_data(file_name,post):
    with open(file_name,'r') as f:
        tmp_set = f.read().split('+++$+++')[:-1]
    data_set = [item.split('\n')[:-1] for item in tmp_set]
    train_X = []
    train_Y = []
    for para in data_set:
        for sent in para[:-1]:
            train_X.append(sent)
        for sent in para[1:]:
            train_Y.append(sent)
    cprint('Train_X length(before) : {}, Train_Y length(before) : {}'.format(len(train_X),len(train_Y)),'cyan',attrs = ['bold'])

    pre_X = []
    pre_Y = []
    for x,y in zip(train_X,train_Y):
        tmp_x = x
        tmp_y = y
        for non in ',"。♫「」f^N.[]':
            tmp_x = tmp_x.replace(non,'')
            tmp_y = tmp_y.replace(non,'')
        if 2 <= len(tmp_x.split()) <= 10 and 2 <= len(tmp_y.split()) <= 10:
            pre_X.append(tmp_x)
            pre_Y.append(tmp_y)
    train_X = pre_X
    train_Y = pre_Y

    cprint('Train_X length(after) : {}, Train_Y length(after) : {}'.format(len(train_X),len(train_Y)),'cyan',attrs = ['bold'])
    return train_X,train_Y
def word2vec(data,post):
    word2index = {'<PAD>':0,'<BOS>':1, '<EOS>':2, '<UNK>':3}
    index2word = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
    count = {}
    for line in data:
        for word in line.strip().split():
            if word not in count.keys():
                count[word] = 0
            count[word]+=1
    tmp = sorted(count.items(), key=lambda x: x[1],reverse = True)
    collect_words = 50000
    cprint('Origin kinds of word : {}, Collected kinds of word : {}'.format(len(tmp),collect_words),'cyan',attrs = ['bold'])
    for item,ind in zip(tmp[:collect_words],list(range(4,collect_words+4))):
        word2index[item[0]] = ind
        index2word[ind] = item[0]
    #print(word2index)
    #print(index2word)

    #print(tmp[:50])
    return word2index, index2word

if __name__ == "__main__":
    X,Y = read_data(sys.argv[1],sys.stdout)
    for i in range(1000):
        print(X[i],Y[i],sep = '********')
#    #print(X,Y)
#    word2vec(X)
#    pass
