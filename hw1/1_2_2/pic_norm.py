import matplotlib.pyplot as plt
import numpy as np
import sys
def get(filename):
    one = []
    two = []
    with open(filename,'r') as f:
        li = f.read().split('\n')[:-1]
        for i in range(len(li)):
            li[i] = li[i].split()
        for i in li[::20]:
            one.append(i[0])
            two.append(i[1])
    return [np.array(one).astype(float),np.array(two).astype(float)]
def get_acc(filename):
    one = []
    with open(filename,'r') as f:
        li = f.read().split('\n')[:-1]
        for i in li:
            one.append(i)
    return np.array(one).astype(float)
norm = get_acc('gradient_norm/norm')
loss = get_acc('gradient_norm/loss')
ind = [i for i in range(600)]
norm = norm * 100
#print(norm)

plt.figure()
#plt.title('Weight')
plt.xlabel('Epoch_num')
if sys.argv[1] == 'loss':
    plt.ylabel('loss')
    plt.plot(ind,loss)
elif sys.argv[1] == 'norm':
    plt.ylabel('norm')
    plt.plot(ind,norm)
plt.legend()
plt.show()

