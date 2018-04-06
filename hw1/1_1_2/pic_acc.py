import matplotlib.pyplot as plt
import numpy as np
index = []
shallow = []
deep = []
medium = []
with open('acc/mnist_acc_shallow.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        index.append(i[0])
        shallow.append(i[1])
with open('acc/mnist_acc_medium.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        medium.append(i[1])

with open('acc/mnist_acc_deep.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        deep.append(i[1])

index = np.array(index)
shallow = np.array(shallow)
deep = np.array(deep)
medium = np.array(medium)

index = index.astype('float')
shallow = shallow.astype('float')
deep = deep.astype('float')
medium = medium.astype('float')
print(deep)
#for i in range(len(shallow)):
#    shallow[i] = 5.0**shallow[i]
#for i in range(len(deep)):
#    deep[i] = 5.0**deep[i]


#print(index)
#print(shallow)

plt.figure()
plt.title('Acc of three models')
plt.xlabel('Epoch_num')
plt.ylabel('acc')
plt.plot(index, shallow,label = 'shallow')
plt.plot(index, deep, color='red',label = 'deep')
plt.plot(index, medium, color='yellow',label = 'medium')
plt.legend()
plt.show()

