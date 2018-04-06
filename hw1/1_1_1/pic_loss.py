import matplotlib.pyplot as plt
import numpy as np
index = []
shallow = []
deep = []
medium = []
with open('result_data/shallow_loss.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        index.append(i[0])
        shallow.append(i[1])
with open('result_data/deep_loss.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        deep.append(i[1])
with open('result_data/medium_loss.txt','r') as f:
    l = f.read().split('\n')
    l = l[:len(l)-1]
    for i in range(len(l)):
        l[i] = l[i].split()
    for i in l:
        medium.append(i[1])
index = np.array(index)
shallow = np.array(shallow)
deep = np.array(deep)
medium = np.array(medium)

index = index.astype('float')
shallow = shallow.astype('float')
deep = deep.astype('float')
medium = medium.astype('float')

#for i in range(len(shallow)):
#    shallow[i] = 5.0**shallow[i]
#for i in range(len(deep)):
#    deep[i] = 5.0**deep[i]


#print(index)
#print(shallow)

plt.figure()
plt.title('Loss of three models')
plt.xlabel('Epoch_num')
plt.ylabel('loss')
plt.plot(index, shallow,label = 'shallow')
plt.plot(index, deep, color='red',label = 'deep')
plt.plot(index, medium, color='yellow',label = 'medium')
plt.legend()
plt.show()

