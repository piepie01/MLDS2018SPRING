import matplotlib.pyplot as plt
import numpy as np
def get(filename):
    one = []
    two = []
    with open(filename,'r') as f:
        li = f.read().split('\n')[:-1]
        for i in range(len(li)):
            li[i] = li[i].split()
        for i in li[::87]:
            one.append(i[0])
            two.append(i[1])
    return [np.array(one).astype(float),np.array(two).astype(float)]
def get_acc(filename):
    one = []
    with open(filename,'r') as f:
        li = f.read().split('\n')[:-1]
        for i in li[::87]:
            i = format(float(i), '.2f')
            one.append(i)
    return np.array(one).astype(float)
data = []
for i in range(8):
    tmp = get('weight/layer_'+str(i))
    data.append(tmp)
acc = []
for i in range(8):
    tmp = get_acc('loss/'+str(i))
    acc.append(tmp)




plt.figure()
plt.title('Weight')
#plt.xlabel('Epoch_num')
#plt.ylabel('loss')
size = 2
color_list = ['black','red','green','yellow','orange','pink','blue','purple']
plt.plot(data[0][0], data[0][1],color = 'black')
plt.plot(data[1][0], data[1][1],color = 'red')
plt.plot(data[2][0], data[2][1],color = 'green')
plt.plot(data[3][0], data[3][1],color = 'yellow')
plt.plot(data[4][0], data[4][1],color = 'orange')
plt.plot(data[5][0], data[5][1],color = 'pink')
plt.plot(data[6][0], data[6][1],color = 'blue')
plt.plot(data[7][0], data[7][1],color = 'purple')
for i in range(8):
    for j,txt in enumerate(acc[i]):
        plt.annotate(txt,(data[i][0][j],data[i][1][j]),color = color_list[i],size = 7)
#plt.scatter(data[6][0], data[6][1],color = 'blue',s = size)
#plt.scatter(data[7][0], data[7][1],color = 'purple',s = size)
#plt.plot(index, deep, color='red',label = 'deep')
#plt.plot(index, medium, color='yellow',label = 'medium')
plt.legend()
plt.show()

