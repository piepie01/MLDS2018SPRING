import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
path_x = []
path_y = []
xx = []
yy = []
ll = []
for k in range(1):
    x = []
    y = []
    loss = []
    with open(str(k),'r') as f:
        da = f.read().split('\n')[:-1]
        for i in range(len(da)):
            da[i] = da[i].split()
        for i in da:
            if i[2] == '0.0':
                path_x.append(i[0])
                path_y.append(i[1])
            else:
                x.append(i[0])
                y.append(i[1])
                loss.append(i[2])
    xx = np.array(x).astype(float)
    yy = np.array(y).astype(float)
    ll = np.array(loss).astype(float)
path_x = np.array(path_x).astype(float)
path_y = np.array(path_y).astype(float)
print(path_x)
print(path_y)
plt.plot(path_x,path_y,'bo',c = 'y')
plt.plot(path_x,path_y,c = 'y')
plt.scatter(xx,yy,c = ll)
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim(-5.0,5.0)
#plt.ylim(-5.0,5.0)
plt.show()
