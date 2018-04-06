import matplotlib.pyplot as plt
import numpy as np
import sys
x = []
y = []
with open('loss_ratio','r') as f:
    da = f.read().split('\n')[:-1]
    for i in range(len(da)):
        da[i] = da[i].split()
    for i in da:
        if float(i[0]) < 10:
            x.append(i[0])
            y.append(i[1])
x = np.array(x).astype(float)
y = np.array(y).astype(float)
plt.scatter(y,x)
plt.xlabel('minimal ratio')
plt.ylabel('loss')
#plt.xlim(-5.0,5.0)
#plt.ylim(-5.0,5.0)
plt.show()
