import matplotlib.pyplot as plt
import numpy as np

trainacc = np.loadtxt(open('interpolar_trainacc','r').read().split('\n'))
testacc = np.loadtxt(open('interpolar_testacc','r').read().split('\n'))
trainloss = np.loadtxt(open('interpolar_trainloss','r').read().split('\n'))
testloss = np.loadtxt(open('interpolar_testloss','r').read().split('\n'))

fig,ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(trainloss[:,0],trainloss[:,1],'-',color='blue',label='train',markersize=2)
ax1.plot(testloss[:,0],testloss[:,1],'--',color='blue',label='test',markersize=2)
ax2.plot(trainacc[:,0],trainacc[:,1],'-',color='red',label='sensitivity',markersize=2)
ax2.plot(testacc[:,0],testacc[:,1],'--',color='red',label='sensitivity',markersize=2)
ax1.legend(['train','test'])#,'sensitivity'])
plt.title('Interpolation_10~1000')
plt.xlabel('alpha')
ax1.set_ylabel('Loss',color='blue')
ax2.set_ylabel('Accuracy',color='red')
plt.show()
