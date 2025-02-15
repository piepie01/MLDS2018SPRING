import matplotlib.pyplot as plt
import numpy as np

trainacc = np.loadtxt(open('difapproach_trainacc','r').read().split('\n'))
testacc = np.loadtxt(open('difapproach_testacc','r').read().split('\n'))
trainloss = np.loadtxt(open('difapproach_trainloss','r').read().split('\n'))
testloss = np.loadtxt(open('difapproach_testloss','r').read().split('\n'))
sensitivity = np.loadtxt(open('difapproach_sensitivity','r').read().split('\n'))

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(trainacc[:,0],trainacc[:,1],'-o',color='blue',label='train')
ax1.plot(testacc[:,0],testacc[:,1],'--o',color='blue',label='test')
ax2.plot(sensitivity[:,0],sensitivity[:,1],'-o',color='red',label='sensitivity')
ax1.legend(['train','test'])#,'sensitivity'])
plt.title('Sensitivity_Accuracy')
plt.xlabel('Batch_Size')
ax1.set_ylabel('Accuracy',color='blue')
ax2.set_ylabel('Sensitivity',color='red')
plt.show()

fig,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(trainloss[:,0],trainloss[:,1],'-o',color='blue',label='train')
ax1.plot(testloss[:,0],testloss[:,1],'--o',color='blue',label='test')
ax2.plot(sensitivity[:,0],sensitivity[:,1],'-o',color='red',label='sensitivity')
ax1.legend(['train','test'])#,'sensitivity'])
plt.title('Sensitivity_Loss')
plt.xlabel('Batch_Size')
ax1.set_ylabel('Loss',color='blue')
ax2.set_ylabel('Sensitivity',color='red')
plt.show()
