import matplotlib.pyplot as plt
import numpy as np

trainacc = np.loadtxt(open('shuffle_trainacc','r').read().split('\n')[:1000])
testacc = np.loadtxt(open('shuffle_testacc','r').read().split('\n')[:1000])
trainloss = np.loadtxt(open('shuffle_trainloss','r').read().split('\n')[:1000])
testloss = np.loadtxt(open('shuffle_testloss','r').read().split('\n')[:1000])

x = []
for i in range(1000):
	x.append(i+1)

plt.scatter(x,trainacc,s=0.5,color='red',label='train')
plt.scatter(x,testacc,s=0.5,color='blue',label='test')
plt.legend(['train','test'])
plt.title('Random_Label_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.scatter(x,trainloss,s=0.5,color='red',label='train')
plt.scatter(x,testloss,s=0.5,color='blue',label='test')
plt.legend(['train','test'])
plt.title('Random_Label_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
