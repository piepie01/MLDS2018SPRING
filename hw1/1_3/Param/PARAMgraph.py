import matplotlib.pyplot as plt
import numpy as np

trainacc = np.loadtxt(open('difparam_trainacc','r').read().split('\n'))
testacc = np.loadtxt(open('difparam_testacc','r').read().split('\n'))
trainloss = np.loadtxt(open('difparam_trainloss','r').read().split('\n'))
testloss = np.loadtxt(open('difparam_testloss','r').read().split('\n'))

plt.scatter(trainloss[:,0],trainloss[:,1]*100,s=0.5,color='red',label='train')
plt.scatter(testloss[:,0],testloss[:,1]*100,s=0.5,color='blue',label='test')
plt.legend(['train','test'])
plt.title('Parameters_to_Loss')
plt.xlabel('Parameters')
plt.ylabel('Loss')
plt.show()

plt.scatter(trainacc[:,0],trainacc[:,1]*100,s=0.5,color='red',label='train')
plt.scatter(testacc[:,0],testacc[:,1]*100,s=0.5,color='blue',label='test')
plt.legend(['train','test'])
plt.title('Parameters_to_Accuracy')
plt.xlabel('Parameters')
plt.ylabel('Accuracy')
plt.show()
