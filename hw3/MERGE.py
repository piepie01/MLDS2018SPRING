import numpy as np
import skimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
from skimage import io
from PIL import Image


if sys.argv[1] is 'C':
    datap = 'Cans'
    outp = 'cgan'
else:
    datap = 'ans'
    outp = 'gan'


fig, axs = plt.subplots(5,5)
for i in range(25):
    image = np.load('samples/'+datap+str(i)+'.npy')
    image = Image.fromarray(((image+1)*127.5).astype(np.uint8))
    axs[i//5,i%5].imshow(image)
    axs[i//5,i%5].axis('off')
fig.savefig('samples/'+outp+'.png')
plt.close()
