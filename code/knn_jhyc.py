import pandas as pd
import numpy as np
import csv as csv
import pdb
import random
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import metrics

from scipy import signal

def threshold(im):
    median = np.median(im)
    return x.ix[random.sample(x.index, n)]

train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)


# as array
values = train_df.drop(['label'],axis=1).values
reval = np.reshape(values, (1176000,28)) # ==42000 images
#reval = np.reshape(values, (28,1176000))
plt.ion() # turn on interactive mode, non-blocking `show`


#
test_label = np.zeros(len(test_df))
#for j in test_df.index:
for j in [0,3]:
    im_test = np.reshape(test_df.ix[j,0:], (28,28))
#plt.imshow(im_test)
    conv_arr = np.zeros(len(train_df))
    for idx in train_df.index:
        im_train = reval[idx*28:idx*28+27,0:27]
        #im = np.reshape(train_df.ix[idx,1:785], (28,28)
        #conv = signal.fftconvolve(im_test,im_train)
        #conv = signal.correlate2d(im_test,im_train)
        max_conv = np.max(conv)
        conv_arr[idx]=max_conv

    avg_score = np.zeros(10)
    for k in range(0,10):np
        avg_score[k] = np.mean(conv_arr[train_df[train_df.label == k].index])
    pdb.set_trace()

# show all images of 0, etc.
for n in range(0,9):
    idx_n = train_df[train_df.label==n].index
    for idx in idx_n:
        im = reval[idx*28:idx*28+27,0:27]
        #im = np.reshape(train_df.ix[idx,1:785], (28,28))
        pdb.set_trace()
        plt.imshow(im)
        plt.show()
        #_ = raw_input("Press [enter] to continue.") # wait for input from the use


        
imfft = np.fft.fft2(im)
        # Now shift so that low spatial frequencies are in the center.
imfft_shift = np.fft.fftshift( imfft )
plt.imshow(np.log(np.abs(imfft_shift)))
_ = raw_input("Press [enter] to continue.") # wait for input from the use
        
rows, cols = im.shape
crow, ccol = rows/2, cols/2
imfft_shift[crow-4:crow+4, ccol-4:ccol+4] = 0.
imfft_ishift = np.fft.ifftshift(imfft_shift)
img_new = np.fft.ifft2(imfft_ishift)
img_new = np.abs(img_new)
plt.imshow(img_new)
_ = raw_input("Press [enter] to continue.") # wait for input from the use
    
pdb.set_trace()

plt.imshow(train_df.drop(['label'],axis=1).values)
plt.gray()
plt.show()

# replace pixel[0,n] with pixel[0..27,0..27,..]
# then we can stack them
tmp = map(str,range(0,28)*28)
tmp = ['col'+s for s in tmp]
colID = ['label']
colID.extend(tmp)

