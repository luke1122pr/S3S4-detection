import numpy as np

import matplotlib.pyplot as plt
'''
big_data=np.load('big_data_signal_label.npy')
#print('big_data',big_data.shape,big_data[:5])
a=big_data[:,1:3]
b=np.zeros((595,1))
for i in range(595):
    
    if(a[i,0]=='1.0' ):
        b[i]=1
    if(a[i,1]=='1.0'):
        b[i]=1
print(a[:5,0],a[:5,1])
print(a[:5],a.shape)
print(b[:5],b.shape,b.sum())'''

hht_set_signal=np.load('hht_set_signal.npy')
fi=0
for i in range(5):
    plt.figure()
    plt.plot(range(10000),hht_set_signal[fi+i])
plt.show()