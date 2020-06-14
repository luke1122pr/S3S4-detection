from hht_libv2 import hht
import numpy as np

big_data=np.load('big_data_signal_label.npy')
print('big_data',big_data.shape,big_data[:5])
a=big_data[:,1:3]
b=np.zeros((595,1))
for i in range(big_data.shape[0]):
     '''   
     if(a[i,0]=='0.0' ):
         a[i,0]=0
     if(a[i,1]=='0.0'):
         a[i,1]=0
     if(a[i,0]=='1.0' ):
         a[i,0]=1
     if(a[i,1]=='1.0'):
         a[i,1]=1'''
     if(a[i,0]=='1.0' ):
         b[i]=1
     if(a[i,1]=='1.0'):
         b[i]=1
print(np.sum(b),b.shape,b[:5])  
'''a=np.array(big_data[:,1:3],dtype=np.int)
print(a.shape,a)
print(np.sum(a[:,0]))
print(np.sum(a[:,1]))'''