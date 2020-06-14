import pickle
import numpy as np

import matplotlib.pyplot as plt
def normalize(X, means_and_stds=None):
        if means_and_stds is None:
            means = [ X[..., i].mean(dtype=np.float32) for i in range(X.shape[-1]) ]
            stds = [ X[..., i].std(dtype=np.float32) for i in range(X.shape[-1]) ]
        else:
            means = means_and_stds[0]
            stds = means_and_stds[1]

        normalized_X = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[-1]):
            normalized_X[..., i] = X[..., i].astype(np.float32) - means[i]
            normalized_X[..., i] = normalized_X[..., i] / stds[i]
        return normalized_X, (means, stds)
def onehot(set):
    re=[]
    
    for i in range (set.shape[0]):
        single=[]
        if(set[i][0]==1):
            single.append(1) 
            single.append(0)    
        else:
            single.append(0) 
            single.append(1)
        '''if(set[i][1]==1):
            single.append(1) 
            single.append(0)     
        else:
            single.append(0) 
            single.append(1)'''
        re.append(single)
    return np.array(re, dtype=np.int)  
big_data=np.load('big_data_signal_label.npy')
#print('big_data',big_data.shape,big_data[:5])

train_set_signal=np.array((big_data[:298,3:]), dtype=np.float)
valid_set_signal=np.array((big_data[298:418,3:]), dtype=np.float)
test_set_signal=np.array((big_data[418:,3:]), dtype=np.float)
train_set_label=np.array((big_data[:298,1:3]), dtype=np.float)
valid_set_label=np.array((big_data[298:418,1:2]), dtype=np.float)
test_set_label=np.array((big_data[418:,1:2]), dtype=np.float)
#print('*',train_set_label[:5],train_set_label.shape)
#train_set_label=onehot(train_set_label)
#print(train_set_label[:5],train_set_label.shape)
valid_set_label=onehot(valid_set_label)
test_set_label=onehot(test_set_label)

train_set_signal=train_set_signal[:,:,np.newaxis]
valid_set_signal=valid_set_signal[:,:,np.newaxis]
test_set_signal=test_set_signal[:,:,np.newaxis]

'''
train_set = combine(train_set_signal, train_set_label)
valid_set = combine(valid_set_signal, valid_set_label)
test_set = combine(test_set_signal, test_set_label)
'''
train_set = [np.array(train_set_signal), np.array(train_set_label)]
valid_set = [np.array(valid_set_signal), np.array(valid_set_label)]
test_set = [np.array(test_set_signal), np.array(test_set_label)]


fi=0
train_set[0], means_and_stds = normalize(train_set[0])
valid_set[0], _ = normalize(valid_set[0], means_and_stds)
test_set[0], _ = normalize(test_set[0], means_and_stds)
for i in range(5):
    plt.figure()
    plt.plot(range(10000),train_set[0][fi+i])
print(train_set_label[fi:fi+5])
plt.show()
