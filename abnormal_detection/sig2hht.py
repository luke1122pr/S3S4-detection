from hht_libv2 import hht
import numpy as np


def hht_func(set,t):
    freq,amp=hht(set[0],t)
    amp_max_value=np.amax(amp[:9], axis=0)
    amp_max_index=np.argmax(amp[:9], axis=0)
    freq_seg=freq[:9]
    Imf_freq_max_total=np.zeros((1,10000))
    Imf_freq_max_=freq_seg[amp_max_index]
    for i in range(Imf_freq_max_.shape[1]-1):
        Imf_freq_max_total[0][i]=Imf_freq_max_[i][i]
    for i in range(1,set.shape[0]):
        
        freq,amp=hht(set[i],t)
        amp_max_value=np.amax(amp[:9], axis=0)
        amp_max_index=np.argmax(amp[:9], axis=0)
        freq_seg=freq[:9]
        Imf_freq_max=np.zeros((1,10000))
        Imf_freq_max_=freq_seg[amp_max_index]
        for i in range(Imf_freq_max_.shape[1]-1):
            Imf_freq_max[0][i]=Imf_freq_max_[i][i] 
        Imf_freq_max_total=np.append(Imf_freq_max_total,Imf_freq_max, axis=0)
        print('Imf_freq_max_total',Imf_freq_max_total.shape)
    return Imf_freq_max_total

big_data=np.load('big_data_signal_label.npy')

duration = 10.0
fs = 1000.0
samples = int(fs*duration)
t = np.arange(samples) / fs
train_set_signal=np.array((big_data[:,3:]), dtype=np.float)
#valid_set_signal=np.array((big_data[298:418,3:]), dtype=np.float)
#test_set_signal=np.array((big_data[418:,3:]), dtype=np.float)
Imf_freq_max_total=hht_func(train_set_signal,t)
print('Imf_freq_max_total',Imf_freq_max_total.shape)
np.save('hht_set_signal.npy', Imf_freq_max_total)