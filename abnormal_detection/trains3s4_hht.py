#!/usr/bin/env python3
import pickle
import numpy as np
import better_exceptions; better_exceptions.hook()
import os
os.environ['TF_KERAS'] = '1'


import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()
from ekg.utils.train_utils import set_wandb_config

# for loging result
import wandb
from wandb.keras import WandbCallback

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_radam import RAdam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from ekg.callbacks import LogBest

from ekg.utils import data_utils
from ekg.utils.data_utils import BaseDataGenerator
from ekg.utils.datasets import BigExamLoader, Audicor10sLoader
import evaluation
from evaluation import print_statistics

from ekg.models.backbone import backbone

def preprocessing(dataloader):
    # make ys one-hot
    dataloader.normal_y = keras.utils.to_categorical(dataloader.normal_y, num_classes=2, dtype=np.int)
    dataloader.abnormal_y = keras.utils.to_categorical(dataloader.abnormal_y, num_classes=2, dtype=np.int)

class AbnormalBigExamLoader(BigExamLoader):
    def load_abnormal_y(self):
        return np.ones((self.abnormal_X.shape[0], ))
    def load_normal_y(self):
        return np.zeros((self.normal_X.shape[0], ))

class AbnormalAudicor10sLoader(Audicor10sLoader):
    def load_abnormal_y(self):
        return np.ones((self.abnormal_X.shape[0], ))
    def load_normal_y(self):
        return np.zeros((self.normal_X.shape[0], ))
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

      
def train():
    dataloaders = list()
    if 'big_exam' in wandb.config.datasets:
        dataloaders.append(AbnormalBigExamLoader(wandb_config=wandb.config))
    if 'audicor_10s' in wandb.config.datasets:
        dataloaders.append(AbnormalAudicor10sLoader(wandb_config=wandb.config))

    g = BaseDataGenerator(dataloaders=dataloaders,
                            wandb_config=wandb.config,
                            preprocessing_fn=preprocessing)
    train_set, valid_set, test_set = g.get()
    big_data=np.load('big_data_signal_label.npy')
    hht_set_signal=np.load('./abnormal_detection/hht_set_signal.npy')
    hht_set_signal=abs(hht_set_signal)
    print('hht_set_signal',hht_set_signal.shape,hht_set_signal[:5])
    a=big_data[:,1:3]
    b=np.zeros((595,1))
    for i in range(595):
        
        if(a[i,0]=='1' ):
            b[i]=1
        if(a[i,1]=='1'):
            b[i]=1
    #print(a[:5,0],a[:5,1])
    #print(a[:5],a.shape)
    #print(b[:5],b.shape)
    duration = 10.0
    fs = 1000.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs
    train_set_signal=np.array((hht_set_signal[:298]), dtype=np.float)
    valid_set_signal=np.array((hht_set_signal[298:418]), dtype=np.float)
    test_set_signal=np.array((hht_set_signal[418:]), dtype=np.float)
    
    #train_set_label=np.array((big_data[:298,1:2]), dtype=np.float)
    #valid_set_label=np.array((big_data[298:418,1:2]), dtype=np.float)
    #test_set_label=np.array((big_data[418:,1:2]), dtype=np.float)
    
    train_set_label=np.array((b[:298]), dtype=np.float)
    valid_set_label=np.array((b[298:418]), dtype=np.float)
    test_set_label=np.array((b[418:]), dtype=np.float)
    print('test_set_label',np.sum(test_set_label),test_set_label.shape)
    #print('*',train_set_label[:5],train_set_label.shape)
    train_set_label=onehot(train_set_label)
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
    
    train_set[0], means_and_stds = normalize(train_set[0])
    valid_set[0], _ = normalize(valid_set[0], means_and_stds)
    test_set[0], _ = normalize(test_set[0], means_and_stds)
        
    
    #print(test_set_label[0])
    #print(valid_set[0][0][:5])
    print('have',train_set[1][:, 0].sum()/ train_set[1][:, 0].shape[0],valid_set[1][:, 0].sum()/ valid_set[1][:, 0].shape[0],test_set[1][:, 0].sum()/ test_set[1][:, 0].shape[0])
    print('have',train_set[1][:, 1].sum()/ train_set[1][:, 0].shape[0],valid_set[1][:, 1].sum()/ valid_set[1][:, 0].shape[0],test_set[1][:, 1].sum()/ test_set[1][:, 0].shape[0])

    # save means and stds to wandb
    #with open(os.path.join(wandb.run.dir, 'means_and_stds.pl'), 'wb') as f:
        #pickle.dump(g.means_and_stds, f)

    model = backbone(wandb.config, include_top=True, classification=True, classes=2)
    model.compile(RAdam(1e-4) if wandb.config.radam else Adam(amsgrad=True), 
                    'binary_crossentropy', metrics=['acc'])
    model.summary()
    wandb.log({'model_params': model.count_params()}, commit=False)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        # ReduceLROnPlateau(patience=10, cooldown=5, verbose=1),
        LogBest(),
        WandbCallback(log_gradients=False, training_data=train_set),
    ]

    model.fit(train_set[0], train_set[1], batch_size=64, epochs=250, validation_data=(valid_set[0], valid_set[1]), callbacks=callbacks, shuffle=True)
    model.save(os.path.join(wandb.run.dir, 'final_model.h5'))

    # load best model from wandb and evaluate
    print('Evaluate the BEST model!')

    from tensorflow.keras.models import load_model
    from ekg.layers import LeftCropLike, CenterCropLike
    from ekg.layers.sincnet import SincConv1D

    custom_objects = {
        'SincConv1D': SincConv1D,
        'LeftCropLike': LeftCropLike, 
        'CenterCropLike': CenterCropLike
    }

    model = load_model(os.path.join(wandb.run.dir, 'model-best.h5'),
                        custom_objects=custom_objects, compile=False)

    evaluation.evaluation(model, test_set)

if __name__ == '__main__':
    wandb.init(project='', entity='')

    # search result
    set_wandb_config({
        # model
        'sincconv_filter_length': 31,
        'sincconv_nfilters': 8,

        'branch_nlayers': 1,

        'ekg_kernel_length': 35,
        'hs_kernel_length': 35,

        'ekg_nfilters': 1,
        'hs_nfilters': 1,

        'final_nlayers': 5,
        'final_kernel_length': 21,
        'final_nonlocal_nlayers': 0,
        'final_nfilters': 8,
        
        'prediction_nlayers': 3,
        'prediction_kernel_length': 5,
        'prediction_nfilters': 8,

        'batch_size': 128,


        'kernel_initializer': 'glorot_uniform',
        'skip_connection': False,
        'crop_center': False,
        'se_block': True,
        
        'prediction_head': False,
        

        'radam': True,

        # data
        'remove_dirty': 2, # deprecated, always remove dirty data
        'datasets': ['big_exam'], # 'big_exam', 'audicor_10s'

        'big_exam_ekg_channels': [], # [0, 1, 2, 3, 4, 5, 6, 7],
        'big_exam_hs_channels': [8],
        'big_exam_only_train': False,

        'audicor_10s_ekg_channels': [0],
        'audicor_10s_hs_channels': [1],
        'audicor_10s_only_train': False,

        'downsample': 'direct', # average
        'with_normal_subjects': True,
        'normal_subjects_only_train': False,

        'tf': 2.2

    }, include_preprocessing_setting=True)

    set_wandb_config({
        'sampling_rate': 500 if 'audicor_10s' in wandb.config.datasets else 1000,
        'n_ekg_channels': data_utils.calculate_n_ekg_channels(wandb.config),
        'n_hs_channels': data_utils.calculate_n_hs_channels(wandb.config)
    }, include_preprocessing_setting=False)

    train()