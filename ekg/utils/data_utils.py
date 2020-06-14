import numpy as np
import pandas as pd
import itertools
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pywt
from tensorflow import keras

import better_exceptions; better_exceptions.hook()

from ..audicor_reader import reader

NUM_PROCESSES = mp.cpu_count()*2

class DataAugmenter:
    def __init__(self, indices_channel_ekg, indices_channel_hs,
                    ekg_scaling_prob, hs_scaling_prob,
                    time_stretch_prob):

        self.indices_channel_ekg = indices_channel_ekg
        self.indices_channel_hs = indices_channel_hs
        self.total_nchannel = len(self.indices_channel_hs) + len(self.indices_channel_ekg)

        self.ekg_scaling_prob = ekg_scaling_prob
        self.hs_scaling_prob = hs_scaling_prob
        self.time_stretch_prob = time_stretch_prob

    # input shape: (10000, 10)
    def ekg_scaling(self, X, ratio):
        X[..., self.indices_channel_ekg] *= ratio

    def hs_scaling(self, X, ratio):
        X[..., self.indices_channel_hs] *= ratio

    def random_crop(self, X, length):
        original_length = X.shape[-2]
        random_offset = np.random.choice(np.arange(0, original_length-length+1, 1))
        return X[random_offset: random_offset+length, :]

    def time_stretch(self, X, ratio):
        rtn = list()
        original_length = X.shape[-2]
        for i in range(X.shape[-1]): # through every channel
            rtn.append(np.interp(np.linspace(0, original_length, ratio*original_length+1), np.arange(original_length), X[..., i]))

        rtn = np.array(rtn).swapaxes(0, 1) # (10000, 10)
        return self.random_crop(rtn, original_length)

    def augment(self, X):
        X  = X.copy().astype(np.float32)

        if np.random.rand() <= self.ekg_scaling_prob:
            self.ekg_scaling(X, np.random.uniform(0.95, 1.05))

        if np.random.rand() <= self.hs_scaling_prob:
            self.hs_scaling(X, np.random.uniform(0.95, 1.05))

        if np.random.rand() <= self.time_stretch_prob:
            X = self.time_stretch(X, np.random.uniform(1.0, 1.05))

        return X

class BaseDataGenerator:
    def __init__(self, dataloaders, wandb_config, preprocessing_fn, **kwargs):
        self.dataloaders = dataloaders
        self.config = wandb_config
        self.preprocessing_per_dataloader = preprocessing_fn

        # set additional variables
        self.__dict__.update(kwargs)

        self.means_and_stds = None # [means, stds], the shapes of means and stds are both [n_channels].

        self.preprocessing()

    def preprocessing(self):
        for dataloader in self.dataloaders:
            self.preprocessing_per_dataloader(dataloader)
        return

    @staticmethod
    def normalize(X, means_and_stds=None, include_info=False):
        def norm(xi, msi):
            if msi is None:
                means = [ xi[..., i].mean(dtype=np.float32) for i in range(xi.shape[-1]) ]
                stds = [ xi[..., i].std(dtype=np.float32) for i in range(xi.shape[-1]) ]
            else:
                means = msi[0]
                stds = msi[1]

            normalized_X = np.zeros_like(xi, dtype=np.float32)
            for i in range(xi.shape[-1]):
                normalized_X[..., i] = xi[..., i].astype(np.float32) - means[i]
                normalized_X[..., i] = normalized_X[..., i] / stds[i]
            return normalized_X, [means, stds]

        if include_info:
            ekg_hs, info = X[0], X[1]
            normalized_ekg_hs, ekg_hs_means_and_stds = norm(ekg_hs, means_and_stds[0] if means_and_stds is not None else None)
            normalized_info, info_means_and_stds = norm(info, means_and_stds[1] if means_and_stds is not None else None)
            return [normalized_ekg_hs, normalized_info], [ekg_hs_means_and_stds, info_means_and_stds]
        return norm(X, means_and_stds)

    def get_split(self, rs=42):
        '''Split the data into training set, validation set, and testing set.
        Outputs:
            train_set, valid_set, test_set
        '''
        def combine(set1, set2):
            if self.config.include_info:
                rtn = [None, None]
                # X
                rtn[0] = [np.append(set1[0][i], set2[0][i], axis=0) for i in range(2)]
                # y
                rtn[1] = np.append(set1[1], set2[1], axis=0)
                return rtn
            return [np.append(set1[i], set2[i], axis=0) for i in range(2)]

        def has_empty(datasets, threshold=2):
            for dataset in datasets:
                y = dataset[1]
                for i in range(y.shape[1]):
                    if (y[:, i, 0] == 1).sum() <= threshold: # # of signals with events
                        return True
            return False

        def split(rs):
            train_set, valid_set, test_set = None, None, None
            for dataloader in self.dataloaders:
                tmp_train_set, tmp_valid_set, tmp_test_set = dataloader.get_split(rs)

                # combine them
                if train_set is None:
                    train_set, valid_set, test_set = tmp_train_set, tmp_valid_set, tmp_test_set
                else:
                    train_set = combine(train_set, tmp_train_set)
                    valid_set = combine(valid_set, tmp_valid_set)
                    test_set = combine(test_set, tmp_test_set)
            return train_set, valid_set, test_set

        while True:
            datasets = split(rs)
            if has_empty(datasets):
                rs += 1
                continue

            return list(datasets)

    def get(self):
        '''Split the data into training set, validation set, and testing set, and then normalize them by training set.
        Outputs:
            train_set, valid_set, test_set
        '''
        def multi_input_format(X, include_info=False):
            if include_info:
                ekg_hs, info = X[0], X[1]
                return {
                    'ekg_hs_input': ekg_hs,
                    'info_input': info
                }
            else:
                return {
                    'ekg_hs_input': X
                }

        train_set, valid_set, test_set = self.get_split()

        # do normalize using means and stds from training data
        train_set[0], self.means_and_stds = self.normalize(train_set[0], include_info=self.config.include_info)
        valid_set[0], _ = self.normalize(valid_set[0], self.means_and_stds, include_info=self.config.include_info)
        test_set[0], _ = self.normalize(test_set[0], self.means_and_stds, include_info=self.config.include_info)

        # convert X format to dict for model
        train_set[0]    = multi_input_format(train_set[0], self.config.include_info)
        valid_set[0]    = multi_input_format(valid_set[0], self.config.include_info)
        test_set[0]     = multi_input_format(test_set[0], self.config.include_info)

        return train_set, valid_set, test_set

def to_spectrogram(data):
    rtn = list()

    __mp_generate_spectrogram = partial(reader.generate_spectrogram, sampling_rates=[1000.]*data.shape[-2]) # ?, 10, 10000
    with mp.Pool(processes=32) as workers:
        for result in tqdm(workers.imap(__mp_generate_spectrogram, data), total=data.shape[0], desc='To spectrogram'):
            rtn.append([Sxx for f, t, Sxx in result])

    return np.array(rtn)

def calculate_channel_set(n_ekg_channels, n_hs_channels, ekg_channels, hs_channels):
    '''Return the all combinations of ekg + hs channels selections.
    Args:
        n_ekg_channels: number of the ekg channels should be selected.
        n_hs_channels: number of the hs channels should be selected.
        ekg_channels: a list of indices of ekg channels should be choosen from.
        hs_channels: a list of indices of hs channels should be choosen from.
    Outpus:
        A list of lists that contain all the combinations of ekg + hs channels.
        (i.e. [[1, 8], [1, 9]] if ekg/hs channels are [1] and [8, 9])
    '''

    def to_list(x):
        return [list(xi) for xi in x]
    def flatten(x):
        return [xi[0] + xi[1] for xi in x]
    all_ekg_channel_set = to_list(itertools.combinations(ekg_channels, n_ekg_channels))
    all_hs_channel_set = to_list(itertools.combinations(hs_channels, n_hs_channels))

    return flatten(to_list(itertools.product(all_ekg_channel_set, all_hs_channel_set)))

def patient_split(*args, **kwargs):
    import warnings
    warnings.warn('The patient_split() method is deprecated, use subject_split() instead!', UserWarning)
    return subject_split(*args, **kwargs)

def subject_split(X, y, subject_id, rs=42, infos=None):
    '''Split the X and y by the subject ID by the ratio (n_train: n_valid: n_test = 0.49: 0.21: 0.3).
    Args:
        X: np.ndarray of shape [n_instance, ...]
        y: np.ndarray of shape [n_instance, ...]
        infos: np.ndarray of shape [n_instance, ...]
    Returns: 
        [X_train, y_train], [X_valid, y_valid], [X_test, y_test]
    '''
    unique_subject_id = np.unique(np.array(subject_id))

    # split subject_id
    train_id, test_id = train_test_split(unique_subject_id, test_size=0.3, random_state=rs)
    train_id, valid_id = train_test_split(train_id, test_size=0.3, random_state=rs)

    _m_train    = [pi in train_id for pi in subject_id]
    _m_test     = [pi in test_id for pi in subject_id]
    _m_valid    = [pi in valid_id for pi in subject_id]

    X_train, y_train    = X[_m_train],  y[_m_train]
    X_test, y_test      = X[_m_test],   y[_m_test]
    X_valid, y_valid    = X[_m_valid],  y[_m_valid]

    if infos is not None:
        X_train     = [X_train, infos[_m_train]]
        X_test      = [X_test,  infos[_m_test]]
        X_valid     = [X_valid, infos[_m_valid]]

    return [[X_train, y_train], [X_valid, y_valid], [X_test, y_test]]

def downsample(X, ratio=2, mode='direct', channels_last=False):
    '''Downsample by either direct mode or averaging mode.
    Args:
        X: shape [n_instances, n_samples, n_channels] if channels_last else
                    [n_instances, n_channels, n_samples]
        ratio: downsample ratio, integer only for now
        mode: either direct or average
    Outputs:
        Downsampled_X
    '''
    if mode == 'direct':
        if channels_last:
            return X[..., ::ratio, :]
        else:
            return X[..., ::ratio]
    elif mode == 'average':
        if channels_last:
            return (X[..., ::ratio, :] + X[..., 1::ratio, :]) / 2.0
        else:
            return (X[..., ::ratio] + X[..., 1::ratio]) / 2.0
    else:
        raise ValueError('Invalid mode')

def calculate_n_ekg_channels(config):
    n_ekg_channels = 99999
    if 'big_exam' in config.datasets:
        n_ekg_channels = min(n_ekg_channels, len(config.big_exam_ekg_channels))
    
    if 'audicor_10s' in config.datasets:
        n_ekg_channels = min(n_ekg_channels, len(config.audicor_10s_ekg_channels))

    return n_ekg_channels

def calculate_n_hs_channels(config):
    n_hs_channels = 99999
    if 'big_exam' in config.datasets:
        n_hs_channels = min(n_hs_channels, len(config.big_exam_hs_channels))
    
    if 'audicor_10s' in config.datasets:
        n_hs_channels = min(n_hs_channels, len(config.audicor_10s_hs_channels))

    return n_hs_channels

def generate_wavelet(signal, fs, scale_length=25):
    '''Generate CWT signals of 22Hz - 75Hz.
    Args:
        signal: np.array of shape [n_samples]
        fs: sampling rate
    Outputs:
        CWT_coef: np.array of shape [25, n_samples]
    '''

    scale = np.arange(scale_length) * (fs / 1000) + (fs / 1000) * 8
    coef, freqs = pywt.cwt(signal, scale, 'cgau5', sampling_period=1/fs)

    return np.abs(coef)

def mp_generate_wavelet(hs, fs, scale_length, desc):
    '''
    Args:
        hs: (?, n_samples, n_channels)
    Outputs:
        wavelet: (?, scale_length, n_samples, n_channels)
    '''
    n_instances = hs.shape[0]
    hs = np.swapaxes(hs, 1, 2) # -> (?, n_channels, n_samples)
    hs = hs.reshape(-1, hs.shape[-1])

    wavelets = list()
    __mp_gw = partial(generate_wavelet, fs=fs, scale_length=scale_length)

    with mp.Pool(processes=8) as workers:
        for wi in tqdm(workers.imap(__mp_gw, hs), total=hs.shape[0], desc=desc):
            wavelets.append(wi)

    wavelets = np.array(wavelets)
    wavelets = wavelets.reshape((n_instances, -1, wavelets.shape[-2], wavelets.shape[-1]))
    wavelets = np.moveaxis(wavelets, 1, -1) # -> (?, scale_length, n_samples, n_channels)
    return wavelets

if __name__ == '__main__':
    pass