import numpy as np
import pandas as pd
import os
from .data_utils import downsample, calculate_channel_set, subject_split

# read config file
import configparser
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config.cfg'))

class DatasetLoader():
    def __init__(self, datadir, 
                    ekg_channels, hs_channels, 
                    sampling_rate, only_train,
                    wandb_config):
        self.datadir = datadir
        self.ekg_channels = ekg_channels
        self.hs_channels = hs_channels
        self.sampling_rate = sampling_rate
        self.only_train = only_train
        self.config = wandb_config

        assert hasattr(self.config, 'n_ekg_channels'), 'n_ekg_channels doesn\'t exist in wandb config.'
        assert hasattr(self.config, 'n_hs_channels'), 'n_hs_channels doesn\'t exist in wandb config.'
        assert hasattr(self.config, 'sampling_rate'), 'sampling_rate doesn\'t exist in wandb config.'
        assert hasattr(self.config, 'downsample'), 'downsample doesn\'t exist in wandb config.'
        assert hasattr(self.config, 'with_normal_subjects'), 'with_normal_patient doesn\'t exist in wandb config.'

        self.channel_set = self.get_channel_set()

        # load_data
        self.abnormal_X = self.load_abnormal_X() # abnormal first
        self.normal_X = self.load_normal_X()

        self.abnormal_y = self.load_abnormal_y()
        self.normal_y = self.load_normal_y()

        self.abnormal_subject_id = self.load_subject_id(is_normal=False)
        self.normal_subject_id = self.load_subject_id(is_normal=True)

    def get_channel_set(self):
        return calculate_channel_set(self.config.n_ekg_channels,
                                        self.config.n_hs_channels,
                                        self.ekg_channels,
                                        self.hs_channels)

    def load_X(self, filename):
        X = np.zeros((0, self.config.n_ekg_channels+self.config.n_hs_channels, self.config.sampling_rate*10))
    
        full_X = np.load(os.path.join(self.datadir, filename)) # (n_instances, n_channels, n_samples)
        # downsample if needed
        if self.config.sampling_rate != self.sampling_rate:
            full_X = downsample(full_X,
                                    self.sampling_rate // self.config.sampling_rate,
                                    self.config.downsample,
                                    channels_last=False)
        # select channels
        for channel_set in self.channel_set:
            X = np.append(X, full_X[:, channel_set, :], axis=0)

        return np.swapaxes(X, 1, 2)

    def load_abnormal_X(self):
        return self.load_X('abnormal_X.npy')
    
    def load_normal_X(self):
        return self.load_X('normal_X.npy')

    def get_split(self, rs=42):
        '''Split the data into training set, validation set, and testing set.
        self.only_train: if True, data won't split and all go to training set.

        Outputs:
            train_set, valid_set, test_set
        '''
        def combine(set1, set2):
            if set1 is None or set2 is None: # return the not-None set if any of the sets is None
                return set1 if set1 is not None else set2

            return [np.append(set1[i], set2[i], axis=0) for i in range(2)]

        if self.only_train:
            X_shape = [shape for shape in self.abnormal_X.shape[1:]] # [n_samples, n_channels]
            y_shape = [shape for shape in self.abnormal_y.shape[1:]] # [?]

            if self.config.with_normal_subjects:
                train_set = [np.append(self.abnormal_X, self.normal_X, axis=0), np.append(self.abnormal_y, self.normal_y, axis=0)]
            else: # only abnormal subjects
                train_set = [self.abnormal_X, self.abnormal_y]
            valid_set = [np.empty([0]+X_shape), np.empty([0]+y_shape)]
            test_set = [np.empty([0]+X_shape), np.empty([0]+y_shape)]
            return [train_set, valid_set, test_set]

        # do abnormal split by abnormal subject ID
        abnormal_training_set, abnormal_valid_set, abnormal_test_set  = subject_split(self.abnormal_X, self.abnormal_y, self.abnormal_subject_id, rs)

        # do normal split by normal subject ID
        if self.config.with_normal_subjects:
            if self.config.normal_subjects_only_train:
                normal_training_set = [self.normal_X, self.normal_y]
                normal_valid_set, normal_test_set = None, None
            else:
                normal_training_set, normal_valid_set, normal_test_set = subject_split(self.normal_X, self.normal_y, self.normal_subject_id, rs)

            # combine
            train_set = combine(normal_training_set, abnormal_training_set)
            valid_set = combine(normal_valid_set, abnormal_valid_set)
            test_set = combine(normal_test_set, abnormal_test_set)
        else: # only abnormal subjects
            train_set, valid_set, test_set = abnormal_training_set, abnormal_valid_set, abnormal_test_set

        return [train_set, valid_set, test_set]

    def load_subject_id(self, is_normal):
        return NotImplementedError('load_subject_id not implemented.')

    def load_abnormal_y(self):
        return NotImplementedError('load_abnormal_y not implemented.')

    def load_normal_y(self):
        return NotImplementedError('load_normal_y not implemented!')

class BigExamLoader(DatasetLoader):
    def __init__(self, wandb_config):
        super().__init__(datadir=config['Big_Exam']['output_dir'],
                            ekg_channels=wandb_config.big_exam_ekg_channels,
                            hs_channels=wandb_config.big_exam_hs_channels,
                            sampling_rate=1000, 
                            only_train=wandb_config.big_exam_only_train,
                            wandb_config=wandb_config)

    def load_subject_id(self, is_normal):
        if is_normal: # assume normal data are all from different subjects
            subject_id = np.arange(self.normal_X.shape[0] // len(self.channel_set), dtype=int)

        else:
            # read abnormal_event for suject id
            df = pd.read_csv(os.path.join(self.datadir, 'abnormal_event.csv'))
            subject_id = df.subject_id.values

        return np.tile(subject_id, len(self.channel_set))

class Audicor10sLoader(DatasetLoader):
    def __init__(self, wandb_config):
        super().__init__(datadir=config['Audicor_10s']['output_dir'],
                            ekg_channels=wandb_config.audicor_10s_ekg_channels,
                            hs_channels=wandb_config.audicor_10s_hs_channels,
                            sampling_rate=500,
                            only_train=wandb_config.audicor_10s_only_train,
                            wandb_config=wandb_config)

    def load_subject_id(self, is_normal):
        if is_normal: # assume normal data are all from different subjects
            subject_id = np.load(os.path.join(self.datadir, 'normal_filenames.npy'))

        else:
            subject_id = np.load(os.path.join(self.datadir, 'abnormal_filenames.npy'))
            # get the filenames by spliting by '/'
            subject_id = np.vectorize(lambda fn: fn.split('/')[-1])(subject_id)

            # get the subject ids by spliting by '_'
            subject_id = np.vectorize(lambda fn: 
                        fn.split('_')[0] if not fn.startswith('MMH') else
                        fn.split('_')[0] + fn.split('_')[1] # fix MMH
            )(subject_id)

        return np.tile(subject_id, len(self.channel_set))