import wandb
import numpy as np
from tensorflow import keras

from lifelines.utils import concordance_index

class LogBest(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', records=['val_loss', 'loss', 'val_acc', 'acc']):
        self.monitor = monitor
        self.records = records

        setattr(self, 'best_' + self.monitor, np.inf if 'loss' in self.monitor else 0)
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if getattr(self, 'best_' + self.monitor) > logs.get(self.monitor): # update
            setattr(self, 'best_' + self.monitor, logs.get(self.monitor))

            log_dict = dict()
            for rs in self.records:
                log_dict['best_' + rs] = logs.get(rs)
            log_dict['best_epoch'] = epoch

            wandb.log(log_dict, commit=False)

class ConcordanceIndex(keras.callbacks.Callback):
    def __init__(self, train_set, valid_set, event_names):
        super(ConcordanceIndex, self).__init__()
        self.train_set = train_set
        self.valid_set = valid_set
        self.event_names = event_names

    def on_epoch_end(self, epoch, logs={}):
        X_train = self.train_set[0]
        X_valid = self.valid_set[0]

        pred_train = self.model.predict(X_train) # (?, n_events)
        pred_valid = self.model.predict(X_valid)

        for i in range(len(self.event_names)):
            cs_train, st_train = self.train_set[1][:, i, 0], self.train_set[1][:, i, 1]
            cs_valid, st_valid = self.valid_set[1][:, i, 0], self.valid_set[1][:, i, 1]

            try:
                train_cindex = concordance_index(st_train, -pred_train[:, i], cs_train)
            except ZeroDivisionError:
                train_cindex = np.nan
            
            try:
                valid_cindex = concordance_index(st_valid, -pred_valid[:, i], cs_valid)
            except ZeroDivisionError:
                valid_cindex = np.nan

            print('Concordance index of {} training set: {:.4f}'.format(self.event_names[i], train_cindex))
            print('Concordance index of {} validation set: {:.4f}'.format(self.event_names[i], valid_cindex))

            # append cindex to logs
            logs['{}_cindex'.format(self.event_names[i])] = train_cindex
            logs['val_{}_cindex'.format(self.event_names[i])] = valid_cindex
