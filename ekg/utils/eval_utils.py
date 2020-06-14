import yaml
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from tensorflow.keras.models import load_model
import tempfile
import wandb

from ..layers import LeftCropLike
from ..layers.sincnet import SincConv1D
from ..layers import CenterCropLike

from .train_utils import allow_gpu_growth; allow_gpu_growth()
from .train_utils import set_wandb_config

import argparse

import matplotlib.pyplot as plt

class YamlParser():
    def __init__(self):
        self.config = None

    def read(self, filename):
        with open(filename, 'r') as f:
            self.config = yaml.safe_load(f)

        for key, value in self.config.items():
            setattr(self, key, value)

        return self

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    '''pretty print for confusion matrixes'''
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = ' ' * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * ' ' + 't/p' + (columnwidth-3)//2 * ' '

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = ' ' * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print('    ' + fst_empty_cell, end=' ')
    # End CHANGES

    for label in labels:
        print('%{0}s'.format(columnwidth) % label, end=' ')

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print('    %{0}s'.format(columnwidth) % label1, end=' ')
        for j in range(len(labels)):
            cell = '%{0}.1f'.format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()

def get_KM_plot(train_pred, test_pred, test_true, event_name):
    '''
    Args:
        train_pred:     np.array of shape (n_samples)
        test_pred:      np.array of shape (n_samples)
        test_true:      np.array of shape (n_samples, 2)
                        [:, :, 0] - censoring states
                        [:, :, 1] - survival times
    '''
    # find median of training set
    median = np.median(train_pred) # single value

    # split testing data into 2 groups by median, high risk / low risk
    high_risk_indices = np.where(test_pred >= median)[0]
    low_risk_indices = np.where(test_pred < median)[0]
    
    high_risk_cs = test_true[high_risk_indices, 0]
    high_risk_st = test_true[high_risk_indices, 1]
    
    low_risk_cs = test_true[low_risk_indices, 0]
    low_risk_st = test_true[low_risk_indices, 1]
    
    # calculate logrank p value
    p_value = logrank_test(high_risk_st, low_risk_st, high_risk_cs, low_risk_cs).p_value
    
    # plot KM curve
    kmf = KaplanMeierFitter()
    a1 = None

    if high_risk_cs.shape[0] != 0:
        kmf.fit(high_risk_st, high_risk_cs, label='high risk')
        a1 = kmf.plot(figsize=(20, 10), title='{} KM curve, logrank p-value: {}'.format(event_name, p_value))
    
    if low_risk_cs.shape[0] != 0:
        kmf.fit(low_risk_st, low_risk_cs, label='low risk')
        if a1 is not None:
            kmf.plot(ax=a1)
        else:
            kmf.plot(figsize=(20, 10), title='{} KM curve, logrank p-value: {}'.format(event_name, p_value))

    plt.tight_layout()
    return plt

def get_survival_scatter(y_pred, cs_true, st_true, event_name):
    '''
    Args:
        y_pred: np.array of shape (n_samples)
        cs_true: np.array of shape (n_samples)
        st_true: np.array of shape (n_samples)
    '''
    plt.figure(figsize=(20, 10))

    # plot normal
    normal_mask = (cs_true == 0)
    plt.scatter(y_pred[normal_mask], st_true[normal_mask], color='black', alpha=0.2, label='censored')

    # plot normal
    abnormal_mask = (cs_true == 1)
    plt.scatter(y_pred[abnormal_mask], st_true[abnormal_mask], marker=6, s=100, c='#ff1a1a', label='event occured')

    plt.xlabel('predicted risk')
    plt.ylabel('survival time (days)')
    
    plt.legend()
    plt.title('{} - cindex: {:.3f}'.format(event_name, concordance_index(st_true, -y_pred, cs_true)))

    plt.tight_layout()
    return plt

def log_configs(configs):
    '''Log the same parts amoung the configs to wandb

    Args:
        configs: list of wandb_config objects
    '''
    def all_same(check_key, check_value, configs):
        for config in configs:
            if config[check_key] != check_value:
                return False
        return True

    # convert configs to dicts
    configs = [vars(config) for config in configs]

    if len(configs) == 1:
        set_wandb_config(configs[0])
        return

    # get the key with the same value across configs
    for key, value in configs[0].items():
        if all_same(key, value, configs):
            set_wandb_config({key: value})

def dict_to_config(d):
    class Object(object):
        pass

    config = Object()
    for key, value in d.items():
        setattr(config, key, value)
    return config

def parse_wandb_models(path, number_models=-1, metric=None):
    '''Parse wandb models with either run paths or a sweep path.
    
    Args:
        path: a list contains either run paths or a sweep path
        number_models: if negative, treat path as run paths, otherwise treat it as a sweep path.
        metric: metric to sort by when parsing a sweep path
    '''
    api = wandb.Api()
    models, configs, model_paths = list(), list(), list()
    sweep_name = ''

    modeldir = tempfile.mkdtemp()

    if number_models > 0: # sweep
        sweep = api.sweep(path[0])
        sweep_name = sweep.config.get('name', '')
        # sort runs by metric
        runs = sorted(sweep.runs, key=lambda run: run.summary.get(metric, np.Inf if 'loss' in metric else 0), 
                            reverse=False if 'loss' in metric else True)
    else:
        runs = [api.run(p) for p in path]

    for run in runs[:number_models]:
        run.file('model-best.h5').download(replace=True, root=modeldir)

        # load model
        models.append(load_model(modeldir + '/model-best.h5', 
                            custom_objects={'SincConv1D': SincConv1D,
                                            'LeftCropLike': LeftCropLike,
                                            'CenterCropLike': CenterCropLike}, compile=False))

        configs.append(dict_to_config(run.config))
        model_paths.append(run.path)

    return models, configs, model_paths, sweep_name

def get_evaluation_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--n_model', type=int, default=-1,
                            help='Number of best models to evaluate.')
    parser.add_argument('-m', '--metric', type=str, default='best_val_loss',
                            help='Which metric to use for selecting best models from the sweep.')
    parser.add_argument('paths', metavar='paths', type=str, nargs='+',
                        help='Run paths or a sweep path of wandb to be evaluated. If n_model >= 1, it will be treated as sweep path.')

    args = parser.parse_args()

    return args

def evaluation_log(wandb_configs, sweep_name, sweep_path, model_paths):
    '''Log basic evaluation config.

    '''
    log_configs(wandb_configs)

    wandb.config.sweep_name = sweep_name
    wandb.config.sweep_path = sweep_path
    wandb.config.n_models = len(model_paths)
    wandb.config.models = model_paths
    wandb.config.evaluation = True