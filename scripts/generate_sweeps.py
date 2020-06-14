import copy

base_setting = {
    'method': 'bayes',

    'metric':{
        'name': 'best_val_loss',
        'goal': 'minimize'
    },

    'early_terminate':{
        'type': 'hyperband'
    },

    'parameters':{
        'sincconv_filter_length':{
            'min': 16,
            'max': 128
        },
        'sincconv_nfilters':{
            'values': [8, 16, 32]
        },
        'branch_nlayers':{
            'values': [1, 2, 3, 4, 5]
        },
        'ekg_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'hs_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'ekg_nfilters':{
            'values': [1, 2, 4, 8, 16, 32]
        },
        'hs_nfilters':{
            'values': [1, 2, 4, 8, 16, 32]
        },
        'final_nlayers':{
            'values': [3, 4, 5, 6]
        },
        'final_kernel_length':{
            'values': [5, 7, 13, 21, 35]
        },
        'final_nonlocal_nlayers':{
            'values': [0]
        },
        'final_nfilters':{
            'values': [8, 16, 32]
        },
        'kernel_initializer':{
            'values': ['glorot_uniform'] # , 'he_normal']
        },
        'skip_connection':{
            'values': [True, False]
        },
        'crop_center':{
            'values': [True]
        },
        'se_block':{
            'values': [True, False]
        },
        'radam':{
            'values': [True, False]
        }
    }
}

def set_parameters(d, key, value, search=False):
    d['parameters'][key] = {
        'values' if search else 'value': copy.deepcopy(value)
    }

def generate_sweep(task, dataset, hs_ekg_setting):
    sweep = copy.deepcopy(base_setting)
    sweep['name'] = '{}/{}/{}'.format(task, dataset, hs_ekg_setting)
    sweep['program'] = './{}/train.py'.format(task)

    set_parameters(sweep, 'datasets', [dataset] if 'hybrid' not in dataset else ['audicor_10s', 'big_exam'])
    
    # audicor_10s
    set_parameters(sweep, 'audicor_10s_ekg_channels', list() if hs_ekg_setting == 'only_hs' else [0])
    set_parameters(sweep, 'audicor_10s_hs_channels', list() if hs_ekg_setting == 'only_ekg' else [1])
    set_parameters(sweep, 'audicor_10s_only_train', False)

    # big_exam
    if 'hybrid' in dataset:
        set_parameters(sweep, 'big_exam_ekg_channels', list() if hs_ekg_setting == 'only_hs' else [1])
    else:
        set_parameters(sweep, 'big_exam_ekg_channels', list() if hs_ekg_setting == 'only_hs' else list(range(8)))

    set_parameters(sweep, 'big_exam_hs_channels', list() if hs_ekg_setting == 'only_ekg' else [8, 9])
    set_parameters(sweep, 'big_exam_only_train', (dataset == 'hybrid/audicor_as_test'))

    # hazard_prediction
    if task == 'hazard_prediction':
        # model
        set_parameters(sweep, 'prediction_head', [True, False], search=True)
        set_parameters(sweep, 'prediction_nlayers', [2, 3, 4, 5], search=True)
        set_parameters(sweep, 'prediction_kernel_length', [5, 7, 13, 21, 35], search=True)
        set_parameters(sweep, 'prediction_nfilters', [8, 16, 32], search=True)

        # data
        events = ['ADHF', 'Mortality'] + (['MI', 'CVD'] if dataset == 'big_exam' else []) # Stoke
        set_parameters(sweep, 'events', events)
        set_parameters(sweep, 'event_weights', [1 for _ in events])
        set_parameters(sweep, 'censoring_limit', 400 if dataset == 'hybrid/audicor_as_test' else 99999)
        set_parameters(sweep, 'batch_size', 64)
    
    return sweep

def get_all_sweeps():
    sweeps = list()
    tasks = ['abnormal_detection', 'hazard_prediction']
    datasets = ['audicor_10s', 'big_exam', 'hybrid/audicor_as_test', 'hybrid/both_as_test']
    hs_ekg_settings = ['only_hs', 'only_ekg', 'whole']

    for task in tasks:
        for dataset in datasets:
            for hs_ekg_setting in hs_ekg_settings:
                sweep = generate_sweep(task, dataset, hs_ekg_setting)
                sweeps.append(sweep)
    return sweeps