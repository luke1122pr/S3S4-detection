import os, sys
import numpy as np
import sklearn.metrics
import better_exceptions; better_exceptions.hook()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ekg.utils.eval_utils import print_cm
from ekg.utils.train_utils import allow_gpu_growth; allow_gpu_growth()

def evaluation(models, test_set):
    print('Testing set s3 baseline:', test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0] , "T/Total")
    #print('Testing set s4 baseline:',1- test_set[1][:, 2].sum() / test_set[1][:, 1].shape[0])

    y_pred_vote = np.zeros_like(test_set[1][:, 1])
    try: # model ensemble
        for model in models:
            y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
            y_pred_vote = y_pred_vote + y_pred

        y_pred = (y_pred_vote > (len(models) / 2)).astype(int)
    except:
        #y_pred = np.argmax(models.predict(test_set[0], batch_size=64), axis=1)
        y_pred = np.argmax(models.predict(test_set[0], batch_size=64)[:,:2], axis=1)
    
    y_true = test_set[1][:, 1]
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')

    print('Total accuracy:', accuracy)
    #print(sklearn.metrics.classification_report(y_true, y_pred))
    #print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['T', 'F'])
    '''
    y_true = test_set[1][:, 3]
    accuracy = sklearn.metrics.accuracy_score(y_true, s4y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(y_true, s4y_pred, average='micro')

    print('Total accuracy:', accuracy)
    #print(sklearn.metrics.classification_report(y_true, s4y_pred))
    #print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, s4y_pred), ['s4', 'no_s4'])
    '''
    return accuracy, precision, recall, f1_score
def evaluation_519(models, test_set):
    print('Testing set s3 baseline:', 1-test_set[1][:, 0].sum() / test_set[1][:, 0].shape[0])
    #print('Testing set s4 baseline:',1- test_set[1][:, 2].sum() / test_set[1][:, 1].shape[0])

    y_pred_vote = np.zeros_like(test_set[1][:, 1])
    try: # model ensemble
        for model in models:
            y_pred = np.argmax(model.predict(test_set[0], batch_size=64), axis=1)
            y_pred_vote = y_pred_vote + y_pred

        y_pred = (y_pred_vote > (len(models) / 2)).astype(int)
    except:
        #y_pred = np.argmax(models.predict(test_set[0], batch_size=64), axis=1)
        y_pred = models.predict(test_set[0], batch_size=64)
        for i in range (y_pred.shape[0]):
            y_pred[i]=int(y_pred[i]+0.5)            
        #s4y_pred = np.argmax(models.predict(test_set[0], batch_size=64)[:,2:], axis=1)
        yy_pred=models.predict(test_set[0], batch_size=64)
    #print('yy_pred',yy_pred[:5],yy_pred.shape)
    #print('y_pred',y_pred[:5])
    
    
    
    y_true = test_set[1][:, 0]
    #print('y_true',y_true[:5],y_true.sum())
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')

    print('Total accuracy:', accuracy)
    #print(sklearn.metrics.classification_report(y_true, y_pred))
    #print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, y_pred), ['s3', 'no_s3'])
    '''
    y_true = test_set[1][:, 3]
    accuracy = sklearn.metrics.accuracy_score(y_true, s4y_pred)
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(y_true, s4y_pred, average='micro')

    print('Total accuracy:', accuracy)
    #print(sklearn.metrics.classification_report(y_true, s4y_pred))
    #print()
    print_cm(sklearn.metrics.confusion_matrix(y_true, s4y_pred), ['s4', 'no_s4'])
    '''
    return accuracy, precision, recall, f1_score
def log_evaluation(models, test_set, log_prefix):
    accuracy, precision, recall, f1_score = evaluation(models, test_set)
    wandb.log({'{}_acc'.format(log_prefix): accuracy})
    wandb.log({'{}_precision'.format(log_prefix): precision})
    wandb.log({'{}_recall'.format(log_prefix): recall})
    wandb.log({'{}_f1_score'.format(log_prefix): f1_score})

def print_statistics(train_set, valid_set, test_set):
    print('Statistics:')
    for set_name, dataset in [['Training set', train_set], ['Validation set', valid_set], ['Testing set', test_set]]:
        number_abnormal = dataset[1][:, 1].sum()
        number_normal = dataset[1][:, 0].sum()

        print('{}:'.format(set_name))
        print('\tAbnormal : Normal = {} : {}'.format(number_abnormal, number_normal))
        print('\tAbnormal Ratio: {:.4f}'.format(number_abnormal / (number_normal + number_abnormal)))
        print()

if __name__ == '__main__':
    import wandb

    from ekg.utils.eval_utils import parse_wandb_models
    from ekg.utils.eval_utils import get_evaluation_args, evaluation_log

    from train import AbnormalBigExamLoader, AbnormalAudicor10sLoader
    from train import preprocessing
    from ekg.utils.data_utils import BaseDataGenerator

    wandb.init(project='s3s4detect5_15', entity='luke1219')

    # get arguments
    args = get_evaluation_args('Abnormal detection evaluation.')

    # parse models and configs
    models, wandb_configs, model_paths, sweep_name = parse_wandb_models(args.paths, args.n_model, args.metric)

    evaluation_log(wandb_configs, sweep_name, 
                    args.paths[0] if args.n_model >= 1 else '',
                    model_paths)

    models[0].summary()

    wandb_config = wandb_configs[0]
    dataloaders = list()
    if 'big_exam' in wandb_config.datasets:
        dataloaders.append(AbnormalBigExamLoader(wandb_config=wandb_config))
    if 'audicor_10s' in wandb_config.datasets:
        dataloaders.append(AbnormalAudicor10sLoader(wandb_config=wandb_config))

    g = BaseDataGenerator(dataloaders=dataloaders,
                            wandb_config=wandb_config,
                            preprocessing_fn=preprocessing)
    train_set, valid_set, test_set = g.get()

    print_statistics(train_set, valid_set, test_set)

    print('Training set:')
    log_evaluation(models, train_set, 'best')

    print('Validation set:')
    log_evaluation(models, valid_set, 'best_val')

    print('Testing set:')
    evaluation(models, test_set)