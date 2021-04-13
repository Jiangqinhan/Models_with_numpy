from DataSource import DataSource
from EmbeddingLayer import EmbeddingLayer
import numpy as np
from Input import SparseInput
from tqdm import tqdm
from DenseLayer import DenseLayer
import utils
import pandas as pd
import logging
import time
from ftrl import FtrlEstimator
from optimizer import Adagrad
from dnn import DeepEstimator,DeepHParams
from Dataset import DENSE_FIELDS,CATEGORY_FIELDS,AGE_BOUNDARIES,VOCAB_LISTS
from wide_layer import WideHParams,WideEstimator
from wide_deep import WideDeepEstimator
def get_deep_hparams(embed_size,hidden_units,L2,learning_rate):
    '''
    暂时只支持 Adagrad方法
    :param embedded_size:
    :param hidden_units:
    :param L2:
    :param learning_rate:
    :return:
    '''
    dense_fields=[(field_name,1) for field_name in DENSE_FIELDS]
    vocab_info=[]
    for vocab_name in CATEGORY_FIELDS:
        if vocab_name=='age_buckets':
            vocab_size=len(AGE_BOUNDARIES)+1
        else:
            vocab_size=len(VOCAB_LISTS[vocab_name])
        vocab_info.append((vocab_name,vocab_size,embed_size))
    embed_fields=[(field,field)for field in CATEGORY_FIELDS]


    optimizer=Adagrad(lr=learning_rate)
    return DeepHParams(dense_fields=dense_fields,
                       L2=L2,
                       optimizer=optimizer,
                       hidden_units=hidden_units,
                       vocab_infos=vocab_info,
                       embed_fields=embed_fields
                       )
def test_wide_deep():
    utils.config_logging('log_{}.log'.format('WideEstimator'))
    n_epoch = 10
    batch_size = 32
    data_source = DataSource('./dataset/train.csv', './dataset/test.csv', batch_size)
    wide_hparams = WideHParams(field_names=CATEGORY_FIELDS,
                               alpha=0.1,
                               beta=1,
                               L1=0.1,
                               L2=0.1)
    deep_hparams = get_deep_hparams(embed_size=16,
                                    hidden_units=[64, 16],
                                    L2=0.01,
                                    learning_rate=0.001)
    start_time = time.time()
    estimator = WideDeepEstimator(wide_hparams=wide_hparams, deep_hparams=deep_hparams, data_source=data_source)
    metrics_history = estimator.train(n_epoch)
    elapsed = time.time() - start_time

    # ************ display result
    logging.info("\n************** TIME COST **************")
    logging.info('{:.2f} seconds for {} epoches'.format(elapsed, n_epoch))
    logging.info('{:.2f} examples per second'.format(
        n_epoch * (data_source.n_train_examples + data_source.n_test_examples) / elapsed))

    logging.info("\n************** LEARNING CURVE **************")
    metrics_history = pd.DataFrame(metrics_history)
    logging.info(metrics_history)
    metrics_history.to_csv('learn_curve_{}.csv'.format(estimator), index=False)

if __name__ == "__main__":
    test_wide_deep()









