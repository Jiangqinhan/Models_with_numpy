from tqdm import tqdm
import numpy as np
from metrics import logloss as log_loss, auc as roc_auc_score
import logging
class Layer:
    '''
    可以没有 backward
    有 backward 的话还要定义
    @property
    def variables(self):返回优化变量 dict {var_name:val}
        @property
    def grads2var(self):返回梯度 dict{grad_name:val}
    @property
    def l2reg_loss(self): L2正则
    在__init__()中一般要储存参数和梯度,每个层要有自己的名字,参数的维度也在这里输入
    '''
    def forward(self,input):
        '''

        :param input:
        :return:
        '''
        raise NotImplementedError()

    @property
    def output_dim(self):
        raise NotImplementedError()


class Model:
    '''
    将层构建为模型,把optimizer连接到模型中
    forward 调用 layers的forward
    backward 调用Layer的backward 并且 optimizer.update 更新参数
    '''
    def __init__(self,optimizer):
        self._optimizer=optimizer

    def forward(self,X):
        raise NotImplementedError()

    def backward(self,prev_grads):
        raise NotImplementedError()

class BaseEstimator:
    ''''
    子类主要定义train_batch和predict方法,这个类的任务是完成
    '''
    def __init__(self,data_source):
        '''
        在这里定义一个模型成员
        :param data_source:
        '''
        self._data_source=data_source

    def get_metrics(self,score,label,perfix):
        score=np.asarray(score)
        label=np.asarray(label)
        metrics={'{}_logloss'.format(perfix):log_loss(label,score),
                 '{}_auc'.format(perfix):roc_auc_score(label,score)}
        pred_labels=(score>0.5).astype(int)
        metrics['{}_accuracy'.format(perfix)]=np.sum(pred_labels==label)/len(label)

        return metrics



    def train_batch(self,features,labels):
        '''
        在train_epoch中被调用
        :param features: dict{fieldname:SparseInput/DenseInput}
        :param labels: array
        :return: ndarray of predicted probability in this batch
        '''
        raise NotImplementedError()

    def predict(self,features):
        '''
        同上
        :param features:
        :return:
        '''
        raise  NotImplementedError()

    def train_epoch(self):
        '''
        extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        :return:
        '''
        batch_stream = self._data_source.train_batches_per_epoch()
        scores=[]
        labels=[]
        for features,_batch_labels in tqdm(batch_stream):
            pred_probas=self.train_batch(features,_batch_labels)
            scores.extend(pred_probas)
            labels.extend(_batch_labels)
        return self.get_metrics(scores,labels,'train')

    def eval_epoch(self):
        batch_stream=self._data_source.test_batches_per_epoch()
        scores=[]
        labels=[]
        for features,batch_labels in tqdm(batch_stream):
            pred=self.predict(features)
            scores.extend(pred)
            labels.extend(batch_labels)
        return self.get_metrics(scores,labels,'test')

    def train(self,n_epochs):
        metrics_history=[]
        for epoch in range(n_epochs):
            logging.info("\n=============== {}-th EPOCH".format(epoch + 1))
            metrics={}
            metrics.update(self.train_epoch())
            metrics.update(self.eval_epoch())
            logging.info(metrics)
            metrics_history.append(metrics)
        return metrics_history

