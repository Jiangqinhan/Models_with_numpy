import numpy as np
from ftrl import FtrlEstimator
from collections import namedtuple
from Base_class import BaseEstimator

class WideLayer:
    '''

    '''
    def __init__(self,field_names,alpha,beta,L1,L2,proba_fn):
        """
        :param proba_fn:    proba_fn(example_idx,logit)=probability
                            之所以用function是因为如果与DNN结合，计算probability还要考虑DNN提供的logit
        """
        self._estimator={field_name :FtrlEstimator(alpha,beta,L1,L2) for field_name in (['bias']+field_names)}
        self._proba_fn=proba_fn

    def __predict_logit(self,sp_features,exmaple_idx):
        '''
        遍历所有field return sum(logit)
        :param sp_features: dict{field_name:SparseInput}
        :param exmaple_idx:
        :return: sum(logit) for estimator belongs to self._estimator
        '''
        logit=0
        for field,estimator in self._estimator.items():
            if field!='bias':
                sp_input=sp_features[field]
                feat_ids,feat_vals=sp_input.get_example_in_order(exmaple_idx)
            else:
                feat_ids=[0]
                feat_vals=[1]
            logit+=estimator.predict_logit(feat_ids,feat_vals)
        return logit

    def train(self,features,labels):
        '''

        :param features: dict{field_name:SparseInput}
        :param labels:
        :return: probabilities from this batch
        '''
        probas=[]
        for example_id,label in enumerate(labels):
            #predict
            logit=self.__predict_logit(features,example_id)
            #这里是重点,如果是lr模型proba_fn对应sigmoid函数,wide&deep就还要考虑deep部分的logit
            pred_proba=self._proba_fn(example_id,logit)
            probas.append(pred_proba)
            #update

            for _,estiamtor in self._estimator.items():
                estiamtor.update(pred_proba,label)
        return np.asarray(probas)


    def predict_logit(self,sp_features):
        '''

        :param sp_features: dict{field_name:SparseInput}
        :return:
        '''
        batch_size=None
        for sp_feat in sp_features.values():
            batch_size=sp_feat.n_total_examples
            break

        logits=[self.__predict_logit(sp_features,example_idx) for example_idx in range(batch_size)]
        return np.asarray(logits)

WideHParams = namedtuple("WideHParams", ['field_names', 'alpha', 'beta', 'L1', 'L2'])


def _sigmoid(example_idx,logit):
    return 1 / (1 + np.exp(-logit))


class WideEstimator(BaseEstimator):
    def __init__(self,hparams,data_source):
        self._layer=WideLayer(field_names=hparams.field_names,
                                alpha=hparams.alpha,
                                beta=hparams.beta,
                                L1=hparams.L1,
                                L2=hparams.L2,
                                proba_fn=_sigmoid)
        super(WideEstimator, self).__init__(data_source)

    def train_batch(self,features,labels):
        return self._layer.train(features,labels)

    def predict(self,features):
        pred_logits = self._layer.predict_logit(features)
        return 1 / (1 + np.exp(- pred_logits))
