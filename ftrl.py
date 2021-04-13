import numpy as np
from collections import defaultdict

class FtrlEstimator:
    '''
    每个field对应一个FtrlEstimator,优点
    代码清晰、易读
    方便扩展。比如某个Field下新增/删除了一个Feature，只有这个Field下的Feature需要重新编号，其他Field不受影响。
    各个Field之间可以并行计算
    这里的设计想法和Embedding Layer一致
    _n : dict {field_id, n_i}
    _z : dict {field_id,z_i}
    因此 使用方法 dict{field:FtrlEstimator}
    for estimator in dict:
        logit+=estimator.predict_logit(....)
    proba= activation(logit)
    for estimator in dict:
        estimator.update(proba,label)


    '''
    def __init__(self,alpha,beta,L1,L2):
        self._alpha = alpha
        self._beta = beta
        self._L1 = L1
        self._L2 = L2
        #在update时使用 都是list
        self.current_feat_ids=None
        self.current_feat_vals=None
        # lazy weights, 实际上是一个临时变量，只在：
        # 1. 对应的feature value != 0, 并且
        # 2. 之前累积的abs(z) > L1
        # 两种情况都满足时，w才在feature id对应的位置上存储一个值
        # 而且w中数据的存储周期，只在一次前代、回代之间，在新的前代开始之前，就清空上次的w
        self._w={}
        #defaultdict 访问的key如果不存在 返回默认值
        self._n=defaultdict(float)
        self._z=defaultdict(float)

    def predict_logit(self,feature_ids,feature_vals):
        '''

        :param feature_ids: list 某个样本的非零的特征名
        :param feature_vals: list 非零特征的值
        :return: float  Sigma wi*feature_val  for all i  belongs to feature_ids
        '''
        self.current_feat_ids=feature_ids
        self.current_feat_vals=feature_vals
        #把以前的梯度清零
        self._w.clear()
        logit=0
        for feat_id,feature_val in zip(feature_ids,feature_vals):
            z=self._z[feat_id]
            sign_z=-1 if z<0 else 1

            if abs(z)>self._L1:
                w=(sign_z * self._L1 - z) / ((self._beta + np.sqrt(self._n[feat_id])) / self._alpha + self._L2)
                self._w[feat_id]=w
                logit+=w*feature_val

        return logit

    def update(self,pred_prob,label):
        '''

        :param pred_prob: sigmoid(wx+b)
        :param label:
        :return:
        '''
        grad2logit=pred_prob-label

        for feat_id,feat_val in zip(self.current_feat_ids,self.current_feat_vals):
            #二项分布对应的梯度
            g=grad2logit*feat_val
            g2=grad2logit*grad2logit
            n=self._n[feat_id]
            self._z[feat_id]+=g
            self._n[feat_id]+=g2
            if feat_id in self._w:
                sigma=(np.sqrt(self._n[feat_id])-np.sqrt(n))/self._alpha
                self._z[feat_id]-=sigma*self._w[feat_id]

