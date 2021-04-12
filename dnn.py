from Base_class import Model,BaseEstimator
from DenseLayer import DenseLayer
from EmbeddingLayer import EmbeddingCombineLayer
from Input import DenseInputCombineLayer
from Activation import ReLU
import numpy as np
from collections import namedtuple
import utils
'''
类的设计不够合理,应该把optimizer和loss_func全部放在Estimator类中
'''
class DeepNetwork(Model):
    def __init__(self,vocab_infos,dense_fields,embed_fields,hidden_units,optimizer,L2):
        '''

        :param vocab_infos:[(vocab_name, vocab_size, embed_size)]
        :param dense_fields:list[(filed_name,field_size)]
        :param embed_fields:(field_name, vocab_name)
        :param hidden_units:a list of ints, n_units for each hidden layer
        :param optimizer:optimizer instance to update the weights 函数指针
        :param L2:L2 regularization for hidden dense layer
        '''
        super(DeepNetwork, self).__init__(optimizer)

        self._embed_combine_layer=EmbeddingCombineLayer(vocab_infos)

        self._dense_combine_layer=DenseInputCombineLayer(dense_fields)
        for (field_name,vocab_name) in embed_fields:
            self._embed_combine_layer.add_embedding(vocab_name,field_name)
        #需要更新参数的层记录下来
        self._optimize_layers = [self._embed_combine_layer]

        self._hidden_layers=[]
        pre_out_dim=self._embed_combine_layer.output_dim+self._dense_combine_layer.output_dim
        for layer_index,layer_shape in enumerate(hidden_units):
            hidden_layer=DenseLayer(name='hidden{}'.format(layer_index),shape=[pre_out_dim,layer_shape])
            self._hidden_layers.append(hidden_layer)
            self._optimize_layers.append(hidden_layer)
            #ReLU层没有参数 不需要记录
            self._hidden_layers.append(ReLU())
            pre_out_dim=layer_shape

        '''
            对于一个二分类的问题,最终应该输出 正样本概率 就是1维的 数 因此 单独加一个 最终层,但是sigmoid函数就不放在这里了
        '''
        final_logit_layer= DenseLayer(name="final_logit", shape=[pre_out_dim, 1], l2reg=L2)
        self._hidden_layers.append(final_logit_layer)
        self._optimize_layers.append(final_logit_layer)

    def forward(self,features):
        '''
        在X中dense input 放在 sparse input 前面
        :param features: dict, mapping from field=>dense ndarray or field=>SparseInput
        :return: logits, [batch_size]
        '''
        dense_input=self._dense_combine_layer.forward(features)
        embedded_input=self._embed_combine_layer.forward(features)
        #print(embedded_input.shape)
        X=np.hstack([dense_input,embedded_input])
        for hidden_layer in self._hidden_layers:
            X=hidden_layer.forward(X)

        return X.flatten()

    def backward(self,prev_grads):
        '''

        :param prev_grads:gradient from loss to logist,[batch_size]
        :return:
        '''
        prev_grads=prev_grads.reshape([-1,1])# 行数永远都是batch_size
        #求梯度要反向
        for hidden_layer in self._hidden_layers[::-1]:
            prev_grads=hidden_layer.backward(prev_grads)
        #X的把DenseInput放在了前面,Dense部分没有梯度
        col_sizes=[self._dense_combine_layer.output_dim,self._embed_combine_layer.output_dim]
        _,grads_for_all_embedding=utils.split_columns(prev_grads,col_sizes)
        self._embed_combine_layer.backward(grads_for_all_embedding)
        #======优化部分=================
        all_vars={}
        all_grad2var={}
        for opt_layer in self._optimize_layers:
            all_vars.update(opt_layer.Variables)
            all_grad2var.update(opt_layer.grad2var)
        self._optimizer.update(all_vars,all_grad2var)


DeepHParams = namedtuple("DeepHParams",
                         ['dense_fields', 'vocab_infos', 'embed_fields', 'hidden_units', 'L2', 'optimizer'])


class DeepEstimator(BaseEstimator):
    def __init__(self,data_source,hparams):
        super(DeepEstimator, self).__init__(data_source)
        self._dnn=DeepNetwork(dense_fields=hparams.dense_fields,
                                vocab_infos=hparams.vocab_infos,
                                embed_fields=hparams.embed_fields,
                                hidden_units=hparams.hidden_units,
                                L2=hparams.L2,
                                optimizer=hparams.optimizer)

    def train_batch(self,features,labels):
        logits=self._dnn.forward(features)
        pred_probas=1/(1+np.exp(-logits))
        grads2logits=pred_probas-labels
        self._dnn.backward(grads2logits)
        return pred_probas

    def predict(self,features):
        logits=self._dnn.forward(features)
        return 1 / (1 + np.exp(-logits))




