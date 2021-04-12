from Base_class import Layer
import numpy as np
import Initialization
class DenseLayer(Layer):
    def __init__(self,name,shape,l2reg=0,init_method='glorot_uniform'):
        '''
        目前只支持一种初始化方式
        :param shape: [in_dim,out_dim]
        :param l2reg:
        :param init_method:
        _W和_b 都是array
        '''
        self._l2reg=l2reg
        self._W = Initialization.get_global_init(init_method)(shape)
        self._b = Initialization.get_global_init('zero')(shape[1])
        self._dW=None
        self._db=None
        self._last_input=None
        self._name=name

    def forward(self,X):
        '''

        :param X:array [batch_size,in_dim]
        :return:
        '''
        self._last_input=X
        return np.dot(X,self._W)+self._b

    def backward(self,prev_grads):
        '''
        对应好维数即可轻松推出公式
        :param prev_grads:
        :return:
        '''
        self._dW=np.dot(self._last_input.T,prev_grads)+self._l2reg*self._W
        self._db=np.sum(prev_grads,axis=0)
        return np.dot(prev_grads,self._W.T)

    @property
    def l2reg_loss(self):
        '''
        sigma(wij**2)
        :return:
        '''
        return 0.5*self._l2reg*np.sum(self._W**2)

    @property
    def Variables(self):
        return {"{}_W".format(self._name):self._W,
                "{}_b".format(self._name):self._b}

    @property
    def grad2var(self):
        return {"{}_W".format(self._name): self._dW,
                "{}_b".format(self._name): self._db}