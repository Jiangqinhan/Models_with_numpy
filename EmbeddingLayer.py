from Base_class import Layer
import numpy as np
from Initialization import  TruncatedNormal
import utils

class EmbeddingLayer(Layer):
    """
    简化起见，不支持use_bias和regularization
    不支持regularization的原因是：weight是稠密的，自然L2 Loss的gradient也是稠密的
    为了L2 Loss而破坏稀疏性，增加内容与耗时，有些得不偿失
    一种改进方案是：只正则化本batch中用到的embedding向量
    embedding layer 与field对应
    W与 vocab对应
    一个vocab会被多个field用所以这里不储存 dw,dw存在combine里
    """
    def __init__(self, W,vocab_name,field_name):
        super(EmbeddingLayer, self).__init__()
        self.vocab_name = vocab_name
        self.field_name = field_name
        self._W = W
        self._last_input = None

    @property
    def output_dim(self):
        return self._W.shape[1]

    def forward(self,X):
        '''

        :param X: SparseInput
        :return: [batch_size,embed_size]
        '''
        self._last_input=X
        output=np.zeros((X.n_total_examples,self._W.shape[1]))
        for (example_id,feature_id,feature_val) in X.get_non_zeros():
            output[example_id,:]+=self._W[feature_id,:]*feature_val

        return output

    def backward(self,prev_grad):
        '''

        :param prev_grad:
        :return: dW dict{feature_id:gradient}
        '''
        dW={}
        X=self._last_input
        for (example_id,feature_id,feature_val) in X.get_non_zeros():
            tmp=prev_grad[example_id,:]*feature_val
            if feature_id in dW:
                dW[feature_id]+=tmp
            else:
                dW[feature_id]=tmp
        return dW

class EmbeddingCombineLayer(Layer):
    '''
    field name 很重要 在forward中要根据fieldname 将SparseInput 选出来
    '''

    def __init__(self, vocab_infos):
        '''
        根据给定的vocab信息list [(vocab_name, vocab_size, embed_size)]
        _weights 存储W
        _grad_to_embed 存储dw
        _embed_layer 存储fieldd对应的embedding层
        '''
        super(EmbeddingCombineLayer,self).__init__()
        self._weights={}
        for vocab_name, vocab_size, embed_size in vocab_infos:
            # TruncatedNormal是TF WDL中embedding_column的默认初始化方式
            # These values are similar to values from a `random_normal_initializer`
            # except that values more than two standard deviations from the mean are discarded and re-drawn
            stddev = 1 / np.sqrt(embed_size)
            initializer = TruncatedNormal(mean=0,
                                          stddev=stddev,
                                          lower=-2 * stddev,
                                          upper=2 * stddev)
            self._weights[vocab_name] = initializer(shape=[vocab_size, embed_size])
        self._grad_to_weights={}
        self._embed_layers=[]

    def add_embedding(self,vocab_name,field_name):
        self._embed_layers.append(EmbeddingLayer(self._weights[vocab_name],vocab_name,field_name))

    @property
    def output_dim(self):
        return sum(layer.output_dim for layer in self._embed_layers)

    def forward(self,sparse_inputs):
        '''

        :param sparse_inputs: dict{field_name:SparseInput,field_name:DenseInput}
        :return: 每个SparseInput贡献一个embedding vector，返回结果是这些embedding vector的拼接
                    拼接顺序由add_embedding的调用顺序决定
        '''
        embed_output=[]
        for layer in self._embed_layers:
            embed_output.append(layer.forward(sparse_inputs[layer.field_name]))
        return np.hstack(embed_output)

    def backward(self,prev_grads):
        # 因为output是每列输出的拼接，自然上一层输入的导数也是各层所需要导数的拼接
        # prev_grads_splits是一个数组，存储对应各层的导数
        col_sizes=[layer.output_dim for layer in self._embed_layers]
        prev_grads=utils.split_columns(prev_grads,col_sizes)
        '''
           clear 这一步很重要, 每次求导得把之前的梯度清空
        '''
        self._grad_to_weights.clear()
        for layer,layer_prev_grads in zip(self._embed_layers,prev_grads):
            # layer_prev_grads: 上一层传入的，Loss对某个layer的输出的梯度
            # layer_grads_to_feat_embed: dict, feat_id==>grads，
            # 由这一个layer造成对某vocab的embedding矩阵的某feat_id对应行的梯度
            layer_grads_to_embed = layer.backward(layer_prev_grads)
            for feat_id,dw in layer_grads_to_embed.items():
                # 表示"对某个vocab的embedding weight中的第feat_id行的总导数"
                key = "{}@{}".format(layer.vocab_name, feat_id)
                if key in self._grad_to_weights:
                    self._grad_to_weights[key]+=dw
                else:
                    self._grad_to_weights[key]=dw

    @property
    def Variables(self):
        return self._weights

    @property
    def grad2var(self):
        return self._grad_to_weights

    @property
    def l2reg_loss(self):
        return 0




