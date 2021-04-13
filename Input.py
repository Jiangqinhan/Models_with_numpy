import numpy as np
from Base_class import Layer
class SparseInput:
    """
    如何表示稀疏输入，很费了一番思考
    TensorFlow中是用sp_ids, sp_weights两上SparseTensor来表示，但是这两个SparseTensor中的indices, dense_shape必须完全相同，是重复的

    后来考虑使用KVPair = namedtuple('KVPair', ['idx_in_batch', 'id', 'value'])表示一个非零特征
    整个稀疏输入就是list of KVPair，程序处理上是方便了很多，但是每个KVPair都是一个namedtuple，生成了大多的small object，会给GC造成压力

    后来还考虑使用一个[n_nonzero, 3]的ndarray来表示，
    第0列是idx_in_batch（行号）
    第1列是id
    第2列是数值
    但是因为ndarray只能有一个dtype，为了容纳value，整个ndarray必须是float，处理起行号和id这样的整数，既不方便，也浪费了空间

    目前决定采用3个list的方式来表示一个理论、稠密形状为[batch_size, max_bag_size]的稀疏输入
    所谓max_bag_size，是一个理论概念，可以认为infinity，在代码中并不出现，也不会对代码造成限制
    比如表示用户行为历史，max_bag_size可以是用户一段历史内阅读的文章数、购买的商品数
    比如表示用户的手机使用习惯，max_bag_size可以是所有app的数目
    这里，我们将这些信息表示成一个bag，而不是sequence，忽略其中的时序关系

    第一个list example_indices: 是[n_non_zeros]的整数数组，表示在[batch_size, max_bag_size]中的行号（样本序号），>=0 and < batch_size
                               而且要求其中的数值是从小到大，排好序的
    第二个list feature_ids:     是[n_non_zeros]的整数数组，表示非零元对应特征的序号，可以重复
    第三个list feature_values:  是[n_non_zeros]的浮点数组，表示非零元对应特征的数值
    举例来说，第i个非零元(0<=i<n_non_zeros)
    它对应哪个样本？example_indices[i]
    它对应哪个特征？feature_ids[i]
    它的数值是多少？values[i]
    """
    def __init__(self,n_total_examples,example_indices,feature_ids,feature_values):
        self.example_indices=example_indices
        self.feature_ids=feature_ids
        self.feature_values=feature_values
        self.n_total_examples = n_total_examples
        self.__nnz_idx=0

    def add(self, example_idx, feat_id, feat_val):
        '''
        构建三个list
        用在Dataset::get_batch_stream中
        '''
        self.example_indices.append(example_idx)
        self.feature_ids.append(feat_id)
        self.feature_values.append(feat_val)


    def get_non_zeros(self):
        '''
        用于需要迭代的场合
        :return:
        '''
        return zip(self.example_indices, self.feature_ids, self.feature_values)

    def print(self):
        print('example indices',self.example_indices)
        print('feature id',self.feature_ids)
        print('feature values',self.feature_values)

    def __move_to_next_example(self,nnz_idx):
        '''
            返回当前样本的所有feature id和feature value
            并把nnz_index移动到下一个样本的起始位置
        '''
        if nnz_idx>len(self.example_indices):
            return None
        end=nnz_idx+1
        while end<len(self.example_indices) and self.example_indices[end]==self.example_indices[nnz_idx]:
            end+=1
        current_feat_ids = self.feature_ids[nnz_idx:end]
        current_feat_vals = self.feature_values[nnz_idx:end]

        return end, current_feat_ids, current_feat_vals


    def get_example_in_order(self,example_idx):
        '''

        :param example_id:
        :return: example_id的非0feat_id 和feat_val
        '''
        if self.__nnz_idx >= len(self.example_indices):
            return [], []

        elif self.example_indices[self.__nnz_idx] == example_idx:
            self.__nnz_idx, feat_ids, feat_vals = self.__move_to_next_example(self.__nnz_idx)
            return feat_ids, feat_vals

        elif self.example_indices[self.__nnz_idx] > example_idx:
            # 等待调用者下次传入更大的example_idx
            return [], []

        else:
            # 如果当前example_index并不是调用者需要的example_idx
            # 则一定是比外界需要用example_idx大，等待调用者传入更大的example_idx
            # 如果比比外界需要用example_idx小，说明调用方式不对
            raise ValueError("incorrect invocation")


class  DenseInputCombineLayer(Layer):
    def __init__(self,field_sizes):
        '''

        :param field_sizes: list[(field_name,size)]
        在forward中用于从input里选出 DenseInput
        '''
        self._field_sizes=field_sizes

    def forward(self,input):
        '''
        :param input: dict{field_name:SparseInput,field_name:DenseInput}
        :return:
        '''
        outputs=[]
        for field_name,in_dim in self._field_sizes:
            tmp=np.asarray(input[field_name])
            #print(input[field_name])
            #print(tmp.shape)
            outputs.append(tmp)
        return np.hstack(outputs)

    @property
    def output_dim(self):
        return sum(in_dim for _,in_dim in self._field_sizes)

