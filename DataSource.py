from  Dataset import Dataset

class DataSource:
    '''
    用于封装dataset 在train_batch_per_epoch 中调用 get_batch_stream
    '''
    def __init__(self,train_path,test_path,batch_size):
        self.train_data=Dataset(train_path)
        self.test_data=Dataset(test_path)
        self.batch_size=batch_size

    def train_batches_per_epoch(self):
        '''
        :return:生成器 被 Estimator::train_epoch 调用
        yield: Xs,dict field_name:Sparse_Input or Dense_Input
               Ys array label
        '''
        return self.train_data.get_batch_stream(self.batch_size, n_repeat=1)

    def test_batches_per_epoch(self):
        return self.test_data.get_batch_stream(self.batch_size, n_repeat=1)

    @property
    def n_train_examples(self):
        return self.train_data.n_examples

    @property
    def n_test_examples(self):
        return self.test_data.n_examples





