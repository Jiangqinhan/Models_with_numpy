import pandas as pd
import sys
import numpy as np
import random
import utils
from Input import SparseInput
import bisect
#import argparse
from tqdm import tqdm

VOCAB_LISTS = {
    'education': ['Bachelors',
                  'HS-grad',
                  '11th',
                  'Masters',
                  '9th',
                  'Some-college',
                  'Assoc-acdm',
                  'Assoc-voc',
                  '7th-8th',
                  'Doctorate',
                  'Prof-school',
                  '5th-6th',
                  '10th',
                  '1st-4th',
                  'Preschool',
                  '12th'],

    'marital_status': ['Married-civ-spouse',
                       'Divorced',
                       'Married-spouse-absent',
                       'Never-married',
                       'Separated',
                       'Married-AF-spouse',
                       'Widowed'],

    'relationship': ['Husband',
                     'Not-in-family',
                     'Wife',
                     'Own-child',
                     'Unmarried',
                     'Other-relative'],

    'workclass': ['Self-emp-not-inc',
                  'Private',
                  'State-gov',
                  'Federal-gov',
                  'Local-gov',
                  'Self-emp-inc',
                  'Without-pay',
                  'Never-worked'],

    'occupation': ['Tech-support',
                   'Craft-repair',
                   'Other-service',
                   'Sales',
                   'Exec-managerial',
                   'Prof-specialty',
                   'Handlers-cleaners',
                   'Machine-op-inspct',
                   'Adm-clerical',
                   'Farming-fishing',
                   'Transport-moving',
                   'Priv-house-serv',
                   'Protective-serv',
                   'Armed-Forces']
}
#例子{education:{'Bachelors':0,...}}
VOCAB_MAPPINGS = {field: {featname: idx for idx, featname in enumerate(featnames)} for field, featnames in
                  VOCAB_LISTS.items()}

AGE_BOUNDARIES = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

DENSE_FIELDS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

DENSE_LOG_MEAN_STD = {'age': (3.6183599219864133, 0.35003117354646957),
                      'education_num': (2.372506496597371, 0.27381608590073075),
                      'capital_gain': (0.7346209104536965, 2.4547377400238553),
                      'capital_loss': (0.35030508122367104, 1.5845809727578963),
                      'hours_per_week': (3.665366478972777, 0.38701441353280025)}

CATEGORY_FIELDS = ['education', 'marital_status', 'relationship', 'workclass', 'occupation', 'age_buckets']

class Dataset:
    '''
    主要工作是划分batch,作为Datasource类实际工作的类
    '''
    def __init__(self,fname):
        '''
        fieldname 列表和对应值矩阵[[field1_value,...]]
        '''
        with open(fname,'rt') as fin:
            self._field_names = fin.readline().strip().split(',')
            self._lines = [line.strip() for line in fin]
    def parse_line(self,line):
        '''
        对于categories 用onehot
        对年龄 按照段分进去
        对dense 数据 log(1+x)平滑 再归一化
        return {feature_name:value}
        对于dense数据 值就是值
        对于sparse数据 值为vocab与自然数的对应 即 feature编号

        '''
        features={}
        contents=dict(zip(self._field_names,line.split(',')))
        label=label = int(contents['income_bracket'] == '>50K')
        for field in ['education', 'marital_status', 'relationship', 'workclass', 'occupation']:
            vocab_mapping=VOCAB_MAPPINGS[field]
            txt_val=contents[field]
            if txt_val in vocab_mapping:
                features[field]=vocab_mapping[txt_val]

        age = int(contents['age'])
        features['age_buckets'] = bisect.bisect(AGE_BOUNDARIES, age)
        for field in DENSE_FIELDS:
            raw_data=float(contents[field])
            logmean, logstd = DENSE_LOG_MEAN_STD[field]
            features[field]=(np.log1p(raw_data)-logmean)/logstd

        return features,label


    def get_batch_stream(self,batch_size,n_repeat):
        '''
        生成器 在DataSource::
        :param batch_size:
        :param n_repeat: 重复遍历数据集次数
        :yield: Xs,dict field_name:Sparse_Input or Dense_Input
                 Ys array label
        '''
        n_repeat = n_repeat if n_repeat > 0 else sys.maxsize
        for _ in range(n_repeat):
            random.shuffle(self._lines)
            for chunk in utils.chunk(self._lines,batch_size):
                Xs={}
                Ys=[]
                for field_name in CATEGORY_FIELDS:
                    Xs[field_name]=SparseInput(len(chunk),[],[],[])

                for field_name in DENSE_FIELDS:
                    #dense input 对应的是[[example1_value],[example2_value]]
                    Xs[field_name]=[]

                for example_id,line in enumerate(chunk):
                    features,label=self.parse_line(line)
                    Ys.append(label)
                    for field in CATEGORY_FIELDS:
                        '''
                        这里主要是针对缺失值,产生的实际效果是缺失的都设为0
                        '''
                        if field in features:
                            Xs[field].add(example_id,features[field],1)

                    for field in DENSE_FIELDS:
                        Xs[field].append([features[field]])

                yield Xs,np.asarray(Ys)

    @property
    def n_examples(self):
        return len(self._lines)

