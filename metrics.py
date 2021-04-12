import numpy as np
import pandas as pd
def tied_rank(x):
    '''
    last_rank 未被写入r的第一个数
    if sorted_x[i][0]!=cur_val: 这里的i 表示第一个 不是 curval的数
    所以 边界情况是 [4, 4, 4, 4] 对应代码是 注意到 i+2 而不是i+1
            if i==len(sorted_x)-1:
            for j in range(last_rank,i+1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0

    '''
    sorted_x=sorted(zip(x,range(len(x))))
    r=[0 for k in x]
    last_rank=0
    cur_val=sorted_x[0][0]
    for i in range(len(sorted_x)):
        if sorted_x[i][0]!=cur_val:
            cur_val=sorted_x[i][0]
            for j in range(last_rank,i):
                r[sorted_x[j][1]]=(last_rank+i+1)/2.0
            last_rank=i
        if i==len(sorted_x)-1:
            for j in range(last_rank,i+1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r

def logloss(y_true,y_pred,normalize=True):
    '''
    二项分布极大似然估计,y_pred表示正样本概率
    :param y_true:
    :param y_pred:
    :param normalize:
    :return:
    '''
    #print('y_true is ',y_true)
    #print('y_pred is',y_pred)
    loss_array = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    if normalize:
        return np.mean(loss_array)
    else:
        return np.sum(loss_array)

def auc(y_true,y_score):
    '''
    result=[sigma(ranki  i belong to positive sample)-M*(M+1)] /(M*N)
    M为正样本个数 N为负样本个数
    :param y_true:
    :param y_score:
    :return:
    '''
    r=tied_rank(y_score)
    M=len([0 for x in y_true if x==1])
    N=len(y_true)-M
    sum_rank=sum([r[i] for i in range(len(r)) if y_true[i]==1])
    auc=(sum_rank-M*(M+1)/2)/(M*N)
    return auc
