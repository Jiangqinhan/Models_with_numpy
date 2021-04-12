import numpy as np
class Adagrad:
    def __init__(self,lr):
        '''

        :param lr:学习率
        _sum_grad2: Adagrad 是一种自适应学习率的优化方法 学习率分母为根号 二阶动量和
        道理是 经常大幅度更新的变量,以后更新幅度更小
        '''
        self._lr=lr
        self._sum_grad2={}

    def update(self,variables,gradients):
        '''

        :param variables:
        :param gradients: dict{gradient_name:val}
        :return:
        '''
        for gradname,gradient in gradients.items():
            g2=gradient*gradient
            if gradname in self._sum_grad2:
                self._sum_grad2[gradname]+=g2
            else:
                self._sum_grad2[gradname]=g2

            delta=self._lr*gradient/(np.sqrt(self._sum_grad2[gradname])+ 1e-6)
            '''
            对变量值进行update embedding层的gradient 名字为vocab_name@feature_id W
            而对应的变量名为vocab
            '''
            if '@'in gradname:
                varname,row=gradname.split('@')
                row=int(row)
                variables[varname][row,:]-=delta

            else:
                variables[gradname]-=delta
