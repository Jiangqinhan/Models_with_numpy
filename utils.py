import logging
def chunk(stream,chunk_size):
    '''
    将数据集分为batchsize大小
    最后一块不够的话 也一样输出
    每次输出[特征行]
    '''
    buf=[]
    for line in stream:
        buf.append(line)
        if(len(buf)==chunk_size):
            yield buf
            del buf[:]
    if len(buf)>0:
        yield buf

def split_columns(prev_grads,col_sizes):
    '''
    把 hstack后的embedding向量拆开
    :param prev_grads:
    :param col_sizes:
    :return:
    '''
    offset=0
    result=[]
    for size in col_sizes:
        result.append(prev_grads[:,offset:(offset+size)])
        offset+=size
    return result

def config_logging(fname):
    logging.basicConfig(level=logging.INFO, format='%(message)s')  # re-format to remove prefix 'INFO:root'

    fh = logging.FileHandler(fname)
    fh.setLevel(logging.INFO)
    logging.getLogger("").addHandler(fh)