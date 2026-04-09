import torch
import math
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.shape[1]
    # PyTorch和Numpy中使用None来插入一个维度，从而自动广播
    mask = torch.arange((maxlen), dtype=torch.float32, 
                        device=X.device)[None, :] < valid_len[:, None]
    # 按位取反运算符~在 PyTorch 中对布尔张量被重载为逐元素逻辑非
    # Python中普通~是一个按位取反运算符（bitwise NOT）
    # 它对整数的二进制表示进行逐位取反操作 ~x == -x-1
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """在最后一个轴上掩蔽元素再执行softmax操作"""
    # X: 3D张量，valid_lens: 1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 展平
            valid_lens = valid_lens.reshape(-1)
        # 将被遮蔽的元素用很大的负值替换，从而softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算改变形状"""
    # 输入X: (batch_size, queries或k-v个数, num_hiddens)
    # --> (~, ~, num_heads, num_hiddens / num_heads)
    # reshape: 按内存顺序(行优先)重新填入
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # --> (~, num_heads, queries或k-v个数, ~)
    # permute: 先遍历第0维，然后第2维，然后第1维，最后第3维
    X = X.permute(0, 2, 1, 3)

    # --> (batch_size * num_heads, ~, ~)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """transpose_qkv的逆操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
    
        
class AdditiveAttention(nn.Module):
    """加性注意力"""
    # key_size为key的特征维度数, query_size为query的特征维度数
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hiddens)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries维数: (batch_size, queries个数, queries维数=d)
    # keys维数: (batch_size, key-value对个数, keys维数=d)
    # values维数: (batch_size, key-value对个数, 值的维度)
    # valid_lens维数: (batch_size, )或(batch_size, queries个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    

class MultiHeadAttention(nn.Module):
    """
    多头注意力 \n
    此处设定p_q = p_k = p_v = p_o / h, 以防计算代价和参数代价增长
    """
    # p_q = p_k = p_v = num_hiddens / num_heads

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)      
        # 这里分开考虑W_o作用在一个head上 
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """ 
        Inputs: 
        queries, keys, values:  (batch_size, no. of queries or k-vs, 
                                query/key/value _size)
        valid_lens: (batch_size, ) or (batch_size, no. of queries)

        Outputs:
        self.W_o(output_concat):    (batch_size, no. of queries, num_hiddens)
        """

        # LHS queries, keys, values: 
        # (batch_size * num_heads, no. of queries or k-vs, 
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )
        
        # output: (batch_size * num_heads, no. of queries,
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)

        return self.W_o(output_concat)
    

# Positional Encoding 位置编码
class PositionalEncoding(nn.Module):
    """
    位置编码 \n
    输入X: (n, d)为一个序列中n个词元的d维嵌入表示 
    位置编码使用相同形状的嵌入矩阵P: (n, d) \n
    输出X + P \n
    p_{i, 2j} = sin(i / 10000^(2j/d))
    p_{i, 2j+1} = cos(i / 10000^(2j/d))
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个行数足够大的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / 
             torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens))
        
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    