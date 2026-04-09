import torch
import torch.utils.data as data
from torch import nn

import mylib
import mypreprocesslib as mpplib


def read_data_nmt():
    """ 载入英语-法语数据集"""
    with open('../fra.txt', 'r') as f:
        return f.read()


def preprocess_nmt(text):
    """ 预处理英语-法语数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """ tokenize 英语-法语数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列  \n
    为了以num_steps相同的小批量进行加载，通过截断或填充 \n
    使每个文本序列具有相同的长度
    """
    if len(line) > num_steps:
        return line[:num_steps]     # 截断
    return line + [padding_token] * (num_steps - len(line))     # 填充


def build_array_nmt(lines, vocab, num_steps):
    """
    将文本序列转换为小批量数据集。\n
    将<eos>添加到所有序列末尾，并记录了文本序列长度(排除了padding_token)
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """ 
    返回翻译数据集的迭代器和词表
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    # 防止词表过大，将出现少于2次的低频词元视为<unk>
    src_vocab = mpplib.Vocab(source, min_freq=2,
                             reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = mpplib.Vocab(target, min_freq=2,
                             reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def sequence_mask(X, valid_len, value=0):
    """屏蔽序列中不相关的项"""
    maxlen = X.shape[1]

    # 用[None, :]表示将形状从(maxlen,) -> (1, maxlen)
    # 用[:, None]表示将形状从(batch_size,) -> (batch_size, 1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]

    X[~mask] = value
    return X


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """seq2seq模型训练"""

    # 对于线性层和循环层，Xavier 初始化通常能带来更好的训练稳定性
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = mylib.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = mylib.Timer()
        metric = mylib.Accumulator(2)   # 损失训练总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 解码器输入:   [<bos>, ^y1, ^y2, ... ]
            # 目标:         [y1, y2, ..., <eos>]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学(每次使用真实的上一词元作为当前输入, 并行计算&防止错误累积)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            mylib.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """seq2seq的预测"""
    # 将net设置为评估模式
    net.eval()

    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 将序列转换为张量并添加batch_size轴: (batch_size=1, num_steps)
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)

    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 初始化解码器输入为<bos> (1, 1)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        # 推理时不使用强制教学
        dec_X = Y.argmax(dim=2)
        # squeeze移除第0维, 形状变为(1,), item()将包含单个元素的张量转换为 Python 标量整数
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq








class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """ 
    带mask的softmax交叉熵损失函数, mask位置的loss不参与计算。
    过滤掉loss中填充词元产生的不相关预测
    pred:       (batch_size, num_steps, vocab_size)
    label:      (batch_size, num_steps)
    valid_len:  (batch_size,)
    """
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'

        # nn.CrossEntropyLoss 要求输入(N, C), target:(N,) N为batch_size, C为类别数
        # 高维情形: 输入(N, C, d1, d2, ...), target: (N, d1, d2, ...)
        # 因此用permute对齐维数

        # unweighted_loss: (batch_size, num_steps)
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        # 这里原书为weighted_loss = (unweighted_loss * weights).mean(dim=1)     
        # 但我认为这里应该除以有效位数比较好吧
        # clamp(min=1.0)防止valid_len为0
        weighted_loss = (unweighted_loss * weights).sum(dim=1) / valid_len.clamp(min=1.0)
        weighted_loss[valid_len == 0] = 0.0

        return weighted_loss



    