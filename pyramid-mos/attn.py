'''
def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked_:
        raise NotImplementedError

    attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
    return attention


def input_fun(**config):
    data = tf.random_normal((
        config['batch_size'], config['sequence_length'], config['hidden_dim']))
    return data


def model_fun(data, **config):
    Q = tf.layers.dense(data, config['hidden_dim'])  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(data, config['hidden_dim'])  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(data, config['n_classes'])  # [batch_size, sequence_length, n_classes]

    attention = attention_fun(Q, K)  # [batch_size, sequence_length, sequence_length]
    output = tf.matmul(attention, V)  # [batch_size, sequence_length, n_classes]
    return output
'''
import torch
import torch.utils.data as data
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.FC = nn.Linear(11904, 1)

    def forward(self, Q, K, V):
        Q = torch.transpose(Q, 0, 1)
        K = torch.transpose(K, 0, 1)
        K = torch.transpose(K, 1, 2)
        attn = torch.bmm(Q, K)/torch.sqrt(torch.tensor(K.shape[-1]))
        attn_soft = self.softmax(attn)
        V = torch.transpose(V, 0, 1)
        res = torch.bmm(attn_soft, V)
        res = torch.flatten(res, start_dim=1)
        print("Flattened:", res)
        linear_out = self.FC(res)
        return linear_out