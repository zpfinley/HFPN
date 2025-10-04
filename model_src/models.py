import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import math

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class Encoder(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output

class Encoder_2(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, d_model, num_layers, nhead, dim_feedforward):
        super(Encoder_2, self).__init__()

        self.encoder_list = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

    def forward(self, src):

        for enc in self.encoder_list:
            src = enc(src)
        return src


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""EncoderLayer is mainly made up of self-attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Trans_Encoder(Module):

    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, d_model, num_layers, nhead, norm=None):
        super(Trans_Encoder, self).__init__()

        self.encoder_list = nn.ModuleList([EncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)])

    def forward(self, v, a):

        r"""Pass the input through the endocder layers in turn.

        """

        for enc in self.encoder_list:

            v = enc(v)
            a = enc(a)

        return v, a


class DecoderLayer(Module):
    r"""DecoderLayer is mainly made up of the proposed cross-modal relation attention (CMRA).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()

        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# model = PositionalEncoding(d_model=256)
# x = torch.rand(11, 32, 256)
# print(model(x).shape)

class ClassTokenFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(ClassTokenFusionLayer, self).__init__()

        self.visual_encoder = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        self.audio_encoder = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, visual, audio):

        visual = self.visual_encoder(visual)

        audio = self.audio_encoder(audio)

        return visual, audio


class ClassTokenFusionEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(ClassTokenFusionEncoder, self).__init__()

        self.vis_pos_emb = PositionalEncoding(d_model=256, dropout=0.1, max_len=11)
        self.aud_pos_emb = PositionalEncoding(d_model=256, dropout=0.1, max_len=11)

        self.visual_audio_encoder = ClassTokenFusionLayer(d_model, nhead)

        self.layers = _get_clones(self.visual_audio_encoder, num_layers)
        self.num_layers = num_layers

    def forward(self, visual, audio):

        # 初始化类别token
        v_class = torch.mean(visual, dim=0)
        a_class = torch.mean(audio, dim=0)

        # 相加
        cls_token = torch.mul(v_class + a_class, 0.5).unsqueeze(0)

        visual = torch.cat([cls_token, visual], dim=0)
        visual = self.vis_pos_emb(visual)
        audio = torch.cat([cls_token, audio], dim=0)
        audio = self.aud_pos_emb(audio)

        for i in range(self.num_layers):

            visual_out, audio_out = self.layers[i](visual, audio)

            v_class = visual_out[0, :, :].unsqueeze(0)
            a_class = audio_out[0, :, :].unsqueeze(0)

            cls_token = torch.mul(v_class + a_class, 0.5)

            visual = torch.cat([cls_token, visual_out[1:, :, :]], dim=0)
            audio = torch.cat([cls_token, audio_out[1:, :, :]], dim=0)

        return visual_out[1:, :, :], audio_out[1:, :, :], cls_token.squeeze()


# model = ClassTokenFusionEncoder(d_model=256, nhead=4, num_layers=2)
# visual = torch.rand(10, 32, 256)
# audio = torch.rand(10, 32, 256)
# out1, o2, o3 = model(visual, audio)
# print(out1.shape, o2.shape, o3.shape)




