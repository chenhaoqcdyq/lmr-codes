"""
Simplified Encoder/Decoder for inference only
"""
import torch.nn as nn
from models.resnet import Resnet1D


class Encoder(nn.Module):
    """CNN Encoder for motion sequences"""
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 causal=False):
        super().__init__()

        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2

        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            input_dim = width
            if causal:
                causal_pad = (filter_t-1)
                block = nn.Sequential(
                    nn.ConstantPad1d((causal_pad,0), 0),
                    nn.Conv1d(input_dim, width, filter_t, stride_t, 0),
                    Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
                )
            else:
                block = nn.Sequential(
                    nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
                )
            blocks.append(block)

        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder_wo_upsample(nn.Module):
    """Decoder without upsampling"""
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)

        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
