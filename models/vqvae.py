"""
Simplified VQ-VAE model for inference only
"""
import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder_wo_upsample
from models.quantize_cnn import QuantizeEMAReset


class VQVAE_251(nn.Module):
    """VQ-VAE for 251-dim motion features"""
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
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
        self.code_dim = code_dim
        self.num_code = nb_code

        input_dim = 251 if args.dataname == 'kit' else 263

        # Encoder
        self.encoder = Encoder(
            input_dim, output_emb_width, down_t, 1, width, depth,
            dilation_growth_rate, activation=activation, norm=norm, causal=causal
        )

        # Decoder
        self.decoder = Decoder_wo_upsample(
            input_dim, output_emb_width, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm
        )

        # Quantizer
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)

    def preprocess(self, x):
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def forward_decoder(self, x):
        """Decode from codebook indices"""
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    """Human motion VQ-VAE wrapper"""
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
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
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(
            args, nb_code, code_dim, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation=activation,
            norm=norm, causal=causal
        )

    def forward_decoder(self, x):
        """Decode from codebook indices"""
        x_out = self.vqvae.forward_decoder(x)
        return x_out
