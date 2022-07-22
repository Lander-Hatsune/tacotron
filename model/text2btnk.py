""" adapted from https://github.com/NVIDIA/tacotron2 """

from math import sqrt

import torch
from torch import nn

from .attention import LocationSensitiveAttention, AttentionWrapper
from .attention import get_mask_from_lengths
from .modules import Prenet, BatchNormConv1dStack

class Encoder(nn.Module):
    """Encoder module:
        - A stack of three 1-d convolution layers, containing 512 filters with shape 5*1,
          followd by Batch Normalization (BN) and ReLU activations
        - Bidirectional LSTM
    """
    def __init__(self, embed_dim,
                 num_convs=3, conv_channels=512, conv_kernel_size=5,
                 conv_dropout=0.5, blstm_units=512):
        super(Encoder, self).__init__()

        # convolution layers followed by batch normalization and ReLU activation
        activations = [nn.ReLU()] * num_convs
        conv_out_channels = [conv_channels] * num_convs
        self.conv1ds = BatchNormConv1dStack(embed_dim, conv_out_channels, kernel_size=conv_kernel_size,
                                            stride=1, padding=(conv_kernel_size -1) // 2,
                                            activations=activations, dropout=conv_dropout)

        # 1 layer Bi-directional LSTM
        self.lstm = nn.LSTM(conv_channels, blstm_units // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        # transpose to (B, embed_dim, T) for convolution,
        # and then back
        x = self.conv1ds(x.transpose(1, 2)).transpose(1, 2)

        # (B, T, conv_channels)
        # TODO: pack_padded, and pad_packed?
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class Tacotron2(nn.Module):
    def __init__(self, model_cfg, n_vocab, embed_dim=512, mel_dim=80,
                 max_decoder_steps=1000, stop_threshold=0.5, r=3):
        super(Tacotron2, self).__init__()

        self.mel_dim = mel_dim

        # Embedding
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        std = sqrt(2.0 / (n_vocab + embed_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Encoder
        encoder_cfg = model_cfg["encoder"]
        encoder_out_dim = encoder_cfg["blstm_units"]
        self.encoder = Encoder(embed_dim, **encoder_cfg)

    def parse_data_batch(self, batch):
        """Parse data batch to form inputs and targets for model training/evaluating
        """
        # use same device as parameters
        device = next(self.parameters()).device

        text, text_length, btnk, btnk_length = batch
        text = text.to(device).long()
        text_length = text_length.to(device).long()
        btnk = btnk.to(device).float()

        return (text, text_length), (btnk)

    def forward(self, inputs):
        inputs, input_lengths = inputs

        B = inputs.size(0)

        # (B, T)
        inputs = self.embedding(inputs)

        # (B, T, embed_dim)
        encoder_outputs = self.encoder(inputs)

        return encoder_outputs

    def inference(self, inputs):
        # Only text inputs
        inputs = inputs, None, None
        return self.forward(inputs)


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, predicts, targets):
        btnk_target = targets
        btnk_target.requires_grad = False

        btnk_predict = predicts.repeat(
            [1, btnk_target.size(1) // predicts.size(1) + 1, 1])

        btnk_loss = nn.MSELoss()(btnk_predict[:, :btnk_target.size(1)],
                                 btnk_target)
        
        return btnk_loss
