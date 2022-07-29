import os
import time
import argparse
import json

import torch
import numpy as np
from torch.utils.data import DataLoader

from model.tacotron import Tacotron, TacotronLoss
from model.text2btnk import Tacotron2, Tacotron2Loss
from utils.text2btnk_dataset import TextMelDataset, TextMelCollate
from utils.logger import TacotronLogger
from utils.utils import data_parallel_workaround
from hparams import create_hparams
from text2btnk_train import *

def evaluate(model, iteration, device, valset, collate_fn):
    """Evaluate on validation set, get validation loss and printing
    """
    model.eval()
    with torch.no_grad():
        valdata_loader = DataLoader(valset, sampler=None, num_workers=1,
                                    shuffle=False, batch_size=1,
                                    pin_memory=False, collate_fn=collate_fn)

        #val_loss = 0.0
        for i, batch in enumerate(valdata_loader):
            inputs, targets = model.parse_data_batch(batch)
            predicts = model(inputs)
            spec_predicts = predicts[1].cpu()

            print(spec_predicts.shape)
            np.save(f'evalout/{i}.npy', spec_predicts)
            if i >= 3:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='out',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_dir', type=str, default='log',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')

    args = parser.parse_args()
    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, criterion = create_model(hparams)
    model = model.to(device)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)
    trainset, valset, collate_fn = prepare_datasets(hparams)

    model, optimizer, _learning_rate, iteration = load_checkpoint(
        'out/checkpoint_8000', model, optimizer)
    
    evaluate(model, iteration, device, valset, collate_fn)
