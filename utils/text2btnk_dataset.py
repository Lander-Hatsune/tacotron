import os
import random
import numpy as np
import torch

from text import text_to_sequence

class TextBtnkDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads btnk from .npy files
    """
    def __init__(self, fname, hparams):
        self.text_cleaners = hparams.text_cleaners
        self.symbols_lang = hparams.symbols_lang
        self.btnk_dim = hparams.btnk_dim
        self.f_list = self.files_to_list(fname)
        random.seed(hparams.seed)
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                # btnk_file_path
                path = 'dataset/btnks/' + parts[0] + '.npy'
                # text
                text = parts[1].strip()
                f_list.append([text, path])
        return f_list

    def get_btnk_text_pair(self, text, file_path):
        text = self.get_text(text)
        btnk = self.get_btnk(file_path)
        return (text, btnk)

    def get_btnk(self, file_path):
        # stored btnk: np.ndarray [shape=(1, T_out, btnk_dim)]
        btnk = torch.from_numpy(np.load(file_path)).squeeze()
        assert btnk.size(1) == self.btnk_dim, (
            'Btnk dimension mismatch: given {}, expected {}'.format(btnk.size(1), self.btnk_dim))

        return btnk

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, self.symbols_lang))
        return text_norm

    def __getitem__(self, index):
        return self.get_btnk_text_pair(*self.f_list[index])

    def __len__(self):
        return len(self.f_list)
    

class TextBtnkCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, r):
        self.r = r

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad btnk
        num_btnks = batch[0][1].size(1)
        max_target_len = max([x[1].size(0) for x in batch])
        if max_target_len % self.r != 0:
            max_target_len += self.r - max_target_len % self.r
            assert max_target_len % self.r == 0

        # include btnk padded
        btnk_padded = torch.FloatTensor(len(batch), max_target_len, num_btnks)
        btnk_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            btnk = batch[ids_sorted_decreasing[i]][1]
            btnk_padded[i, :btnk.size(0), :] = btnk
            output_lengths[i] = btnk.size(0)

        return text_padded, input_lengths, btnk_padded, output_lengths
