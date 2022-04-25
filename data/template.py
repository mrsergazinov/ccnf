import torch
from torch.utils.data import Dataset

import numpy as np

class DataSet(Dataset):
  def __init__(self, seqs, length, pred_len):
    # info
    self.length = length
    self.pred_len = pred_len
    self.labels = [seq[0] for seq in seqs]
    self.seqs = [seq[1] for seq in seqs]
    
    # init
    len_idx = np.array([len(seq) for seq in self.seqs])
    len_idx = len_idx - self.length - self.pred_len + 1
    self.len_idx = np.insert(np.cumsum(len_idx), 0, 0)
  
  def __getitem__(self, index):

    idx_seq = np.argmax(self.len_idx > index) - 1
    seq = self.seqs[idx_seq]
    label = self.labels[idx_seq]

    s_begin = index - self.len_idx[idx_seq]
    s_end = s_begin + self.length
    r_begin = s_end
    r_end = r_begin + self.pred_len

    seq_x = seq[s_begin:s_end]
    seq_y = seq[r_begin:r_end]

    return label, seq_x, seq_y
  
  def __len__(self):
    return self.len_idx[-1]