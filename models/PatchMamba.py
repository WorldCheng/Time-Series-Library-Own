import torch
from torch import nn
import math
from mamba_ssm import Mamba
from layers.Embed import PatchEmbedding

"""
export CUDA_VISIBLE_DEVICES=0
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model PatchMamba \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --d_ff 16 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --patience 5 \
  --train_epochs 15\
  --itr 1
"""

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    PatchMamba: Combining Patching from PatchTST and SSM from Mamba
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # 1. Patching and Embedding
        # 将 [bs, seq_len, n_vars] 映射为 [bs * n_vars, patch_num, d_model]
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # 2. Mamba Layers (替换 Transformer Encoder)
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=configs.d_model,
                d_state=configs.d_ff, 
                d_conv=configs.d_conv,
                expand=configs.expand,
            ) for _ in range(configs.e_layers)
        ])
        
        # 归一化层 (保持 PatchTST 的维度习惯)
        self.norm = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))

        # 3. Prediction Head
        # 计算 Patch 数量：patch_num = (seq_len - patch_len) / stride + 2 (包含padding)
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * self.patch_num
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc):
        # Instance Normalization (防止非平稳性)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patching: [bs, seq_len, n_vars] -> [bs, n_vars, seq_len] -> [bs * n_vars, patch_num, d_model]
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Mamba Processing
        for mamba in self.mamba_layers:
            enc_out = mamba(enc_out)
        
        # 归一化处理
        enc_out = self.norm(enc_out)

        # Reshape back: [bs * nvars, patch_num, d_model] -> [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # Transpose for FlattenHead: [bs, nvars, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Head & De-Normalization
        dec_out = self.head(enc_out)  # [bs, nvars, target_window]
        dec_out = dec_out.permute(0, 2, 1) # [bs, target_window, nvars]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc)
        # 其他任务（imputation等）逻辑可参照 forecast 类似实现
        return None