"""
Sequence models over landmark features of shape (B, T, J, C).

  bilstm       2-layer bidirectional LSTM over flattened per-frame features
  gru          2-layer bidirectional GRU over flattened per-frame features
  timesformer  TimeSformer-style transformer: every (frame, joint) pair is a token, and each
               block applies *divided space-time attention* (Bertasius et al., 2021): attention
               across time for each joint, then across joints within each frame. The original
               TimeSformer does this over image patches; here the "patches" are landmarks, which
               makes it small enough for real-time CPU / TFLite inference.

All models use fixed-length input (T frames resampled per clip), so there is no padding mask and
the graphs export cleanly to TFLite.
"""

import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, num_joints, channels, num_classes, cell="lstm", hidden=192, layers=2,
                 dropout=0.3):
        super().__init__()
        in_dim = num_joints * channels
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(4 * hidden), nn.Dropout(dropout),
                                  nn.Linear(4 * hidden, num_classes))

    def forward(self, x):                         # x: (B, T, J, C)
        b, t = x.shape[:2]
        h = self.inp(x.reshape(b, t, -1))
        h, _ = self.rnn(h)
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=-1)  # mean + max over time
        return self.head(pooled)


class DividedSpaceTimeBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):                         # x: (B, T, J, D)
        b, t, j, d = x.shape
        # Temporal attention: each joint attends over its own trajectory.
        xt = x.permute(0, 2, 1, 3).reshape(b * j, t, d)
        h = self.norm_t(xt)
        xt = xt + self.attn_t(h, h, h, need_weights=False)[0]
        x = xt.reshape(b, j, t, d).permute(0, 2, 1, 3)
        # Spatial attention: joints within a frame attend to each other.
        xs = x.reshape(b * t, j, d)
        h = self.norm_s(xs)
        xs = xs + self.attn_s(h, h, h, need_weights=False)[0]
        x = xs.reshape(b, t, j, d)
        return x + self.mlp(self.norm_m(x))


class LandmarkTimeSformer(nn.Module):
    def __init__(self, num_joints, channels, num_classes, num_frames=32, dim=96, depth=4,
                 heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(channels, dim)
        self.joint_pos = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, num_frames, 1, dim))
        nn.init.trunc_normal_(self.joint_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList(DividedSpaceTimeBlock(dim, heads, dropout=dropout)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def forward(self, x):                         # x: (B, T, J, C)
        h = self.embed(x) + self.joint_pos + self.time_pos[:, : x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h).mean(dim=(1, 2)))


MODELS = ("bilstm", "gru", "timesformer")


def build_sequence_model(name, num_joints, channels, num_classes, num_frames=32, **kw):
    if name == "bilstm":
        return RNNClassifier(num_joints, channels, num_classes, cell="lstm", **kw)
    if name == "gru":
        return RNNClassifier(num_joints, channels, num_classes, cell="gru", **kw)
    if name == "timesformer":
        return LandmarkTimeSformer(num_joints, channels, num_classes, num_frames=num_frames, **kw)
    raise ValueError(f"Unknown model {name}; choose from {MODELS}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
