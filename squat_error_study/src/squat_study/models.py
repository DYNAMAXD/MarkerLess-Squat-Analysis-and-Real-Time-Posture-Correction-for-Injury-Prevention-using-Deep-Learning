"""
models.py   (MODULE)
====================
All neural models. They share ONE interface so the training code is identical for each:

    logits, alpha = model(joints, frame, tempo)
        joints [B, T, J, 6]   position + velocity of each joint in the body frame
        frame  [B, T, F]      frame-wise angle / geometry features
        tempo  [B, 4]         rep duration, descent, ascent, bottom fraction
        logits [B, n_classes]
        alpha  [B, T]         "which frames mattered" (None for models without attention pooling)

Models (names used in configs and result tables)
------------------------------------------------
    cnn1d        1D temporal CNN                          (sequence baseline)
    bigru_attn   Bi-GRU + attention pooling               (sequence baseline, common in squat papers)
    bilstm       Bi-LSTM + mean pooling                   (sequence baseline)
    stgcn        spatio-temporal graph convolution        (skeleton baseline)
    transformer  spatial + temporal attention, alpha      (the proposed model, from ai_lead_handoff)

The tree / linear baselines (decision_tree, logreg, random_forest, xgboost) live in train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

# ----------------------------------------------------------------------------
# 1. Skeleton graph (used by ST-GCN)
# ----------------------------------------------------------------------------
BONES = [("Hip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RBigToe"),
         ("RAnkle", "RSmallToe"), ("RAnkle", "RHeel"),
         ("Hip", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LBigToe"),
         ("LAnkle", "LSmallToe"), ("LAnkle", "LHeel"),
         ("Hip", "Neck"), ("Neck", "Head"), ("Head", "Nose"),
         ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
         ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist")]


def skeleton_adjacency(joint_names):
    """Symmetric-normalised adjacency with self loops, shape [J, J]."""
    ix = {n: i for i, n in enumerate(joint_names)}
    J = len(joint_names)
    A = torch.eye(J)
    for a, b in BONES:
        if a in ix and b in ix:
            A[ix[a], ix[b]] = A[ix[b], ix[a]] = 1.0
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


# ----------------------------------------------------------------------------
# 2. Shared pieces
# ----------------------------------------------------------------------------
class AttnPool(nn.Module):
    """Attention pooling over time: returns pooled vector [B,d] and the weights alpha [B,T]."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, h):                                   # h [B,T,d]
        alpha = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (alpha.unsqueeze(-1) * h).sum(1), alpha


class Head(nn.Module):
    """Concatenates the pooled representation with the tempo numbers and classifies."""

    def __init__(self, d, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 4, d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d, n_classes))

    def forward(self, rep, tempo):
        return self.net(torch.cat([rep, tempo], dim=1))


def flat_input(joints, frame):
    """[B,T,J,6] + [B,T,F] -> [B,T,J*6+F]."""
    return torch.cat([joints.flatten(2), frame], dim=2)


# ----------------------------------------------------------------------------
# 3. Sequence baselines
# ----------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, J, F, n_classes, d=64, dropout=0.1, **_):
        super().__init__()
        c_in = J * 6 + F
        w = [d, 2 * d, 2 * d]
        layers, c = [], c_in
        for o in w:
            layers += [nn.Conv1d(c, o, 5, padding=2), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(dropout)]
            c = o
        self.conv = nn.Sequential(*layers)
        self.head = Head(2 * c, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        h = self.conv(flat_input(joints, frame).transpose(1, 2))      # [B,C,T]
        rep = torch.cat([h.mean(2), h.amax(2)], dim=1)
        return self.head(rep, tempo), None


class RNNClassifier(nn.Module):
    """Bi-GRU (attention pooling) or Bi-LSTM (mean pooling)."""

    def __init__(self, J, F, n_classes, d=64, dropout=0.1, cell="gru", pooling="attn", **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(J * 6 + F, d), nn.GELU())
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(2 * d) if pooling == "attn" else None
        self.head = Head(2 * d, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        h, _ = self.rnn(self.proj(flat_input(joints, frame)))          # [B,T,2d]
        if self.pool is not None:
            rep, alpha = self.pool(h)
        else:
            rep, alpha = h.mean(1), None
        return self.head(rep, tempo), alpha


# ----------------------------------------------------------------------------
# 4. ST-GCN (skeleton baseline)
# ----------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, dropout, k=9):
        super().__init__()
        self.register_buffer("A", A)
        self.spatial = nn.Conv2d(c_in, c_out, 1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.temporal = nn.Sequential(nn.Conv2d(c_out, c_out, (k, 1), padding=(k // 2, 0)),
                                      nn.BatchNorm2d(c_out), nn.Dropout(dropout))
        self.res = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):                                   # x [B,C,T,J]
        y = torch.einsum("bctj,jk->bctk", self.spatial(x), self.A)
        y = self.temporal(Fn.gelu(self.bn1(y)))
        return Fn.gelu(y + self.res(x))


class STGCN(nn.Module):
    def __init__(self, J, F, n_classes, joint_names, d=64, dropout=0.1, **_):
        super().__init__()
        A = skeleton_adjacency(joint_names)
        self.blocks = nn.Sequential(STGCNBlock(6, d // 2, A, dropout),
                                    STGCNBlock(d // 2, d, A, dropout),
                                    STGCNBlock(d, d, A, dropout))
        self.frame_mlp = nn.Sequential(nn.Linear(F, d // 2), nn.GELU())
        self.head = Head(d + d // 2, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        x = joints.permute(0, 3, 1, 2)                      # [B,6,T,J]
        g = self.blocks(x).mean(dim=(2, 3))                 # [B,d]
        f = self.frame_mlp(frame).mean(1)                   # [B,d/2]
        return self.head(torch.cat([g, f], dim=1), tempo), None


# ----------------------------------------------------------------------------
# 5. The proposed model: spatio-temporal transformer with frame attention (alpha)
# ----------------------------------------------------------------------------
class STBlock(nn.Module):
    """spatial attention over the tokens of one frame -> temporal attention over the frames -> MLP"""

    def __init__(self, d, heads, dropout, spatial, temporal):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ta = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):                                   # x [B,T,N,d]  (N = J joint tokens + 1 frame token)
        B, T, N, d = x.shape
        if self.spatial:
            h = self.n1(x).reshape(B * T, N, d)
            x = x + self.sa(h, h, h, need_weights=False)[0].reshape(B, T, N, d)
        else:                                               # ablation: no attention, plain mean mixing
            mixed = x[:, :, :-1].mean(2, keepdim=True)
            x = torch.cat([x[:, :, :-1], x[:, :, -1:] + mixed], dim=2)
        if self.temporal:
            h = self.n2(x).permute(0, 2, 1, 3).reshape(B * N, T, d)
            x = x + self.ta(h, h, h, need_weights=False)[0].reshape(B, N, T, d).permute(0, 2, 1, 3)
        return x + self.mlp(self.n3(x))


class SquatTransformer(nn.Module):
    def __init__(self, J, F, n_classes, T=64, d_model=64, n_blocks=4, n_heads=4, dropout=0.1,
                 spatial_attn=True, temporal_attn=True, pooling="attn", **_):
        super().__init__()
        self.J = J
        self.joint_embed = nn.Linear(6, d_model)
        self.frame_embed = nn.Linear(F, d_model)
        self.joint_id = nn.Parameter(torch.zeros(1, 1, J + 1, d_model))     # one id per token (last = frame token)
        self.time_pos = nn.Parameter(torch.zeros(1, T, 1, d_model))
        nn.init.normal_(self.joint_id, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)
        self.blocks = nn.ModuleList([STBlock(d_model, n_heads, dropout, spatial_attn, temporal_attn)
                                     for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model) if pooling == "attn" else None
        self.head = Head(d_model, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        tok = torch.cat([self.joint_embed(joints), self.frame_embed(frame).unsqueeze(2)], dim=2)
        x = tok + self.joint_id + self.time_pos
        for blk in self.blocks:
            x = blk(x)
        h = self.norm(x[:, :, -1])                          # the T frame tokens [B,T,d]
        if self.pool is not None:
            rep, alpha = self.pool(h)
        else:
            rep, alpha = h.mean(1), None
        return self.head(rep, tempo), alpha


# ----------------------------------------------------------------------------
# 6. Factory
# ----------------------------------------------------------------------------
NEURAL_MODELS = ["cnn1d", "bigru_attn", "bilstm", "stgcn", "transformer"]


def build_model(name, J, F, T, n_classes, joint_names, mcfg):
    d, dr = mcfg.get("d_model", 64), mcfg.get("dropout", 0.1)
    if name == "cnn1d":
        return CNN1D(J, F, n_classes, d=d, dropout=dr)
    if name == "bigru_attn":
        return RNNClassifier(J, F, n_classes, d=d, dropout=dr, cell="gru", pooling="attn")
    if name == "bilstm":
        return RNNClassifier(J, F, n_classes, d=d, dropout=dr, cell="lstm", pooling="mean")
    if name == "stgcn":
        return STGCN(J, F, n_classes, joint_names, d=d, dropout=dr)
    if name == "transformer":
        return SquatTransformer(J, F, n_classes, T=T, d_model=d, n_blocks=mcfg.get("n_blocks", 4),
                                n_heads=mcfg.get("n_heads", 4), dropout=dr,
                                spatial_attn=mcfg.get("spatial_attn", True),
                                temporal_attn=mcfg.get("temporal_attn", True),
                                pooling=mcfg.get("pooling", "attn"))
    raise ValueError(f"unknown neural model: {name}")


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
