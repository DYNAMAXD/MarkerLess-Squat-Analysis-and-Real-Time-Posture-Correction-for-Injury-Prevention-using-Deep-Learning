"""
models.py - the models of the study. Every neural model has one interface:

    logits, alpha = model(joints [B,T,J,6], frame [B,T,F], tempo [B,4])
    alpha [B,T] = attention-pooling weights over the frames ("which frames mattered"); None for mean / max pooling

Classical baselines (engine.py, on the 46-number summary row + 4 tempo numbers): logreg, random_forest, xgboost.
Neural models (all end in attention pooling by default; config model.pooling = attn | mean | max | meanmax):
    cnn1d           temporal 1D convolutions over the per-frame vector
    bilstm          bidirectional LSTM over the per-frame vector
    bigru           bidirectional GRU over the per-frame vector
    cnn_lstm        1D convolutions (LayerNorm, residual) then a bidirectional LSTM
    tcn             temporal convolutional network: residual blocks of dilated convolutions (dilation 1, 2, 4, 8)
    st_transformer  spatio-temporal transformer (hand-off design): joint + frame tokens, spatial then temporal attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

TREE_MODELS = ["logreg", "random_forest", "xgboost"]
NEURAL_MODELS = ["cnn1d", "bilstm", "bigru", "cnn_lstm", "tcn", "st_transformer"]
TRANSFORMERS = ["st_transformer"]

# what each model receives and how it processes it (used by the per-model reports)
MODEL_INFO = {
    "logreg": dict(family="classical baseline", input="summary row: 46 statistics per rep from rep_features.csv (min / max / at-bottom of each pipeline angle) + 4 tempo numbers = 50 values",
                   processing="median imputation of missing values, z-score, multinomial logistic regression (L2, C = 1); sample weights: every recording counts equally"),
    "random_forest": dict(family="classical baseline", input="summary row (46) + tempo (4) = 50 values per rep",
                          processing="median imputation, 300 decision trees (bootstrap), majority of tree probabilities"),
    "xgboost": dict(family="classical baseline", input="summary row (46) + tempo (4) = 50 values per rep",
                    processing="median imputation, 300 gradient-boosted trees, depth 4, learning rate 0.1, row / column subsampling 0.9"),
    "cnn1d": dict(family="convolutional", input="per frame: 22 joints x 6 (position + velocity, body frame) + 19 frame features = 151 channels, 64 frames",
                  processing="3 Conv1d layers (kernel 5; 64 -> 128 -> 128 channels; BatchNorm, GELU, dropout): each output sees 13 frames; attention pooling over the 64 frames -> alpha; + 4 tempo numbers -> classifier"),
    "bilstm": dict(family="recurrent", input="per frame: 151 channels (as 1D-CNN), 64 frames",
                   processing="linear projection to 64, 2-layer bidirectional LSTM (64 per direction): every frame summarises the whole rep in both directions; attention pooling -> alpha; + tempo -> classifier"),
    "bigru": dict(family="recurrent", input="per frame: 151 channels, 64 frames",
                  processing="linear projection to 64, 2-layer bidirectional GRU (64 per direction); attention pooling -> alpha; + tempo -> classifier"),
    "cnn_lstm": dict(family="hybrid", input="per frame: 151 channels, 64 frames",
                     processing="3 Conv1d blocks (kernel 5, 64 channels, LayerNorm, residual) extract local movement, then a 2-layer bidirectional LSTM (32 per direction) relates them over the rep; attention pooling (32) -> alpha; tempo through a small MLP -> classifier"),
    "tcn": dict(family="convolutional", input="per frame: 151 channels, 64 frames",
                processing="input projection to 64 channels, 4 residual blocks of two dilated Conv1d (kernel 3, dilation 1, 2, 4, 8; non-causal): the receptive field covers the whole 64-frame rep; attention pooling -> alpha; + tempo -> classifier"),
    "st_transformer": dict(family="attention", input="per frame: 22 joint tokens (6 values each) + 1 frame token (19 features) = 23 tokens, 64 frames",
                           processing="tokens embedded to 64 + learned joint and time tags; 4 blocks of [spatial self-attention over the 23 tokens of a frame -> temporal self-attention over the 64 frames of each token -> MLP 64-256-64]; the 64 frame tokens -> attention pooling (32) -> alpha; tempo MLP -> classifier"),
}


# ----------------------------------------------------------------------------------------------- pooling + head
class AttnPool(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, hidden or d), nn.Tanh(), nn.Linear(hidden or d, 1, bias=False))

    def forward(self, h):
        alpha = torch.softmax(self.score(h).squeeze(-1), dim=1)
        return (alpha.unsqueeze(-1) * h).sum(1), alpha


class Pool(nn.Module):
    """[B,T,d] -> (rep, alpha). kind: attn (alpha) | mean | max | meanmax (rep is 2d wide)."""

    def __init__(self, d, kind="attn", hidden=None):
        super().__init__()
        self.kind = kind
        self.attn = AttnPool(d, hidden) if kind == "attn" else None
        self.out_dim = 2 * d if kind == "meanmax" else d

    def forward(self, h):
        if self.kind == "attn":
            return self.attn(h)
        if self.kind == "mean":
            return h.mean(1), None
        if self.kind == "max":
            return h.amax(1), None
        return torch.cat([h.mean(1), h.amax(1)], dim=1), None


class Head(nn.Module):
    """[rep ; tempo] -> classes; tempo_mlp: tempo through MLP(4 -> 64 -> 64) first."""

    def __init__(self, d, n_classes, dropout, tempo_mlp=False, hidden=None):
        super().__init__()
        self.tempo = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64)) if tempo_mlp else None
        t = 64 if tempo_mlp else 4
        self.net = nn.Sequential(nn.Linear(d + t, hidden or d), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden or d, n_classes))

    def forward(self, rep, tempo):
        return self.net(torch.cat([rep, self.tempo(tempo) if self.tempo is not None else tempo], dim=1))


def flat_input(joints, frame):
    return torch.cat([joints.flatten(2), frame], dim=2)


# ----------------------------------------------------------------------------------------------- models
class CNN1D(nn.Module):
    def __init__(self, J, F, n_classes, d=64, dropout=0.1, pooling="attn", kernel=5, layers=3, dilated=False, **_):
        super().__init__()
        chans = [d] + [2 * d] * (layers - 1)
        mods, c = [], J * 6 + F
        for i, o in enumerate(chans):
            dil = 2 ** i if dilated else 1
            mods += [nn.Conv1d(c, o, kernel, padding=dil * (kernel - 1) // 2, dilation=dil), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(dropout)]
            c = o
        self.conv = nn.Sequential(*mods)
        self.pool = Pool(c, pooling)
        self.head = Head(self.pool.out_dim, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        rep, alpha = self.pool(self.conv(flat_input(joints, frame).transpose(1, 2)).transpose(1, 2))
        return self.head(rep, tempo), alpha


class RNNClassifier(nn.Module):
    def __init__(self, J, F, n_classes, d=64, dropout=0.1, cell="gru", pooling="attn", rnn_layers=2, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(J * 6 + F, d), nn.GELU())
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(d, d, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0)
        self.pool = Pool(2 * d, pooling)
        self.head = Head(self.pool.out_dim, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        rep, alpha = self.pool(self.rnn(self.proj(flat_input(joints, frame)))[0])
        return self.head(rep, tempo), alpha


class CNNLSTM(nn.Module):
    def __init__(self, J, F, n_classes, d=64, dropout=0.1, pooling="attn", rnn_layers=2, kernel=5, layers=3, **_):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(J * 6 + F if i == 0 else d, d, kernel, padding=kernel // 2) for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d, d // 2, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0)
        self.norm = nn.LayerNorm(d)
        self.pool = Pool(d, pooling, 32)
        self.head = Head(self.pool.out_dim, n_classes, dropout, tempo_mlp=True)

    def forward(self, joints, frame, tempo):
        x = flat_input(joints, frame)
        for i, (conv, ln) in enumerate(zip(self.convs, self.norms)):
            h = self.drop(Fn.gelu(ln(conv(x.transpose(1, 2)).transpose(1, 2))))
            x = x + h if i > 0 else h
        rep, alpha = self.pool(self.norm(self.lstm(x)[0]))
        return self.head(rep, tempo), alpha


class TCNBlock(nn.Module):
    def __init__(self, c, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(dropout),
                                 nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    """Temporal convolutional network (Bai et al. 2018), non-causal: the whole rep is available offline."""

    def __init__(self, J, F, n_classes, d=64, dropout=0.1, pooling="attn", kernel=3, levels=4, **_):
        super().__init__()
        self.inp = nn.Conv1d(J * 6 + F, d, 1)
        self.blocks = nn.Sequential(*[TCNBlock(d, kernel, 2 ** i, dropout) for i in range(levels)])
        self.pool = Pool(d, pooling)
        self.head = Head(self.pool.out_dim, n_classes, dropout)

    def forward(self, joints, frame, tempo):
        h = self.blocks(self.inp(flat_input(joints, frame).transpose(1, 2))).transpose(1, 2)
        rep, alpha = self.pool(h)
        return self.head(rep, tempo), alpha


class STBlock(nn.Module):
    def __init__(self, d, heads, dropout, spatial, temporal, mlp_mult=4):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ta = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_mult * d, d), nn.Dropout(dropout))

    def forward(self, x):
        B, T, N, d = x.shape
        if self.spatial:
            h = self.n1(x).reshape(B * T, N, d)
            x = x + self.sa(h, h, h, need_weights=False)[0].reshape(B, T, N, d)
        else:
            x = torch.cat([x[:, :, :-1], x[:, :, -1:] + x[:, :, :-1].mean(2, keepdim=True)], dim=2)
        if self.temporal:
            h = self.n2(x).permute(0, 2, 1, 3).reshape(B * N, T, d)
            x = x + self.ta(h, h, h, need_weights=False)[0].reshape(B, N, T, d).permute(0, 2, 1, 3)
        return x + self.mlp(self.n3(x))


class STTransformer(nn.Module):
    def __init__(self, J, F, n_classes, T=64, d=64, n_blocks=4, n_heads=4, dropout=0.1, spatial_attn=True, temporal_attn=True,
                 pooling="attn", **_):
        super().__init__()
        self.joint_embed, self.frame_embed = nn.Linear(6, d), nn.Linear(F, d)
        self.token_id = nn.Parameter(torch.randn(1, 1, J + 1, d) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, T, 1, d) * 0.02)
        self.blocks = nn.ModuleList(STBlock(d, n_heads, dropout, spatial_attn, temporal_attn) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        self.pool = Pool(d, pooling, 32)
        self.head = Head(self.pool.out_dim, n_classes, dropout, tempo_mlp=True, hidden=64)

    def forward(self, joints, frame, tempo):
        x = torch.cat([self.joint_embed(joints), self.frame_embed(frame).unsqueeze(2)], dim=2)
        x = x + self.token_id + self.time_pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        rep, alpha = self.pool(self.norm(x[:, :, -1]))
        return self.head(rep, tempo), alpha


# ----------------------------------------------------------------------------------------------- factory
def build_model(name, J, F, T, n_classes, joint_names, mcfg):
    d, dr, pl = mcfg.get("d_model", 64), mcfg.get("dropout", 0.1), mcfg.get("pooling", "attn")
    if name == "cnn1d":
        return CNN1D(J, F, n_classes, d=d, dropout=dr, pooling=pl, kernel=mcfg.get("kernel", 5), layers=mcfg.get("conv_layers", 3),
                     dilated=mcfg.get("dilated", False))
    if name in ("bilstm", "bigru"):
        return RNNClassifier(J, F, n_classes, d=d, dropout=dr, cell="lstm" if name == "bilstm" else "gru", pooling=pl,
                             rnn_layers=mcfg.get("rnn_layers", 2))
    if name == "cnn_lstm":
        return CNNLSTM(J, F, n_classes, d=d, dropout=dr, pooling=pl, rnn_layers=mcfg.get("rnn_layers", 2),
                       kernel=mcfg.get("kernel", 5), layers=mcfg.get("conv_layers", 3))
    if name == "tcn":
        return TCN(J, F, n_classes, d=d, dropout=dr, pooling=pl, kernel=mcfg.get("tcn_kernel", 3), levels=mcfg.get("tcn_levels", 4))
    if name == "st_transformer":
        return STTransformer(J, F, n_classes, T=T, d=d, n_blocks=mcfg.get("n_blocks", 4), n_heads=mcfg.get("n_heads", 4), dropout=dr,
                             spatial_attn=mcfg.get("spatial_attn", True), temporal_attn=mcfg.get("temporal_attn", True), pooling=pl)
    raise ValueError(f"unknown neural model {name}")


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
