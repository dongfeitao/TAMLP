import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import cv2
import torch.multiprocessing as mp
import random
import math
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== Global Random Seed Setup =====================
SEED = 42

def set_seed(seed=SEED):
    """Fix all random seeds to ensure full reproducibility of experiments"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
mp.set_start_method('spawn', force=True)

# ================ Global Hyperparameters (aligned with Table S1) ================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-3
train_epoch = 100
num_classes = 5
feature_dim = 128
n_fold = 5
n_head = 4
transformer_ffn_dim = 512
transformer_dropout = 0.3

# Alpha Evolution hyperparameters
N_pop = 50
G_max = 50
alpha_coeff = 0.8
theta_coeff = 0.3
lambda_fit = 0.01
neuron_candidates = [64, 128, 256]
w_lb, w_ub = -0.3, 0.3
max_neuron_global = 256  # Physical maximum neuron count, aligned with Eq.S1
sync_freq = 5  # Synchronous gradient update frequency
K_history = 5  # Number of individuals in history matrix B
d_b = 0.6  # Learning rate for evolution path b

# ==================== Dataset Path Configuration ====================
DATA_CONFIG = {
    "seu": {
        "index_csv": "./southeast_dataset_partition.csv",
        "image_root": "./se_fold_images",
        "cond2_img_path": "./se_cond2_images_fixed",
        "weight_save_dir": "./weights/seu"
    },
    "lab": {
        "index_csv": "./lab_dataset_partition.csv",
        "image_root": "./lab_fold_images",
        "cond2_img_path": "./lab_cond2_images_fixed",
        "weight_save_dir": "./weights/lab"
    }
}

# ==================== Network Modules ====================
class SingleCNN(nn.Module):
    """2D CNN feature extractor for single-channel image input"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128 * 32 * 32, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)
        return self.relu(self.fc(x))

class PositionalEncoding(nn.Module):
    """Fixed sine-cosine positional encoding"""
    def __init__(self, dim: int, seq_len: int = 3):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(1000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CrossAttention(nn.Module):
    """Cross-attention module for inter-domain feature interaction"""
    def __init__(self, dim):
        super().__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.d_k = dim

    def forward(self, q_in, kv_in):
        Q = self.wq(q_in)
        K = self.wk(kv_in)
        V = self.wv(kv_in)
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn_weight = torch.softmax(attn_score, dim=-1)
        cross_out = torch.matmul(attn_weight, V)
        return cross_out

class MultiHeadSelfAttn(nn.Module):
    """Multi-head self-attention with feed-forward network"""
    def __init__(self, dim, heads, ffn_hidden=512, dropout=0.3):
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads
        self.wqkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        qkv = self.wqkv(x).reshape(B, L, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn_w = torch.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn_w, v).permute(0, 2, 1, 3).reshape(B, L, D)
        x = self.norm1(residual + self.drop1(self.out_proj(attn_out)))
        residual2 = x
        x = self.ffn(x)
        x = self.norm2(residual2 + self.drop2(x))
        return x

class SerialAttnFusion(nn.Module):
    """Serial feature fusion: cross-attention followed by multi-head self-attention"""
    def __init__(self, dim=128, heads=4, seq_len=3):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim=dim, seq_len=seq_len)
        self.cross_att = CrossAttention(dim)
        self.multi_head_att = MultiHeadSelfAttn(
            dim=dim, heads=heads,
            ffn_hidden=transformer_ffn_dim,
            dropout=transformer_dropout
        )

    def forward(self, time_feat, freq_feat, tf_feat):
        B = time_feat.size(0)
        tokens_raw = torch.stack([time_feat, freq_feat, tf_feat], dim=1)
        x_embed = self.pos_enc(tokens_raw)
        q_in = x_embed[:, 2:3, :]
        kv_in = x_embed[:, 0:2, :]
        x_cross = self.cross_att(q_in, kv_in)
        x_recom = torch.cat([x_embed[:, 0:2, :], x_cross], dim=1)
        x_att = self.multi_head_att(x_recom)
        fuse_vec = torch.mean(x_att, dim=1)
        return fuse_vec

class FeatureBackbone(nn.Module):
    """Multi-branch CNN + Transformer feature fusion backbone"""
    def __init__(self):
        super().__init__()
        self.cnn_t = SingleCNN()
        self.cnn_f = SingleCNN()
        self.cnn_tf = SingleCNN()
        self.fusion = SerialAttnFusion(feature_dim, n_head)

    def forward(self, ti, fi, tfi):
        ft = self.cnn_t(ti)
        ff = self.cnn_f(fi)
        ftf = self.cnn_tf(tfi)
        fuse_feat = self.fusion(ft, ff, ftf)
        return fuse_feat

# ========== BaseMLP: Fixed max dimension + zero-mask for active neurons ==========
class BaseMLP(nn.Module):
    """
    MLP with fixed physical input dimension (256) and dynamic effective neuron count.
    Weight matrix always has shape [256, feature_dim] regardless of active_n.
    Neurons beyond active_n are zero-masked and do not contribute to forward propagation.
    """
    def __init__(self, max_in_dim=256):
        super().__init__()
        self.max_in_dim = max_in_dim
        self.in_layer = nn.Linear(feature_dim, self.max_in_dim)
        self.hid1 = nn.Linear(self.max_in_dim, 128)
        self.hid2 = nn.Linear(128, 128)
        self.hid3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, feat, active_n):
        x = self.relu(self.in_layer(feat))
        # Zero-mask: only first active_n neurons are effective
        mask = torch.zeros_like(x)
        mask[..., :active_n] = 1.0
        x = x * mask
        x = self.relu(self.hid1(x))
        x = self.relu(self.hid2(x))
        x = self.relu(self.hid3(x))
        return self.out(x)

# ==================== Dataset & Feature Extraction ====================
class FoldBearingDataset(Dataset):
    """
    Dataset for loading fold-wise images.
    Samples are sorted by sample ID to match partition CSV order for traceability.
    Class folders follow C/H/O/B/I convention.
    """
    def __init__(self, fold_img_root, split="train"):
        self.root = os.path.join(fold_img_root, split) if split else fold_img_root
        self.samples = []
        fault2label = {"C": 0, "H": 1, "O": 2, "B": 3, "I": 4}

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        for fault in os.listdir(self.root):
            fd = os.path.join(self.root, fault)
            if not os.path.isdir(fd):
                continue
            for fname in os.listdir(fd):
                if "_time.png" in fname:
                    base = fname.replace("_time.png", "")
                    self.samples.append({
                        "t": os.path.join(fd, f"{base}_time.png"),
                        "f": os.path.join(fd, f"{base}_freq.png"),
                        "tf": os.path.join(fd, f"{base}_tf.png"),
                        "label": fault2label[fault]
                    })

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {self.root}")

        # Sort samples by sample ID to exactly match partition CSV order
        self.samples.sort(key=lambda x: int(os.path.basename(x["t"]).split("_")[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        t = cv2.imread(item["t"], 0) / 255.0
        f = cv2.imread(item["f"], 0) / 255.0
        tf = cv2.imread(item["tf"], 0) / 255.0
        t = torch.from_numpy(t).float().unsqueeze(0)
        f = torch.from_numpy(f).float().unsqueeze(0)
        tf = torch.from_numpy(tf).float().unsqueeze(0)
        return t, f, tf, item["label"]

def extract_all_fixed_features(backbone, index_csv, image_root):
    """Extract fixed backbone features for all Condition 1 samples"""
    df = pd.read_csv(index_csv)
    df_cond1 = df[df["is_train"] == True].reset_index(drop=True)
    all_t, all_f, all_tf, all_lab, all_foldid = [], [], [], [], []

    fault_dir_map = {"D0": "C", "D1": "H", "D2": "O", "D3": "B", "D4": "I"}

    for _, row in df_cond1.iterrows():
        subfile_id = row["subfile_id"]
        fold_id = int(row["fold"])
        sid = int(row["sample_id"])
        fault = row["fault_label"]
        fd_name = fault_dir_map[fault]
        base_name = f"sample_{sid:06d}"

        t_path = os.path.join(image_root, f"fold{fold_id}_img/train/{fd_name}/{base_name}_time.png")
        f_path = os.path.join(image_root, f"fold{fold_id}_img/train/{fd_name}/{base_name}_freq.png")
        tf_path = os.path.join(image_root, f"fold{fold_id}_img/train/{fd_name}/{base_name}_tf.png")

        t = cv2.imread(t_path, 0) / 255.0
        f = cv2.imread(f_path, 0) / 255.0
        tf = cv2.imread(tf_path, 0) / 255.0
        t = torch.from_numpy(t).float().unsqueeze(0)
        f = torch.from_numpy(f).float().unsqueeze(0)
        tf = torch.from_numpy(tf).float().unsqueeze(0)

        all_t.append(t)
        all_f.append(f)
        all_tf.append(tf)
        all_lab.append(int(row["fault_label"][1]))
        all_foldid.append(fold_id)

    ti_tensor = torch.stack(all_t, dim=0)
    fi_tensor = torch.stack(all_f, dim=0)
    tfi_tensor = torch.stack(all_tf, dim=0)
    y_tensor = torch.tensor(all_lab, dtype=torch.long)
    foldid_tensor = torch.tensor(all_foldid, dtype=torch.long)

    backbone.eval()
    full_X = []
    full_Y = y_tensor
    with torch.no_grad():
        ds = TensorDataset(ti_tensor, fi_tensor, tfi_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        for ti, fi, tfi in dl:
            ti, fi, tfi = ti.to(device), fi.to(device), tfi.to(device)
            feat = backbone(ti, fi, tfi)
            full_X.append(feat.cpu())
    full_X = torch.cat(full_X, dim=0)
    return full_X, full_Y, foldid_tensor

def five_fold_evaluate_individual(args):
    """
    Full training + 5-fold validation for synchronous update steps.
    All candidates share the same global fixed hidden/output layer parameters.
    Implements learning rate decay and random scaling augmentation as specified in Table S1.
    """
    n_neuron, w_cpu, b_cpu, full_X, full_Y, foldid_tensor, sub_seed, lamb, max_n, fixed_hidden_state = args
    set_seed(sub_seed)
    acc_list = []

    for fold_idx in range(n_fold):
        # Fold index is 1-based in foldid_tensor
        mask_train = (foldid_tensor != (fold_idx + 1))
        mask_val = (foldid_tensor == (fold_idx + 1))
        X_tr = full_X[mask_train]
        Y_tr = full_Y[mask_train]
        X_vl = full_X[mask_val]
        Y_vl = full_Y[mask_val]

        mlp = BaseMLP(max_in_dim=256).to(device)
        mlp.in_layer.weight.data = w_cpu.clone().to(device)
        mlp.in_layer.bias.data = b_cpu.clone().to(device)

        # Load globally shared fixed hidden/output layer parameters
        mlp.hid1.weight.data = fixed_hidden_state['hid1.weight'].clone().to(device)
        mlp.hid1.bias.data = fixed_hidden_state['hid1.bias'].clone().to(device)
        mlp.hid2.weight.data = fixed_hidden_state['hid2.weight'].clone().to(device)
        mlp.hid2.bias.data = fixed_hidden_state['hid2.bias'].clone().to(device)
        mlp.hid3.weight.data = fixed_hidden_state['hid3.weight'].clone().to(device)
        mlp.hid3.bias.data = fixed_hidden_state['hid3.bias'].clone().to(device)
        mlp.out.weight.data = fixed_hidden_state['out.weight'].clone().to(device)
        mlp.out.bias.data = fixed_hidden_state['out.bias'].clone().to(device)

        # Freeze hidden and output layers, only update input layer
        for name, param in mlp.named_parameters():
            if "in_layer" not in name:
                param.requires_grad = False

        tr_ds = TensorDataset(X_tr, Y_tr)
        vl_ds = TensorDataset(X_vl, Y_vl)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        opt = optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=lr, weight_decay=1e-4)
        # Learning rate decay strategy: multiply by 0.5 every 10 epochs (aligned with Table S1)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        mlp.train()
        for _ in range(train_epoch):
            for xf, yf in tr_loader:
                xf, yf = xf.to(device), yf.to(device)
                # Random scaling data augmentation [0.9, 1.1] (aligned with Table S1)
                scale_factor = torch.FloatTensor(xf.size(0), 1).uniform_(0.9, 1.1).to(device)
                xf_aug = xf * scale_factor
                pred = mlp(xf_aug, active_n=n_neuron)
                loss = loss_fn(pred, yf)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

        # Validation
        mlp.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xf, yf in vl_loader:
                xf, yf = xf.to(device), yf.to(device)
                out = mlp(xf, active_n=n_neuron)
                pred = torch.argmax(out, dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(yf.cpu().numpy())
        fold_acc = accuracy_score(trues, preds)
        acc_list.append(fold_acc)

    acc5 = float(np.mean(acc_list))
    fit = 1.0 - acc5 + lamb * (n_neuron / max_n)
    return fit, acc5, n_neuron

def five_fold_infer_only(args):
    """Inference-only evaluation for intermediate iterations (no gradient update)"""
    n_neuron, w_cpu, b_cpu, full_X, full_Y, foldid_tensor, sub_seed, lamb, max_n, fixed_hidden_state = args
    set_seed(sub_seed)
    acc_list = []

    for fold_idx in range(n_fold):
        mask_val = (foldid_tensor == (fold_idx + 1))
        X_vl = full_X[mask_val]
        Y_vl = full_Y[mask_val]

        mlp = BaseMLP(max_in_dim=256).to(device)
        mlp.in_layer.weight.data = w_cpu.clone().to(device)
        mlp.in_layer.bias.data = b_cpu.clone().to(device)
        mlp.hid1.weight.data = fixed_hidden_state['hid1.weight'].clone().to(device)
        mlp.hid1.bias.data = fixed_hidden_state['hid1.bias'].clone().to(device)
        mlp.hid2.weight.data = fixed_hidden_state['hid2.weight'].clone().to(device)
        mlp.hid2.bias.data = fixed_hidden_state['hid2.bias'].clone().to(device)
        mlp.hid3.weight.data = fixed_hidden_state['hid3.weight'].clone().to(device)
        mlp.hid3.bias.data = fixed_hidden_state['hid3.bias'].clone().to(device)
        mlp.out.weight.data = fixed_hidden_state['out.weight'].clone().to(device)
        mlp.out.bias.data = fixed_hidden_state['out.bias'].clone().to(device)

        mlp.eval()
        preds, trues = [], []
        with torch.no_grad():
            vl_ds = TensorDataset(X_vl, Y_vl)
            vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            for xf, yf in vl_loader:
                xf, yf = xf.to(device), yf.to(device)
                out = mlp(xf, active_n=n_neuron)
                pred = torch.argmax(out, dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(yf.cpu().numpy())
        fold_acc = accuracy_score(trues, preds)
        acc_list.append(fold_acc)

    acc5 = float(np.mean(acc_list))
    fit = 1.0 - acc5 + lamb * (n_neuron / max_n)
    return fit, acc5, n_neuron

class AlphaEvolutionFull:
    """
    Full Alpha Evolution algorithm with global fixed hidden layers.
    Hidden/output layers are initialized once globally and shared across all individuals.
    Fitness uses 5-fold average accuracy; final model retrained on full Condition 1 data.
    """
    def __init__(self, full_X, full_Y, foldid_tensor, base_seed):
        self.full_X = full_X
        self.full_Y = full_Y
        self.foldid_tensor = foldid_tensor
        self.pop_size = N_pop
        self.G_max = G_max
        self.alpha = alpha_coeff
        self.theta = theta_coeff
        self.lamb = lambda_fit
        self.neuron_set = neuron_candidates
        self.w_lb = w_lb
        self.w_ub = w_ub
        self.max_n = max_neuron_global
        self.base_seed = base_seed
        self.sync_freq = sync_freq
        self.K = K_history
        self.d_b = d_b

        # ========== Global fixed hidden + output layer parameters (initialized once) ==========
        set_seed(base_seed)
        _temp_mlp = BaseMLP(max_in_dim=256).to('cpu')
        self.fixed_hidden_state = {
            'hid1.weight': _temp_mlp.hid1.weight.data.clone(),
            'hid1.bias': _temp_mlp.hid1.bias.data.clone(),
            'hid2.weight': _temp_mlp.hid2.weight.data.clone(),
            'hid2.bias': _temp_mlp.hid2.bias.data.clone(),
            'hid3.weight': _temp_mlp.hid3.weight.data.clone(),
            'hid3.bias': _temp_mlp.hid3.bias.data.clone(),
            'out.weight': _temp_mlp.out.weight.data.clone(),
            'out.bias': _temp_mlp.out.bias.data.clone(),
        }
        del _temp_mlp

    def bound_map(self, val_tensor):
        """Eq.(6) Half-distance boundary shrinkage strategy"""
        upper = torch.ones_like(val_tensor) * self.w_ub
        lower = torch.ones_like(val_tensor) * self.w_lb
        out = torch.where(val_tensor > self.w_ub, (val_tensor + upper) / 2, val_tensor)
        out = torch.where(out < self.w_lb, (out + lower) / 2, out)
        return out

    def init_population(self):
        """Initialize population with random input layer parameters"""
        pop = []
        for idx in range(self.pop_size):
            np.random.seed(self.base_seed + idx)
            n = int(np.random.choice(self.neuron_set))
            w_rand = torch.FloatTensor(256, feature_dim).uniform_(self.w_lb, self.w_ub)
            b_rand = torch.FloatTensor(256).uniform_(self.w_lb, self.w_ub)
            pop.append({"active_n": n, "w": w_rand, "b": b_rand})
        return pop

    def init_history_and_Q(self, pop, pool):
        """Initialize history matrix B and adaptive base vector Q"""
        task_list = []
        for idx, ind in enumerate(pop):
            task = (
                ind["active_n"], ind["w"].cpu(), ind["b"].cpu(),
                self.full_X, self.full_Y, self.foldid_tensor,
                self.base_seed + 10000 + idx, self.lamb, self.max_n,
                self.fixed_hidden_state
            )
            task_list.append(task)
        init_fits = pool.map(five_fold_infer_only, task_list)

        # Select top-K individuals by fitness (ascending = lower is better)
        fit_list = [f[0] for f in init_fits]
        sorted_idx = np.argsort(fit_list)
        self.B = []
        for i in range(self.K):
            idx = sorted_idx[i]
            self.B.append({
                "active_n": pop[idx]["active_n"],
                "w": pop[idx]["w"].clone(),
                "b": pop[idx]["b"].clone(),
                "fit": fit_list[idx]
            })

        # Initialize Q: weighted mean of history individuals
        self.Q_w = torch.stack([b["w"] for b in self.B], dim=0).mean(dim=0)
        self.Q_b = torch.stack([b["b"] for b in self.B], dim=0).mean(dim=0)
        self.Q_n = np.mean([b["active_n"] for b in self.B])

    def sample_H(self, pop):
        """Eq.(1) Sample with replacement to generate evolution matrix H"""
        idx = np.random.randint(0, self.pop_size, self.pop_size)
        H = [pop[i] for i in idx]
        return H

    def calculate_weights(self):
        """Eq.(5) Inverse fitness weighting (for minimization objective)"""
        fits = np.array([b["fit"] for b in self.B])
        inv_fits = 1.0 / (fits + 1e-8)  # Better fitness → higher weight
        omega = inv_fits / inv_fits.sum()
        return omega

    def update_Q(self):
        """Eq.(3) + Eq.(4) Dual-path update of adaptive base vector Q"""
        omega = self.calculate_weights()
        if np.random.rand() < 0.5:
            # Path b: Q + identity diagonal matrix weighting
            diag_A_w = torch.ones_like(self.Q_w)
            diag_A_b = torch.ones_like(self.Q_b)
            diag_A_n = 128.0
            self.Q_w = self.d_b * self.Q_w + (1 - self.d_b) * diag_A_w
            self.Q_b = self.d_b * self.Q_b + (1 - self.d_b) * diag_A_b
            self.Q_n = self.d_b * self.Q_n + (1 - self.d_b) * diag_A_n
        else:
            # Path c: weighted sum of history matrix
            w_list = [b["w"] * omega[i] for i, b in enumerate(self.B)]
            b_list = [b["b"] * omega[i] for i, b in enumerate(self.B)]
            n_list = [b["active_n"] * omega[i] for i, b in enumerate(self.B)]
            self.Q_w = torch.stack(w_list, dim=0).sum(dim=0)
            self.Q_b = torch.stack(b_list, dim=0).sum(dim=0)
            self.Q_n = np.sum(n_list)

    def alpha_update_full(self, m_ind, z_ind, u_ind):
        """Eq.(2) Full Alpha operator update with adaptive base vector Q"""
        n_m, w_m, b_m = m_ind["active_n"], m_ind["w"], m_ind["b"]
        n_z, w_z, b_z = z_ind["active_n"], z_ind["w"], z_ind["b"]
        n_u, w_u, b_u = u_ind["active_n"], u_ind["w"], u_ind["b"]

        dr_w = torch.randn_like(w_m)
        dr_b = torch.randn_like(b_m)
        dr_n = np.random.randn()

        new_w = self.Q_w + self.alpha * dr_w + self.theta * (w_z + w_m - self.Q_w - w_u)
        new_b = self.Q_b + self.alpha * dr_b + self.theta * (b_z + b_m - self.Q_b - b_u)
        new_n_float = self.Q_n + self.alpha * dr_n + self.theta * (n_z + n_m - self.Q_n - n_u)
        return new_n_float, new_w, new_b

    def update_B(self, best_ind, best_fit):
        """Update history matrix: add new best, remove oldest"""
        self.B.pop(0)
        self.B.append({
            "active_n": best_ind["active_n"],
            "w": best_ind["w"].clone(),
            "b": best_ind["b"].clone(),
            "fit": best_fit
        })

    def run(self):
        pop = self.init_population()
        best_fit = float("inf")
        best_ind = None
        pool = mp.Pool(processes=max(2, self.pop_size // 2))

        # Initialize history matrix and base vector
        self.init_history_and_Q(pop, pool)

        for g in range(self.G_max):
            H = self.sample_H(pop)  # Eq.(1)
            self.update_Q()  # Eq.(3)-(5)

            task_list = []
            for j in range(self.pop_size):
                m_j = H[j]
                z_j = pop[np.random.randint(0, self.pop_size)]
                u_j = pop[np.random.randint(0, self.pop_size)]
                n_float, w_raw, b_raw = self.alpha_update_full(m_j, z_j, u_j)  # Eq.(2)

                # Discrete neuron mapping
                n_corr = min(self.neuron_set, key=lambda x: abs(x - n_float))
                # Boundary shrinkage (Eq.6)
                w_corr = self.bound_map(w_raw)
                b_corr = self.bound_map(b_raw)

                sub_seed = self.base_seed + g * 100 + j
                task = (
                    n_corr, w_corr.cpu(), b_corr.cpu(),
                    self.full_X, self.full_Y, self.foldid_tensor,
                    sub_seed, self.lamb, self.max_n,
                    self.fixed_hidden_state
                )
                task_list.append(task)

            # Synchronous update frequency
            if (g + 1) % self.sync_freq == 0:
                eval_res = pool.map(five_fold_evaluate_individual, task_list)
            else:
                eval_res = pool.map(five_fold_infer_only, task_list)

            # Eq.(7) Greedy selection to update population
            new_pop = []
            iter_best_fit = float("inf")
            iter_best_ind = None
            for k in range(self.pop_size):
                fit_h, acc_h, nh = eval_res[k]
                old = pop[k]
                old_task = (
                    old["active_n"], old["w"].cpu(), old["b"].cpu(),
                    self.full_X, self.full_Y, self.foldid_tensor,
                    self.base_seed + 9999 + k, self.lamb, self.max_n,
                    self.fixed_hidden_state
                )
                fit_old, _, _ = five_fold_infer_only(old_task)

                if fit_h <= fit_old:
                    sel = {"active_n": nh, "w": w_corr.to(device), "b": b_corr.to(device)}
                    new_pop.append(sel)
                    if fit_h < iter_best_fit:
                        iter_best_fit = fit_h
                        iter_best_ind = sel
                else:
                    new_pop.append(old)
            pop = new_pop

            # Update history matrix
            self.update_B(iter_best_ind, iter_best_fit)

            if iter_best_fit < best_fit:
                best_fit = iter_best_fit
                best_ind = iter_best_ind

            print(f"[Full Alpha] Iter {g + 1}/{self.G_max}, current best fit = {best_fit:.4f}")

        pool.close()
        pool.join()
        return best_ind

def retrain_final_model(backbone, best_ind, full_X, full_Y, fixed_hidden_state):
    """
    Retrain final model on full Working Condition 1 dataset.
    Uses all Condition 1 samples for final training, not best-fold checkpoint.
    Includes learning rate decay and random scaling augmentation.
    """
    n_opt = best_ind["active_n"]
    w_opt = best_ind["w"]
    b_opt = best_ind["b"]

    mlp = BaseMLP(max_in_dim=256).to(device)
    mlp.in_layer.weight.data = w_opt.clone()
    mlp.in_layer.bias.data = b_opt.clone()

    # Load global fixed hidden layers
    mlp.hid1.weight.data = fixed_hidden_state['hid1.weight'].clone().to(device)
    mlp.hid1.bias.data = fixed_hidden_state['hid1.bias'].clone().to(device)
    mlp.hid2.weight.data = fixed_hidden_state['hid2.weight'].clone().to(device)
    mlp.hid2.bias.data = fixed_hidden_state['hid2.bias'].clone().to(device)
    mlp.hid3.weight.data = fixed_hidden_state['hid3.weight'].clone().to(device)
    mlp.hid3.bias.data = fixed_hidden_state['hid3.bias'].clone().to(device)
    mlp.out.weight.data = fixed_hidden_state['out.weight'].clone().to(device)
    mlp.out.bias.data = fixed_hidden_state['out.bias'].clone().to(device)

    for name, param in mlp.named_parameters():
        if "in_layer" not in name:
            param.requires_grad = False

    ds = TensorDataset(full_X, full_Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=lr, weight_decay=1e-4)
    # Learning rate decay strategy
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    mlp.train()
    for _ in range(train_epoch):
        for xf, yf in loader:
            xf, yf = xf.to(device), yf.to(device)
            # Random scaling data augmentation
            scale_factor = torch.FloatTensor(xf.size(0), 1).uniform_(0.9, 1.1).to(device)
            xf_aug = xf * scale_factor
            pred = mlp(xf_aug, active_n=n_opt)
            loss = loss_fn(pred, yf)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    return backbone, mlp

# ==================== Confusion Matrix & Testing ====================
def plot_confusion_matrix(y_true, y_pred, class_names, save_name, normalize=False):
    """Generate and save confusion matrix plot"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Row-Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Raw Confusion Matrix"

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    return cm

def test_cond2(backbone, mlp, best_n, dataset_tag, index_csv):
    """
    Evaluate model on Working Condition 2 test set.
    Saves per-sample predictions with sample_id for full traceability.
    Auto-computes all metrics and generates both raw and row-normalized confusion matrices.
    """
    cfg = DATA_CONFIG[dataset_tag]
    cond2_img_path = cfg["cond2_img_path"]
    class_names = ["C", "H", "O", "B", "I"]
    class_labels = [0, 1, 2, 3, 4]

    if not os.path.exists(cond2_img_path):
        raise FileNotFoundError(f"Condition 2 path not found: {cond2_img_path}")

    # Load dataset (sorted by sample ID to match partition CSV)
    ds = FoldBearingDataset(cond2_img_path, split="")
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=0)

    # Load sample ID list from partition CSV (matches dataset order after sorting)
    df_full = pd.read_csv(index_csv)
    cond2_df = df_full[df_full["fold"] == -1].reset_index(drop=True)
    sample_ids = cond2_df["sample_id"].values.astype(int)

    backbone.eval()
    mlp.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ti, fi, tfi, lab in dl:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            feat = backbone(ti, fi, tfi)
            out = mlp(feat, active_n=best_n)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(lab.cpu().numpy())

    # ========== Save per-sample raw predictions with sample ID ==========
    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": trues,
        "y_pred": preds
    })
    pred_save_path = f"{dataset_tag}_cond2_sample_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)
    print(f"✅ Per-sample predictions saved to: {pred_save_path}")

    # ========== Auto-compute all metrics ==========
    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average="macro")
    recall_per_class = recall_score(trues, preds, average=None, labels=class_labels)

    print(f"\n==== {dataset_tag} Condition 2 Test Results ====")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    print(f"Per-class Recall: {[round(r, 4) for r in recall_per_class]}")

    # ========== Auto-generate both confusion matrix plots ==========
    # 1. Raw count confusion matrix (corresponds to Fig. S8)
    plot_confusion_matrix(
        trues, preds, class_names,
        f"{dataset_tag}_cond2_raw_confusion.png",
        normalize=False
    )
    # 2. Row-normalized confusion matrix (corresponds to Fig. 11)
    plot_confusion_matrix(
        trues, preds, class_names,
        f"{dataset_tag}_cond2_row_norm.png",
        normalize=True
    )

    return acc, f1_macro, recall_per_class

def run_one_dataset(dataset_tag):
    cfg = DATA_CONFIG[dataset_tag]
    index_csv = cfg["index_csv"]
    image_root = cfg["image_root"]
    weight_dir = cfg["weight_save_dir"]
    os.makedirs(weight_dir, exist_ok=True)

    backbone = FeatureBackbone().to(device)
    backbone.eval()
    full_X, full_Y, foldid_tensor = extract_all_fixed_features(backbone, index_csv, image_root)
    print(f"[{dataset_tag}] Condition 1 total samples: {len(full_Y)}")

    ae = AlphaEvolutionFull(full_X, full_Y, foldid_tensor, base_seed=SEED)
    best_ind = ae.run()
    best_n = best_ind["active_n"]
    print(f"[{dataset_tag}] Evolution finished, optimal active neurons = {best_n}")

    backbone_final, mlp_final = retrain_final_model(
        backbone, best_ind, full_X, full_Y, ae.fixed_hidden_state
    )

    # ========== Save complete checkpoint with full configuration ==========
    save_dict = {
        "backbone": backbone_final.state_dict(),
        "mlp": mlp_final.state_dict(),
        "mlp_active_n": best_n,
        "mlp_max_in_dim": 256,
        "fixed_hidden_state": ae.fixed_hidden_state,
        "random_seed": SEED,
        "feature_dim": feature_dim,
        "num_classes": num_classes
    }
    save_path = os.path.join(weight_dir, "global_opt_final.pth")
    torch.save(save_dict, save_path)
    print(f"✅ Full checkpoint saved to: {save_path}")

    # Test on Condition 2 with traceable output
    acc_cond2, f1_cond2, recall_cond2 = test_cond2(
        backbone_final, mlp_final, best_n, dataset_tag, index_csv
    )
    return best_ind, acc_cond2, f1_cond2

if __name__ == "__main__":
    set_seed(SEED)
    print("==== Full Alpha Evolution (Eq.1-7 + sync frequency + global fixed hidden layers) ====")
    run_one_dataset("seu")
    # run_one_dataset("lab")  # Uncomment to run laboratory dataset
