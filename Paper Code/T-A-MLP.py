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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)
mp.set_start_method('spawn', force=True)

# ================ Global Hyperparameters ================
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
max_neuron_global = 256
sync_freq = 5
K_history = 5
d_b = 0.6

# ==================== Dataset Path Configuration ====================
DATA_CONFIG = {
    "seu": {
        "index_csv": "./southeast_dataset_partition.csv",
        "image_root": "./se_fold_images",
        "cond1_full_img_path": "./se_cond1_full_images",
        "cond2_img_path": "./se_cond2_images_fixed",
        "weight_save_dir": "./weights/seu",
        "num_channels": 8
    },
    "lab": {
        "index_csv": "./lab_dataset_partition.csv",
        "image_root": "./lab_fold_images",
        "cond1_full_img_path": "./lab_cond1_full_images",
        "cond2_img_path": "./lab_cond2_images_fixed",
        "weight_save_dir": "./weights/lab",
        "num_channels": 4
    }
}

# ==================== Network Modules ====================
class SingleCNN(nn.Module):
    """可配置输入通道数的2D CNN特征提取器"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 1, 1)
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
    def __init__(self, dim: int, seq_len: int = 3):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CrossAttention(nn.Module):
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
        return torch.matmul(attn_weight, V)

class MultiHeadSelfAttn(nn.Module):
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
            nn.Linear(dim, ffn_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim), nn.Dropout(dropout)
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
        return torch.mean(x_att, dim=1)

# ===================== 特征骨干网络（双CNN架构） =====================
class FeatureBackbone(nn.Module):
    """
    时域/频域共享多通道CNN，时频域单独单通道CNN
    三域输出特征维度统一为128，送入融合模块
    """
    def __init__(self, in_channels=8):
        super().__init__()
        # 时域、频域共享的多通道CNN
        self.shared_multi_cnn = SingleCNN(in_channels=in_channels)
        # 时频域单独的单通道CNN
        self.single_tf_cnn = SingleCNN(in_channels=1)
        # 特征融合模块
        self.fusion = SerialAttnFusion(feature_dim, n_head)

    def forward(self, ti, fi, tfi):
        ft = self.shared_multi_cnn(ti)   # 时域多通道特征
        ff = self.shared_multi_cnn(fi)   # 频域多通道特征（共享权重）
        ftf = self.single_tf_cnn(tfi) # 时频单通道特征
        return self.fusion(ft, ff, ftf)

# ========== BaseMLP ==========
class BaseMLP(nn.Module):
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
        mask = torch.zeros_like(x)
        mask[..., :active_n] = 1.0
        x = x * mask
        x = self.relu(self.hid1(x))
        x = self.relu(self.hid2(x))
        x = self.relu(self.hid3(x))
        return self.out(x)

# ==================== Dataset & Feature Extraction ====================
class FoldBearingDataset(Dataset):
    def __init__(self, fold_img_root, num_channels=8, split="train"):
        self.root = os.path.join(fold_img_root, split) if split else fold_img_root
        self.num_channels = num_channels
        self.samples = []
        fault2label = {"C": 0, "H": 1, "O": 2, "B": 3, "I": 4}

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        for fault in os.listdir(self.root):
            fd = os.path.join(self.root, fault)
            if not os.path.isdir(fd):
                continue
            for fname in os.listdir(fd):
                if "_time_ch0.png" in fname:
                    base = fname.replace("_time_ch0.png", "")
                    self.samples.append({
                        "base": os.path.join(fd, base),
                        "label": fault2label[fault]
                    })

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {self.root}")
        self.samples.sort(key=lambda x: int(os.path.basename(x["base"]).split("_")[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        base_path = item["base"]

        # 时域：加载全部通道
        t_list = []
        for ch in range(self.num_channels):
            img = cv2.imread(f"{base_path}_time_ch{ch}.png", 0)
            t_list.append(img.astype(np.float32) / 255.0)
        t = np.stack(t_list, axis=0)  # [C, H, W]

        # 频域：加载全部通道
        f_list = []
        for ch in range(self.num_channels):
            img = cv2.imread(f"{base_path}_freq_ch{ch}.png", 0)
            f_list.append(img.astype(np.float32) / 255.0)
        f = np.stack(f_list, axis=0)  # [C, H, W]

        # 时频：仅加载单张0通道图
        tf_img = cv2.imread(f"{base_path}_tf.png", 0)
        tf = (tf_img.astype(np.float32) / 255.0)[np.newaxis, ...]  # [1, H, W]

        t = torch.from_numpy(t).float()
        f = torch.from_numpy(f).float()
        tf = torch.from_numpy(tf).float()
        return t, f, tf, item["label"]

# ==================== 每折独立预训练函数 ====================
def pre_train_single_fold(fold_idx, index_csv, image_root, base_seed, num_channels=8):
    fold_img_path = os.path.join(image_root, f"fold{fold_idx + 1}_img")
    train_ds = FoldBearingDataset(fold_img_path, num_channels=num_channels, split="train")
    val_ds = FoldBearingDataset(fold_img_path, num_channels=num_channels, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    set_seed(base_seed + fold_idx * 1000)
    backbone = FeatureBackbone(in_channels=num_channels).to(device)
    mlp = BaseMLP(max_in_dim=256).to(device)

    opt = optim.Adam([
        {"params": backbone.parameters()},
        {"params": mlp.parameters()}
    ], lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    backbone.train()
    mlp.train()
    for epoch in range(train_epoch):
        for ti, fi, tfi, lab in train_loader:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            scale = torch.FloatTensor(ti.size(0), 1, 1, 1).uniform_(0.9, 1.1).to(device)
            ti_aug = ti * scale
            fi_aug = fi * scale
            tfi_aug = tfi * scale
            feat = backbone(ti_aug, fi_aug, tfi_aug)
            pred = mlp(feat, active_n=256)
            loss = loss_fn(pred, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    backbone.eval()
    mlp.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    for name, param in mlp.named_parameters():
        if "in_layer" not in name:
            param.requires_grad = False

    def extract_features(loader):
        feats, labels = [], []
        with torch.no_grad():
            for ti, fi, tfi, lab in loader:
                ti, fi, tfi = ti.to(device), fi.to(device), tfi.to(device)
                feat = backbone(ti, fi, tfi)
                feats.append(feat.cpu())
                labels.append(lab)
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_feat, train_lab = extract_features(train_loader)
    val_feat, val_lab = extract_features(val_loader)

    frozen_backbone = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
    frozen_mlp_layers = {k: v.cpu().clone() for k, v in mlp.state_dict().items() if "in_layer" not in k}
    return {
        "frozen_backbone": frozen_backbone,
        "frozen_mlp": frozen_mlp_layers,
        "train_feat": train_feat,
        "train_lab": train_lab,
        "val_feat": val_feat,
        "val_lab": val_lab
    }

# ===================== 五折评估函数 =====================
def five_fold_evaluate_individual(args):
    n_neuron, w_cpu, b_cpu, fold_data_list, sub_seed, lamb, max_n, num_ch = args
    set_seed(sub_seed)
    acc_list = []
    for fold_idx in range(n_fold):
        fd = fold_data_list[fold_idx]
        X_tr, Y_tr = fd["train_feat"], fd["train_lab"]
        X_vl, Y_vl = fd["val_feat"], fd["val_lab"]
        frozen_mlp = fd["frozen_mlp"]

        mlp = BaseMLP(max_in_dim=256).to(device)
        mlp.load_state_dict(frozen_mlp, strict=False)
        mlp.in_layer.weight.data = w_cpu.clone().to(device)
        mlp.in_layer.bias.data = b_cpu.clone().to(device)
        for name, param in mlp.named_parameters():
            if "in_layer" not in name:
                param.requires_grad = False

        tr_ds = TensorDataset(X_tr, Y_tr)
        vl_ds = TensorDataset(X_vl, Y_vl)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        opt = optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        mlp.train()
        for _ in range(train_epoch):
            for xf, yf in tr_loader:
                xf, yf = xf.to(device), yf.to(device)
                scale_factor = torch.FloatTensor(xf.size(0), 1).uniform_(0.9, 1.1).to(device)
                xf_aug = xf * scale_factor
                pred = mlp(xf_aug, active_n=n_neuron)
                loss = loss_fn(pred, yf)
                opt.zero_grad()
                loss.backward()
                opt.step()
            scheduler.step()

        mlp.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xf, yf in vl_loader:
                xf, yf = xf.to(device), yf.to(device)
                out = mlp(xf, active_n=n_neuron)
                pred = torch.argmax(out, dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(yf.cpu().numpy())
        acc_list.append(accuracy_score(trues, preds))

    acc5 = float(np.mean(acc_list))
    fit = 1.0 - acc5 + lamb * (n_neuron / max_n)
    return fit, acc5, n_neuron

def five_fold_infer_only(args):
    n_neuron, w_cpu, b_cpu, fold_data_list, sub_seed, lamb, max_n, num_ch = args
    set_seed(sub_seed)
    acc_list = []
    for fold_idx in range(n_fold):
        fd = fold_data_list[fold_idx]
        X_vl, Y_vl = fd["val_feat"], fd["val_lab"]
        frozen_mlp = fd["frozen_mlp"]

        mlp = BaseMLP(max_in_dim=256).to(device)
        mlp.load_state_dict(frozen_mlp, strict=False)
        mlp.in_layer.weight.data = w_cpu.clone().to(device)
        mlp.in_layer.bias.data = b_cpu.clone().to(device)

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
        acc_list.append(accuracy_score(trues, preds))

    acc5 = float(np.mean(acc_list))
    fit = 1.0 - acc5 + lamb * (n_neuron / max_n)
    return fit, acc5, n_neuron

# ===================== Alpha进化类 =====================
class AlphaEvolutionFull:
    def __init__(self, index_csv, image_root, base_seed, num_channels=8):
        self.index_csv = index_csv
        self.image_root = image_root
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
        self.num_ch = num_channels

        print("Pre-training backbone and fixed layers for each fold...")
        self.fold_data_list = []
        for fold_idx in range(n_fold):
            print(f"  Fold {fold_idx + 1}/{n_fold} pre-training...")
            fold_info = pre_train_single_fold(fold_idx, index_csv, image_root, base_seed, num_channels)
            self.fold_data_list.append(fold_info)
        print("All folds pre-trained and frozen successfully.\n")

    def bound_map(self, val_tensor):
        upper = torch.ones_like(val_tensor) * self.w_ub
        lower = torch.ones_like(val_tensor) * self.w_lb
        out = torch.where(val_tensor > self.w_ub, (val_tensor + upper) / 2, val_tensor)
        out = torch.where(out < self.w_lb, (out + lower) / 2, out)
        return out

    def init_population(self):
        pop = []
        for idx in range(self.pop_size):
            np.random.seed(self.base_seed + idx)
            n = int(np.random.choice(self.neuron_set))
            w_rand = torch.FloatTensor(256, feature_dim).uniform_(self.w_lb, self.w_ub)
            b_rand = torch.FloatTensor(256).uniform_(self.w_lb, self.w_ub)
            pop.append({"active_n": n, "w": w_rand, "b": b_rand})
        return pop

    def init_history_and_Q(self, pop, pool):
        task_list = []
        for idx, ind in enumerate(pop):
            task = (
                ind["active_n"], ind["w"].cpu(), ind["b"].cpu(),
                self.fold_data_list,
                self.base_seed + 10000 + idx, self.lamb, self.max_n, self.num_ch
            )
            task_list.append(task)
        init_fits = pool.map(five_fold_infer_only, task_list)
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
        self.Q_w = torch.stack([b["w"] for b in self.B], dim=0).mean(dim=0)
        self.Q_b = torch.stack([b["b"] for b in self.B], dim=0).mean(dim=0)
        self.Q_n = np.mean([b["active_n"] for b in self.B])

    def sample_H(self, pop):
        idx = np.random.randint(0, self.pop_size, self.pop_size)
        return [pop[i] for i in idx]

    def calculate_weights(self):
        fits = np.array([b["fit"] for b in self.B])
        inv_fits = 1.0 / (fits + 1e-8)
        return inv_fits / inv_fits.sum()

    def update_Q(self):
        omega = self.calculate_weights()
        if np.random.rand() < 0.5:
            diag_A_w = torch.ones_like(self.Q_w)
            diag_A_b = torch.ones_like(self.Q_b)
            diag_A_n = 128.0
            self.Q_w = self.d_b * self.Q_w + (1 - self.d_b) * diag_A_w
            self.Q_b = self.d_b * self.Q_b + (1 - self.d_b) * diag_A_b
            self.Q_n = self.d_b * self.Q_n + (1 - self.d_b) * diag_A_n
        else:
            w_list = [b["w"] * omega[i] for i, b in enumerate(self.B)]
            b_list = [b["b"] * omega[i] for i, b in enumerate(self.B)]
            n_list = [b["active_n"] * omega[i] for i, b in enumerate(self.B)]
            self.Q_w = torch.stack(w_list, dim=0).sum(dim=0)
            self.Q_b = torch.stack(b_list, dim=0).sum(dim=0)
            self.Q_n = np.sum(n_list)

    def alpha_update_full(self, m_ind, z_ind, u_ind):
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
        self.init_history_and_Q(pop, pool)

        for g in range(self.G_max):
            H = self.sample_H(pop)
            self.update_Q()
            task_list = []
            for j in range(self.pop_size):
                m_j = H[j]
                z_j = pop[np.random.randint(0, self.pop_size)]
                u_j = pop[np.random.randint(0, self.pop_size)]
                n_float, w_raw, b_raw = self.alpha_update_full(m_j, z_j, u_j)
                n_corr = min(self.neuron_set, key=lambda x: abs(x - n_float))
                w_corr = self.bound_map(w_raw)
                b_corr = self.bound_map(b_raw)
                sub_seed = self.base_seed + g * 100 + j
                task = (n_corr, w_corr.cpu(), b_corr.cpu(), self.fold_data_list, sub_seed, self.lamb, self.max_n, self.num_ch)
                task_list.append(task)

            if (g + 1) % self.sync_freq == 0:
                eval_res = pool.map(five_fold_evaluate_individual, task_list)
            else:
                eval_res = pool.map(five_fold_infer_only, task_list)

            new_pop = []
            iter_best_fit = float("inf")
            iter_best_ind = None
            for k in range(self.pop_size):
                fit_h, acc_h, nh = eval_res[k]
                old = pop[k]
                old_task = (old["active_n"], old["w"].cpu(), old["b"].cpu(), self.fold_data_list, self.base_seed + 9999 + k, self.lamb, self.max_n, self.num_ch)
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
            self.update_B(iter_best_ind, iter_best_fit)

            if iter_best_fit < best_fit:
                best_fit = iter_best_fit
                best_ind = iter_best_ind
            print(f"[Full Alpha] Iter {g + 1}/{self.G_max}, current best fit = {best_fit:.4f}")

        pool.close()
        pool.join()
        return best_ind

# ===================== 最终模型全量预训练 =====================
def retrain_final_model(best_ind, dataset_tag, base_seed):
    cfg = DATA_CONFIG[dataset_tag]
    full_img_path = cfg["cond1_full_img_path"]
    n_opt = best_ind["active_n"]
    w_opt = best_ind["w"]
    b_opt = best_ind["b"]
    num_ch = cfg["num_channels"]

    print("\nPre-training final full backbone on Condition 1 dataset...")
    set_seed(base_seed)
    full_ds = FoldBearingDataset(full_img_path, num_channels=num_ch, split="")
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    backbone = FeatureBackbone(in_channels=num_ch).to(device)
    mlp = BaseMLP(max_in_dim=256).to(device)

    opt = optim.Adam([
        {"params": backbone.parameters()},
        {"params": mlp.parameters()}
    ], lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    backbone.train()
    mlp.train()
    for epoch in range(train_epoch):
        for ti, fi, tfi, lab in full_loader:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            scale = torch.FloatTensor(ti.size(0), 1, 1, 1).uniform_(0.9, 1.1).to(device)
            ti_aug = ti * scale
            fi_aug = fi * scale
            tfi_aug = tfi * scale
            feat = backbone(ti_aug, fi_aug, tfi_aug)
            pred = mlp(feat, active_n=256)
            loss = loss_fn(pred, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    for name, param in mlp.named_parameters():
        if "in_layer" not in name:
            param.requires_grad = False

    mlp.in_layer.weight.data = w_opt.clone().to(device)
    mlp.in_layer.bias.data = b_opt.clone().to(device)

    opt_final = optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=lr, weight_decay=1e-4)
    scheduler_final = optim.lr_scheduler.StepLR(opt_final, step_size=10, gamma=0.5)

    mlp.train()
    for epoch in range(train_epoch):
        for ti, fi, tfi, lab in full_loader:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            scale = torch.FloatTensor(ti.size(0), 1, 1, 1).uniform_(0.9, 1.1).to(device)
            ti_aug = ti * scale
            fi_aug = fi * scale
            tfi_aug = tfi * scale
            feat = backbone(ti_aug, fi_aug, tfi_aug)
            pred = mlp(feat, active_n=n_opt)
            loss = loss_fn(pred, lab)
            opt_final.zero_grad()
            loss.backward()
            opt_final.step()
        scheduler_final.step()

    print("Final model training completed.")
    return backbone, mlp

# ==================== 测试与混淆矩阵 ====================
def plot_confusion_matrix(y_true, y_pred, class_names, save_name, normalize=False):
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
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    return cm

def test_cond2(backbone, mlp, best_n, dataset_tag, index_csv):
    cfg = DATA_CONFIG[dataset_tag]
    cond2_img_path = cfg["cond2_img_path"]
    num_ch = cfg["num_channels"]
    class_names = ["C", "H", "O", "B", "I"]
    class_labels = [0, 1, 2, 3, 4]

    if not os.path.exists(cond2_img_path):
        raise FileNotFoundError(f"Condition 2 path not found: {cond2_img_path}")

    ds = FoldBearingDataset(cond2_img_path, num_channels=num_ch, split="")
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=0)
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

    pred_df = pd.DataFrame({"sample_id": sample_ids, "y_true": trues, "y_pred": preds})
    pred_save_path = f"{dataset_tag}_cond2_sample_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)
    print(f"✅ Per-sample predictions saved to: {pred_save_path}")

    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average="macro")
    recall_per_class = recall_score(trues, preds, average=None, labels=class_labels)
    print(f"\n==== {dataset_tag} Condition 2 Test Results ====")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    print(f"Per-class Recall: {[round(r, 4) for r in recall_per_class]}")

    plot_confusion_matrix(trues, preds, class_names, f"{dataset_tag}_cond2_raw_confusion.png", normalize=False)
    plot_confusion_matrix(trues, preds, class_names, f"{dataset_tag}_cond2_row_norm.png", normalize=True)
    return acc, f1_macro, recall_per_class

def run_one_dataset(dataset_tag):
    cfg = DATA_CONFIG[dataset_tag]
    index_csv = cfg["index_csv"]
    image_root = cfg["image_root"]
    weight_dir = cfg["weight_save_dir"]
    num_ch = cfg["num_channels"]
    os.makedirs(weight_dir, exist_ok=True)

    ae = AlphaEvolutionFull(index_csv, image_root, base_seed=SEED, num_channels=num_ch)
    best_ind = ae.run()
    best_n = best_ind["active_n"]
    print(f"\n[{dataset_tag}] Evolution finished, optimal active neurons = {best_n}")

    backbone_final, mlp_final = retrain_final_model(best_ind, dataset_tag, base_seed=SEED)
    save_dict = {
        "backbone": backbone_final.state_dict(),
        "mlp": mlp_final.state_dict(),
        "mlp_active_n": best_n,
        "mlp_max_in_dim": 256,
        "random_seed": SEED,
        "feature_dim": feature_dim,
        "num_classes": num_classes
    }
    save_path = os.path.join(weight_dir, "global_opt_final.pth")
    torch.save(save_dict, save_path)
    print(f"✅ Full checkpoint saved to: {save_path}")

    acc_cond2, f1_cond2, recall_cond2 = test_cond2(backbone_final, mlp_final, best_n, dataset_tag, index_csv)
    return best_ind, acc_cond2, f1_cond2

if __name__ == "__main__":
    set_seed(SEED)
    print("==== Full Alpha Evolution (multi-channel time/freq, single-channel tf) ====")
    run_one_dataset("seu")
    # run_one_dataset("lab")
