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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ====================== Global random seed setting ======================
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

# ==================== Global hyper‑parameters (fully consistent with Table 2 in the paper) ====================
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

# Alpha evolution hyper‑parameters (optimal values in Table 8 of the paper)
N_pop = 50
G_max = 50
alpha_coeff = 0.8
theta_coeff = 0.3
lambda_fit = 0.01
neuron_candidates = [64, 128, 256]
w_lb, w_ub = -0.3, 0.3

# ==================== Dataset path configuration (matched with preprocessing outputs) ====================
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

# ==================== Network module: CNN‑Transformer feature fusion [Complete implementation in Section 2.2 of the paper] ====================
class SingleCNN(nn.Module):
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
    """Sine‑cosine positional encoding in Equation (10) of the paper, added element‑wise"""
    def __init__(self, dim: int, seq_len: int = 3):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class CrossAttention(nn.Module):
    """Time‑frequency features as Q; time‑domain and frequency‑domain features as KV; Equation (9) in the paper"""
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
    """Standard Transformer Encoder sub‑module: multi‑head self‑attention + FFN, residual connection + LN + Dropout"""
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
    """CNN‑Transformer serial fusion module, strictly matching the block diagram in Figure 3 of the paper"""
    def __init__(self, dim=128, heads=4, seq_len=3):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim=dim, seq_len=seq_len)
        self.cross_att = CrossAttention(dim)
        self.multi_head_att = MultiHeadSelfAttn(dim=dim, heads=heads,
                                                ffn_hidden=transformer_ffn_dim,
                                                dropout=transformer_dropout)

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
    """CNN‑Transformer backbone network: [Feature fusion is performed first, the backbone is frozen throughout,
    and only the input layer of the MLP is optimized subsequently]"""
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


class BaseMLP(nn.Module):
    """
    MLP: Only the input layer participates in Alpha evolution optimization;
    hidden layers and output layer are fixed and excluded from evolutionary search.
    Table 7 of the paper: parameters of hidden1‑3 and output layers are fixed;
    only the neuron number, weight W and bias b of the input layer are optimized.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_layer = nn.Linear(feature_dim, in_dim)
        self.hid1 = nn.Linear(in_dim, 128)
        self.hid2 = nn.Linear(128, 128)
        self.hid3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, feat):
        x = self.relu(self.in_layer(feat))
        x = self.relu(self.hid1(x))
        x = self.relu(self.hid2(x))
        x = self.relu(self.hid3(x))
        return self.out(x)


# ==================== Dataset loading class ====================
class FoldBearingDataset(Dataset):
    def __init__(self, fold_img_root, split="train"):
        self.root = os.path.join(fold_img_root, split)
        self.samples = []
        fault2label = {"C": 0, "H": 1, "O": 2, "B": 3, "I": 4}
        if not os.path.exists(self.root):
            print(f"[Warning] Folder does not exist: {self.root}")
            return
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
            print(f"[Warning] No samples loaded from path {self.root}!")

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


def extract_fixed_features(backbone: FeatureBackbone, data_loader):
    """
    ✅ Key timing point: Run the CNN‑Transformer backbone to extract **fixed fused features**,
    freeze backbone parameters, and send features to Alpha evolution for MLP optimization.
    Fully consistent with the logic in the paper: feature fusion first, then algorithm optimization.
    """
    backbone.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for ti, fi, tfi, lab in data_loader:
            ti, fi, tfi = ti.to(device), fi.to(device), tfi.to(device)
            feat = backbone(ti, fi, tfi)
            all_features.append(feat.cpu())
            all_labels.append(lab)
    X = torch.cat(all_features, dim=0)
    Y = torch.cat(all_labels, dim=0)
    return X, Y


def train_mlp_only(args):
    """
    Sub‑task inside Alpha evolution: CNN‑Transformer features have been extracted in advance,
    only the input layer of MLP is trained; parameters of hid1‑3 and output layer are fixed and not updated,
    consistent with Algorithm 1 in the paper.
    """
    n_neuron, w_cpu, b_cpu, X_train, Y_train, X_val, Y_val, dev, epoch_num, lamb, max_n, sub_seed = args
    set_seed(sub_seed)

    w = w_cpu.to(dev)
    b = b_cpu.to(dev)
    mlp = BaseMLP(n_neuron).to(dev)
    mlp.in_layer.weight.data = w.clone()
    mlp.in_layer.bias.data = b.clone()

    # Freeze MLP hidden layers and output layer, only update the input layer! [Core constraint of the paper]
    for name, param in mlp.named_parameters():
        if "in_layer" not in name:
            param.requires_grad = False

    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = optim.Adam(filter(lambda p: p.requires_grad, mlp.parameters()), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(epoch_num):
        mlp.train()
        for xf, yf in train_loader:
            xf, yf = xf.to(dev), yf.to(dev)
            pred = mlp(xf)
            loss = loss_fn(pred, yf)
            opt.zero_grad()
            loss.backward()
            opt.step()

    mlp.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xf, yf in val_loader:
            xf, yf = xf.to(dev), yf.to(dev)
            out = mlp(xf)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(yf.cpu().numpy())
    acc = accuracy_score(trues, preds)
    fit = 1 - acc + lamb * (n_neuron / max_n)
    return fit, acc, n_neuron, mlp.in_layer.weight.data.cpu(), mlp.in_layer.bias.data.cpu()


class AlphaEvolutionSerial:
    def __init__(self, X_train, Y_train, X_val, Y_val, base_seed):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.pop_size = N_pop
        self.G_max = G_max
        self.alpha = alpha_coeff
        self.theta = theta_coeff
        self.lamb = lambda_fit
        self.neuron_set = neuron_candidates
        self.w_lb = w_lb
        self.w_ub = w_ub
        self.max_neuron = max(self.neuron_set)
        self.base_seed = base_seed

    def bound_map(self, val):
        if val > self.w_ub:
            return (val + self.w_ub) / 2
        elif val < self.w_lb:
            return (val + self.w_lb) / 2
        return val

    def init_population(self):
        pop = []
        for idx in range(self.pop_size):
            np.random.seed(self.base_seed + idx)
            n = np.random.choice(self.neuron_set)
            w_rand = torch.FloatTensor(n, feature_dim).uniform_(self.w_lb, self.w_ub)
            b_rand = torch.FloatTensor(n).uniform_(self.w_lb, self.w_ub)
            pop.append({"n": n, "w": w_rand, "b": b_rand})
        return pop

    def sample_H(self, pop):
        idx = np.random.randint(0, self.pop_size, self.pop_size)
        H = [pop[i] for i in idx]
        return H

    def alpha_update(self, m_ind, z_ind, u_ind):
        n_m, w_m, b_m = m_ind["n"], m_ind["w"], m_ind["b"]
        n_z, w_z, b_z = z_ind["n"], z_ind["w"], z_ind["b"]
        n_u, w_u, b_u = u_ind["n"], u_ind["w"], u_ind["b"]
        dr = np.random.randn()
        new_w = w_m + self.alpha * dr + self.theta * (w_z + w_m - w_u)
        new_b = b_m + self.alpha * dr + self.theta * (b_z + b_m - b_u)
        new_n_float = n_m + self.alpha * dr
        return new_n_float, new_w, new_b

    def run_serial_optimize(self):
        pop = self.init_population()
        best_fit = float("inf")
        best_ind = None
        pool = mp.Pool(processes=max(2, self.pop_size // 2))

        for g in range(self.G_max):
            H = self.sample_H(pop)
            task_args_list = []
            for j in range(self.pop_size):
                m_j = H[j]
                z_j = pop[np.random.randint(0, self.pop_size)]
                u_j = pop[np.random.randint(0, self.pop_size)]
                n_float, w_raw, b_raw = self.alpha_update(m_j, z_j, u_j)
                n_corr = min(self.neuron_set, key=lambda x: abs(x - n_float))
                w_corr = torch.zeros_like(w_raw)
                b_corr = torch.zeros_like(b_raw)
                for i in range(w_corr.shape[0]):
                    for jd in range(w_corr.shape[1]):
                        w_corr[i, jd] = self.bound_map(w_raw[i, jd])
                for i in range(b_corr.shape[0]):
                    b_corr[i] = self.bound_map(b_raw[i])

                sub_process_seed = self.base_seed + g * 1000 + j
                task = (
                    n_corr, w_corr.cpu(), b_corr.cpu(),
                    self.X_train, self.Y_train,
                    self.X_val, self.Y_val,
                    device, train_epoch, self.lamb, self.max_neuron,
                    sub_process_seed
                )
                task_args_list.append(task)

            eval_results = pool.map(train_mlp_only, task_args_list)
            new_pop = []
            for k in range(self.pop_size):
                fit_h, acc_h, nh, wh, bh = eval_results[k]
                old = pop[k]
                old_fit, _, _, _, _ = train_mlp_only((
                    old["n"], old["w"].cpu(), old["b"].cpu(),
                    self.X_train, self.Y_train,
                    self.X_val, self.Y_val,
                    device, train_epoch, self.lamb, self.max_neuron,
                    self.base_seed + 9999 + k
                ))
                if fit_h <= old_fit:
                    select = {"n": nh, "w": wh.to(device), "b": bh.to(device)}
                    new_pop.append(select)
                    if fit_h < best_fit:
                        best_fit = fit_h
                        best_ind = select
                else:
                    new_pop.append(pop[k])
            pop = new_pop
            print(f"Evolution iteration {g+1}/{self.G_max} | Global optimal fitness = {best_fit:.4f}")
        pool.close()
        pool.join()
        return best_ind


def eval_full_model(backbone, mlp, loader):
    backbone.eval()
    mlp.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ti, fi, tfi, lab in loader:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            fuse_feat = backbone(ti, fi, tfi)
            out = mlp(fuse_feat)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(lab.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1_macro = f1_score(trues, preds, average="macro")
    return acc, f1_macro


# ====================== Confusion‑matrix plotting function ======================
def plot_confusion_matrix(y_true, y_pred, class_names, save_name, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Row‑Normalized Confusion Matrix (Diagonal = Recall per Class)"
        # Print recall of each class
        recall_per_class = np.diag(cm)
        print("\n------Recall of each class------")
        for cls, rec in zip(class_names, recall_per_class):
            print(f"{cls}: {rec:.4f}")
    else:
        fmt = "d"
        title = "Raw Confusion Matrix (Sample Count)"

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


# ==================== [One‑shot generalization test for working condition 2] ====================
def test_cond2_once(best_model_info, dataset_tag: str):
    cfg = DATA_CONFIG[dataset_tag]
    cond2_img_path = cfg["cond2_img_path"]

    if not os.path.exists(cond2_img_path):
        print(f"Error: Working‑condition‑2 image folder {cond2_img_path} does not exist!")
        return

    cond2_dataset = FoldBearingDataset(cond2_img_path, split="")
    cond2_loader = DataLoader(cond2_dataset, batch_size, shuffle=False, num_workers=0)
    print(f"\n===== [{dataset_tag}] One‑shot cross‑condition generalization test for working condition‑2 =====")
    print(f"Total number of test samples under working condition‑2: {len(cond2_dataset)}")

    weight_path = best_model_info["path"]
    checkpoint = torch.load(weight_path, map_location=device)
    backbone = FeatureBackbone().to(device)
    backbone.load_state_dict(checkpoint["backbone"])
    mlp = BaseMLP(checkpoint["mlp_in_dim"]).to(device)
    mlp.load_state_dict(checkpoint["mlp"])

    backbone.eval()
    mlp.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ti, fi, tfi, lab in cond2_loader:
            ti, fi, tfi, lab = ti.to(device), fi.to(device), tfi.to(device), lab.to(device)
            fuse_feat = backbone(ti, fi, tfi)
            out = mlp(fuse_feat)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(lab.cpu().numpy())

    class_names = ["C", "H", "O", "B", "I"]

    if dataset_tag == "lab":
        raw_cm = plot_confusion_matrix(trues, preds, class_names,
                                        save_name="lab_cond2_raw_confusion.png",
                                        normalize=False)
        norm_cm = plot_confusion_matrix(trues, preds, class_names,
                                        save_name="lab_cond2_row_norm_confusion.png",
                                        normalize=True)
        print("\n==== Laboratory dataset ‑ Raw confusion matrix under working condition 2 (sample count) ====")
        print(raw_cm)
        print("\n==== Laboratory dataset ‑ Row‑normalized confusion matrix under working condition 2 ====")
        print(norm_cm)
        return
    else:
        acc = accuracy_score(trues, preds)
        f1_macro = f1_score(trues, preds, average="macro")
        print(f"Cross‑condition test results for working condition‑2  Acc = {acc:.4f}, Macro‑F1 = {f1_macro:.4f}")
        return acc, f1_macro


# ==================== Working condition 1: 5‑fold cross‑validation training ====================
def main_kfold_serial_train(dataset_tag: str):
    cfg = DATA_CONFIG[dataset_tag]
    index_csv = cfg["index_csv"]
    image_root = cfg["image_root"]
    weight_save_dir = cfg["weight_save_dir"]
    os.makedirs(weight_save_dir, exist_ok=True)

    if not os.path.exists(index_csv):
        print(f"Error: Dataset index file {index_csv} does not exist! dataset={dataset_tag}")
        return None

    df_all = pd.read_csv(index_csv)
    cond1_df = df_all[df_all["is_train"] == True].reset_index(drop=True)
    fold_record = []

    for fold_idx in range(n_fold):
        fold_seed = SEED + fold_idx * 100
        set_seed(fold_seed)

        print(f"\n========== [{dataset_tag} dataset] Fold {fold_idx+1} ==========")
        train_df = cond1_df[cond1_df["fold"] != (fold_idx+1)]
        val_df = cond1_df[cond1_df["fold"] == (fold_idx+1)]

        fold_img_root = os.path.join(image_root, f"fold{fold_idx+1}_img")
        train_set = FoldBearingDataset(fold_img_root, "train")
        val_set = FoldBearingDataset(fold_img_root, "val")

        print(f"Fold {fold_idx+1}, training samples:{len(train_set)}, validation samples:{len(val_set)}")

        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=0)

        backbone = FeatureBackbone().to(device)
        # =========Core workflow: [CNN‑Transformer performs feature fusion first to extract fixed features,
        # then Alpha evolution optimizes the MLP]=========
        X_train, Y_train = extract_fixed_features(backbone, train_loader)
        X_val, Y_val = extract_fixed_features(backbone, val_loader)

        ae_serial = AlphaEvolutionSerial(X_train, Y_train, X_val, Y_val, base_seed=fold_seed)
        best_ind = ae_serial.run_serial_optimize()
        best_n, best_w, best_b = best_ind["n"], best_ind["w"], best_ind["b"]
        best_mlp = BaseMLP(best_n).to(device)
        best_mlp.in_layer.weight.data = best_w.clone()
        best_mlp.in_layer.bias.data = best_b.clone()

        save_dict = {
            "backbone": backbone.state_dict(),
            "mlp": best_mlp.state_dict(),
            "mlp_in_dim": best_n,
            "seed_used": fold_seed
        }
        save_path = os.path.join(weight_save_dir, f"fold{fold_idx+1}_serial_best.pth")
        torch.save(save_dict, save_path)

        best_acc, best_f1 = eval_full_model(backbone, best_mlp, val_loader)
        fold_record.append({"dataset": dataset_tag, "fold": fold_idx + 1, "acc": best_acc, "f1": best_f1, "path":save_path})
        print(f"[{dataset_tag}] Validation‑set results of fold {fold_idx+1} | Acc={best_acc:.4f}, F1={best_f1:.4f}")

    acc_list = [x["acc"] for x in fold_record]
    f1_list = [x["f1"] for x in fold_record]
    print(f"\n===== [{dataset_tag}] Summary of 5‑fold cross‑validation under working condition 1 =====")
    print(f"Mean accuracy {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Mean macro‑F1 {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    best_fold_item = max(fold_record, key=lambda x:x["acc"])
    print(f"\nOptimal model selected according to validation‑set performance: Fold {best_fold_item['fold']}, validation Acc={best_fold_item['acc']:.4f}")
    return best_fold_item


# ==================== Program entry [Timing sequence: run working‑condition‑2 test after all working‑condition‑1 training is finished] ====================
if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 80)
    print("========== Stage 1: 5‑fold training on all datasets under working condition 1 (CNN‑Transformer feature fusion first, then Alpha evolution optimizes MLP) ==========")
    print("=" * 80)

    best_seu_model = main_kfold_serial_train("seu")
    best_lab_model = main_kfold_serial_train("lab")

    print("\n" + "="*80)
    print("========== Stage 2: All training tasks under working condition‑1 completed, start cross‑condition generalization test for working condition‑2 ==========")
    print("="*80)

    acc_seu, f1_seu = test_cond2_once(best_seu_model, "seu")
    test_cond2_once(best_lab_model, "lab")
