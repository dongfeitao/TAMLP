import argparse
import csv
import gc
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sy import (CONFIGS, BASE_FIELDS, ROLE_FIELDS, gap_decimal, gap_tag,
                make_samples, partition_name, split_fold, write_csv, load_dataset_configs)

FEATURE_DIM = 128
NUM_CLASSES = 5
MAX_NEURONS = 256
NEURON_CHOICES = (64, 128, 256)
CLASS_FOLDERS = ("C", "H", "O", "B", "I")
DATASET_NAMES = {"seu": "southeast", "lab": "laboratory"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_local_checkpoint(path):
    # Only load checkpoints created by this script; never untrusted pickle files.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only
        return torch.load(path, map_location="cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------- Original architecture ----------------
class SingleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128 * 32 * 32, FEATURE_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        return self.relu(self.fc(self.flat(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, seq_len=3):
        super().__init__()
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wq, self.wk, self.wv = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.d_k = dim

    def forward(self, q_in, kv_in):
        q, k, v = self.wq(q_in), self.wk(kv_in), self.wv(kv_in)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k), dim=-1)
        return torch.matmul(weights, v)


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, dim, heads, ffn_hidden=512, dropout=0.3):
        super().__init__()
        self.heads, self.d_k = heads, dim // heads
        self.wqkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch, length, dim = x.shape
        residual = x
        qkv = self.wqkv(x).reshape(batch, length, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k), dim=-1)
        attention = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(batch, length, dim)
        x = self.norm1(residual + self.drop1(self.out_proj(attention)))
        return self.norm2(x + self.drop2(self.ffn(x)))


class SerialAttnFusion(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim)
        self.cross_att = CrossAttention(dim)
        self.multi_head_att = MultiHeadSelfAttn(dim, heads)

    def forward(self, time_feat, freq_feat, tf_feat):
        x = self.pos_enc(torch.stack([time_feat, freq_feat, tf_feat], dim=1))
        cross = self.cross_att(x[:, 2:3], x[:, :2])
        return self.multi_head_att(torch.cat([x[:, :2], cross], dim=1)).mean(dim=1)


class FeatureBackbone(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.shared_multi_cnn = SingleCNN(in_channels)
        self.single_tf_cnn = SingleCNN(1)
        self.fusion = SerialAttnFusion()

    def forward(self, time, freq, tf):
        return self.fusion(self.shared_multi_cnn(time), self.shared_multi_cnn(freq), self.single_tf_cnn(tf))


class BaseMLP(nn.Module):
    def __init__(self, max_in_dim=MAX_NEURONS):
        super().__init__()
        self.in_layer = nn.Linear(FEATURE_DIM, max_in_dim)
        self.hid1 = nn.Linear(max_in_dim, 128)
        self.hid2, self.hid3 = nn.Linear(128, 128), nn.Linear(128, 128)
        self.out = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, feat, active_n):
        x = self.relu(self.in_layer(feat))
        mask = torch.zeros_like(x)
        mask[..., :active_n] = 1.0
        x = x * mask
        return self.out(self.relu(self.hid3(self.relu(self.hid2(self.relu(self.hid1(x)))))))


# ---------------- Dataset identity and image checks ----------------
def read_checked_csv(path, expected, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"CSV schema mismatch: {path}")
        records = list(reader)
    if len(records) != len(expected):
        raise ValueError(f"CSV sample count mismatch: {path}")
    for record, target in zip(records, expected):
        for key in fields:
            value = record[key]
            if key == "gap_seconds":
                valid = gap_decimal(value) == gap_decimal(target[key])
            else:
                valid = value == str(target[key])
            if not valid:
                raise ValueError(f"CSV mismatch in {path}: sample {target['sample_id']}, field {key}")
    return expected


class FoldBearingDataset(Dataset):
    """Use explicit sample IDs and fail on missing OR extra PNGs."""
    def __init__(self, root, rows, num_channels):
        self.root = Path(root)
        self.rows = sorted(rows, key=lambda row: row["sample_id"])
        self.num_channels = num_channels
        if not self.rows or len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError(f"Empty or duplicate sample IDs: {root}")
        expected_files = set()
        for row in self.rows:
            base = self.root / CLASS_FOLDERS[int(row["fault_label"][1:])] / f"sample_{row['sample_id']:06d}"
            for ch in range(num_channels):
                expected_files.add(Path(f"{base}_time_ch{ch}.png"))
                expected_files.add(Path(f"{base}_freq_ch{ch}.png"))
            expected_files.add(Path(f"{base}_tf.png"))
        actual = set(self.root.rglob("*.png"))
        if expected_files != actual:
            missing = sorted(str(p) for p in expected_files - actual)[:3]
            extra = sorted(str(p) for p in actual - expected_files)[:3]
            raise ValueError(f"Image/manifest mismatch in {root}; missing={missing}, extra={extra}")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def read_image(path):
        # Keep original OpenCV RGB-to-grayscale behavior for the TF image.
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (128, 128):
            raise ValueError(f"Unreadable or wrong-size image: {path}")
        return image.astype(np.float32) / 255.0

    def __getitem__(self, index):
        row = self.rows[index]
        label = int(row["fault_label"][1:])
        base = self.root / CLASS_FOLDERS[label] / f"sample_{row['sample_id']:06d}"
        time = np.stack([self.read_image(f"{base}_time_ch{ch}.png") for ch in range(self.num_channels)])
        freq = np.stack([self.read_image(f"{base}_freq_ch{ch}.png") for ch in range(self.num_channels)])
        tf = self.read_image(f"{base}_tf.png")[None]
        return torch.from_numpy(time), torch.from_numpy(freq), torch.from_numpy(tf), label, row["sample_id"]


def make_loader(dataset, args, shuffle=False, seed=None):
    generator = torch.Generator().manual_seed(args.seed if seed is None else seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


def validate_inputs(cfg, args):
    rows = make_samples(cfg)
    read_checked_csv(args.indices / partition_name(cfg), rows, BASE_FIELDS)
    if not (args.images / "COMPLETE.json").is_file():
        raise ValueError("Image preprocessing has no root COMPLETE.json; finish ycl.py first")
    run_config = json.loads((args.images / "run_config.json").read_text(encoding="utf-8"))
    if (run_config.get("schema_version") != 2
            or run_config.get("block_points", {}).get(cfg.name) != cfg.points_per_block
            or run_config.get("segments_per_block", {}).get(cfg.name) != cfg.segments
            or run_config.get("skip_rows", {}).get(cfg.name) != cfg.skip_rows):
        raise ValueError("Image preprocessing configuration does not match the new raw-block indices")
    if cfg.name not in run_config["datasets"]:
        raise ValueError("Requested dataset is absent from preprocessing run")
    datasets = []
    if args.stage in ("cv", "all"):
        if gap_decimal(args.gap) not in [gap_decimal(g) for g in run_config["gaps"]]:
            raise ValueError("Requested gap is absent from preprocessing run")
        for fold in range(1, 6):
            expected = split_fold(rows, cfg, fold, args.gap)
            manifest = args.indices / cfg.name / gap_tag(args.gap) / f"fold{fold}.csv"
            read_checked_csv(manifest, expected, ROLE_FIELDS)
            folder = args.images / f"{cfg.prefix}_fold_images" / gap_tag(args.gap) / f"fold{fold}_img"
            if not (folder / "COMPLETE.json").is_file():
                raise ValueError(f"Incomplete fold images: {folder}")
            read_checked_csv(folder / "manifest.csv", expected, ROLE_FIELDS)
            train = [r for r in expected if r["role"] == "train"]
            val = [r for r in expected if r["role"] == "validation"]
            complete = json.loads((folder / "COMPLETE.json").read_text())
            if complete != {"train": len(train), "validation": len(val)}:
                raise ValueError(f"Invalid fold completion counts: {folder}")
            datasets.append((FoldBearingDataset(folder / "train", train, cfg.channels),
                             FoldBearingDataset(folder / "val", val, cfg.channels)))
    if args.stage in ("final", "all"):
        # Validate presence and identity without evaluating condition 2.
        for condition, suffix in ((1, "cond1_full_images"), (2, "cond2_images_fixed")):
            FoldBearingDataset(args.images / f"{cfg.prefix}_{suffix}",
                               [r for r in rows if r["condition"] == condition], cfg.channels)
    return rows, datasets


# ---------------- Training and metrics ----------------
def classification_metrics(truth, predictions):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    if truth.shape != predictions.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Invalid prediction arrays")
    if ((truth < 0) | (truth >= NUM_CLASSES) | (predictions < 0) | (predictions >= NUM_CLASSES)).any():
        raise ValueError("Invalid class IDs")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (truth, predictions), 1)
    tp = np.diag(cm).astype(float)
    denominators = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2*tp, denominators, out=np.zeros(NUM_CLASSES), where=denominators != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=1) != 0)
    return {"accuracy": float(tp.sum()/cm.sum()), "macro_f1": float(f1.mean()),
            "recall_per_class": recall.tolist(), "confusion_matrix": cm.tolist()}


def train_network(backbone, mlp, loader, args, device, active_n, epochs, input_only=False):
    if input_only:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        for name, p in mlp.named_parameters():
            p.requires_grad_(name.startswith("in_layer."))
        parameters = [p for p in mlp.parameters() if p.requires_grad]
    else:
        parameters = list(backbone.parameters()) + list(mlp.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        backbone.eval() if input_only else backbone.train()
        mlp.train()
        loss_sum, count = 0.0, 0
        for time, freq, tf, labels, _ in loader:
            time, freq, tf, labels = [x.to(device) for x in (time, freq, tf, labels)]
            scale = torch.empty(time.size(0), 1, 1, 1, device=device).uniform_(0.9, 1.1)
            if input_only:
                with torch.no_grad():
                    features = backbone(time*scale, freq*scale, tf*scale)
            else:
                features = backbone(time*scale, freq*scale, tf*scale)
            loss = criterion(mlp(features, active_n), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(labels)
            count += len(labels)
        scheduler.step()
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs:
            print(f"  epoch {epoch+1}/{epochs}: training loss={loss_sum/count:.6f}", flush=True)
    backbone.eval()
    mlp.eval()


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    features, labels, ids = [], [], []
    for time, freq, tf, target, sample_ids in loader:
        features.append(backbone(time.to(device), freq.to(device), tf.to(device)).cpu())
        labels.append(target)
        ids.append(sample_ids)
    return torch.cat(features), torch.cat(labels), torch.cat(ids)


def pre_train_single_fold(fold, datasets, cfg, args, device, output):
    train, val = datasets
    seed = args.seed + (fold-1)*1000
    set_seed(seed)
    backbone, mlp = FeatureBackbone(cfg.channels).to(device), BaseMLP().to(device)
    print(f"Fold {fold}: train={len(train)}, validation={len(val)}", flush=True)
    train_network(backbone, mlp, make_loader(train, args, True, seed), args, device,
                  MAX_NEURONS, args.epochs)
    features, labels, ids = extract_features(backbone, make_loader(val, args), device)
    folder = output / f"fold{fold}"
    folder.mkdir()
    torch.save({"backbone": cpu_state(backbone), "mlp": cpu_state(mlp),
                "fold": fold, "seed": seed, "gap_seconds": str(gap_decimal(args.gap)),
                "num_channels": cfg.channels}, folder / "pretrained.pt")
    # Cache validation features for exactly the same frozen model in every candidate.
    torch.save({"features": features, "labels": labels, "sample_ids": ids}, folder / "validation_features.pt")
    state = cpu_state(mlp)
    del backbone, mlp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"fold": fold, "features": features, "labels": labels, "sample_ids": ids,
            "mlp_state": state, "train_count": len(train), "validation_count": len(val)}


def install_candidate(mlp, candidate):
    if candidate["active_n"] not in NEURON_CHOICES:
        raise ValueError("Invalid neuron count")
    if candidate["w"].shape != mlp.in_layer.weight.shape or candidate["b"].shape != mlp.in_layer.bias.shape:
        raise ValueError("Candidate tensor shape mismatch")
    if not torch.isfinite(candidate["w"]).all() or not torch.isfinite(candidate["b"]).all():
        raise ValueError("Nonfinite candidate")
    with torch.no_grad():
        mlp.in_layer.weight.copy_(candidate["w"].to(mlp.in_layer.weight.device))
        mlp.in_layer.bias.copy_(candidate["b"].to(mlp.in_layer.bias.device))


class FrozenFoldEvaluator:
    def __init__(self, folds, args, device):
        self.folds, self.args, self.device = folds, args, device
        self.models = []
        for fold in folds:
            model = BaseMLP().to(device)
            model.load_state_dict(fold["mlp_state"])
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.models.append(model)

    @torch.no_grad()
    def __call__(self, candidate, details=False):
        results, all_predictions = [], []
        for fold, model in zip(self.folds, self.models):
            install_candidate(model, candidate)
            predictions = []
            for start in range(0, len(fold["features"]), self.args.batch_size):
                x = fold["features"][start:start+self.args.batch_size].to(self.device)
                predictions.extend(model(x, candidate["active_n"]).argmax(dim=1).cpu().tolist())
            metrics = classification_metrics(fold["labels"].numpy(), predictions)
            results.append(metrics)
            if details:
                all_predictions.append(predictions)
        accuracy = float(np.mean([r["accuracy"] for r in results]))
        fitness = 1.0-accuracy+self.args.penalty*(candidate["active_n"]/MAX_NEURONS)
        answer = {"fitness": fitness, "accuracy": accuracy}
        if details:
            answer.update(folds=results, predictions=all_predictions)
        return answer


def clone_candidate(candidate):
    return {"active_n": int(candidate["active_n"]),
            "w": candidate["w"].detach().cpu().clone(), "b": candidate["b"].detach().cpu().clone()}


class AlphaEvolutionFull:
    """Same update equations, with candidate identity and comparable scores fixed.

    Candidates stay on CPU. GPU evaluation is sequential to avoid the original
    25-process single-GPU contention. History contains scored, valid individuals.
    """
    def __init__(self, evaluator, args, output):
        self.evaluate, self.args, self.output = evaluator, args, Path(output)
        self.rng = np.random.default_rng(args.seed)
        self.generator = torch.Generator().manual_seed(args.seed)

    def bound_map(self, value):
        # Original half-distance correction, then enforce the declared bounds.
        value = torch.where(value > self.args.upper, (value+self.args.upper)/2, value)
        value = torch.where(value < self.args.lower, (value+self.args.lower)/2, value)
        return value.clamp(self.args.lower, self.args.upper)

    def initial_population(self):
        return [{"active_n": int(self.rng.choice(NEURON_CHOICES)),
                 "w": torch.empty(MAX_NEURONS, FEATURE_DIM).uniform_(self.args.lower, self.args.upper, generator=self.generator),
                 "b": torch.empty(MAX_NEURONS).uniform_(self.args.lower, self.args.upper, generator=self.generator)}
                for _ in range(self.args.population)]

    def run(self):
        population = self.initial_population()
        scores = []
        for i, candidate in enumerate(population, 1):
            scores.append(self.evaluate(candidate))
            print(f"Initial candidate {i}/{len(population)}: fitness={scores[-1]['fitness']:.6f}", flush=True)
        order = np.argsort([s["fitness"] for s in scores])
        history = [(clone_candidate(population[i]), dict(scores[i])) for i in order[:min(5, len(population))]]
        best_index = int(order[0])
        best, best_score = clone_candidate(population[best_index]), dict(scores[best_index])
        qw = torch.stack([c["w"] for c, _ in history]).mean(0)
        qb = torch.stack([c["b"] for c, _ in history]).mean(0)
        qn = float(np.mean([c["active_n"] for c, _ in history]))
        log_path = self.output / "evolution_history.csv"
        fields = ["generation", "best_fitness", "best_cv_accuracy", "active_neurons", "accepted"]
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for generation in range(self.args.generations+1):
                accepted = 0
                if generation:
                    if self.rng.random() < 0.5:
                        qw = 0.6*qw+0.4*torch.ones_like(qw)
                        qb = 0.6*qb+0.4*torch.ones_like(qb)
                        qn = 0.6*qn+0.4*128.0
                    else:
                        weights = np.array([1/(s["fitness"]+1e-8) for _, s in history])
                        weights /= weights.sum()
                        qw = sum(c["w"]*float(w) for (c, _), w in zip(history, weights))
                        qb = sum(c["b"]*float(w) for (c, _), w in zip(history, weights))
                        qn = sum(c["active_n"]*float(w) for (c, _), w in zip(history, weights))
                    h_indices = self.rng.integers(0, len(population), size=len(population))
                    candidates = []
                    for m_index in h_indices:
                        m = population[int(m_index)]
                        z, u = [population[int(self.rng.integers(len(population)))] for _ in range(2)]
                        new_n = qn+self.args.alpha*self.rng.normal()+self.args.theta*(z["active_n"]+m["active_n"]-qn-u["active_n"])
                        new_w = qw+self.args.alpha*torch.randn(qw.shape, generator=self.generator)+self.args.theta*(z["w"]+m["w"]-qw-u["w"])
                        new_b = qb+self.args.alpha*torch.randn(qb.shape, generator=self.generator)+self.args.theta*(z["b"]+m["b"]-qb-u["b"])
                        candidates.append({"active_n": min(NEURON_CHOICES, key=lambda n: abs(n-new_n)),
                                           "w": self.bound_map(new_w), "b": self.bound_map(new_b)})
                    for i, candidate in enumerate(candidates):
                        score = self.evaluate(candidate)
                        if score["fitness"] <= scores[i]["fitness"]:
                            population[i], scores[i] = clone_candidate(candidate), dict(score)
                            accepted += 1
                    current = int(np.argmin([s["fitness"] for s in scores]))
                    history = history[1:]+[(clone_candidate(population[current]), dict(scores[current]))]
                    if scores[current]["fitness"] < best_score["fitness"]:
                        best, best_score = clone_candidate(population[current]), dict(scores[current])
                writer.writerow(dict(generation=generation, best_fitness=best_score["fitness"],
                                     best_cv_accuracy=best_score["accuracy"], active_neurons=best["active_n"], accepted=accepted))
                stream.flush()
                torch.save({"candidate": best, "score": best_score, "generation": generation},
                           self.output / "search_best.pt")
                print(f"Generation {generation}/{self.args.generations}: fitness={best_score['fitness']:.6f}, CV accuracy={best_score['accuracy']:.6f}", flush=True)
        return best, best_score


def export_cv(candidate, evaluator, folds, cfg, args, output):
    result = evaluator(candidate, details=True)
    metrics_rows = []
    for fd, metrics, predictions in zip(folds, result["folds"], result["predictions"]):
        fold = fd["fold"]
        metrics_rows.append(dict(dataset=cfg.name, gap_seconds=str(gap_decimal(args.gap)),
            fold=fold, train_count=fd["train_count"], validation_count=fd["validation_count"],
            accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"]))
        prediction_rows = [dict(sample_id=int(sid), y_true=int(y), y_pred=int(p))
                           for sid, y, p in zip(fd["sample_ids"], fd["labels"], predictions)]
        write_csv(output / f"fold{fold}" / "validation_predictions.csv", prediction_rows,
                  ["sample_id", "y_true", "y_pred"])
        checkpoint = load_local_checkpoint(output / f"fold{fold}" / "pretrained.pt")
        checkpoint["mlp"]["in_layer.weight"] = candidate["w"].clone()
        checkpoint["mlp"]["in_layer.bias"] = candidate["b"].clone()
        checkpoint.update(active_n=candidate["active_n"], metrics=metrics)
        torch.save(checkpoint, output / f"fold{fold}" / "selected_model.pt")
    write_csv(output / "cv_fold_metrics.csv", metrics_rows, list(metrics_rows[0]))
    summary = {"dataset": cfg.name, "gap_seconds": str(gap_decimal(args.gap)),
               "interpretation": "CV model-selection scores; folds used for evolutionary search",
               "fold_weighting": "equal", "std_ddof": 1, "fitness": result["fitness"],
               "active_neurons": candidate["active_n"]}
    for metric in ("accuracy", "macro_f1"):
        values = np.array([r[metric] for r in metrics_rows])
        summary[metric+"_mean"] = float(values.mean())
        summary[metric+"_std"] = float(values.std(ddof=1))
        summary[metric+"_mean_percent"] = float(values.mean()*100)
        summary[metric+"_std_percentage_points"] = float(values.std(ddof=1)*100)
    save_json(output / "cv_summary.json", summary)
    return summary


def cv_settings(args):
    return {key: getattr(args, key) for key in ("seed", "batch_size", "epochs", "lr")}


@torch.no_grad()
def predict_images(backbone, mlp, loader, device, active_n):
    backbone.eval()
    mlp.eval()
    rows = []
    for time, freq, tf, labels, ids in loader:
        logits = mlp(backbone(time.to(device), freq.to(device), tf.to(device)), active_n)
        predictions = logits.argmax(dim=1).cpu().tolist()
        rows.extend(dict(sample_id=int(sid), y_true=int(y), y_pred=int(p))
                    for sid, y, p in zip(ids, labels, predictions))
    return rows


def plot_confusions(cm, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, sums, out=np.zeros_like(cm, dtype=float), where=sums != 0)
    for name, values in (("raw", cm), ("row_normalized", normalized)):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(values, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_FOLDERS,
               yticklabels=CLASS_FOLDERS, xlabel="Predicted label", ylabel="True label")
        for i in range(5):
            for j in range(5):
                text = f"{values[i,j]:.2f}" if name != "raw" else str(values[i,j])
                ax.text(j, i, text, ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output / f"condition2_confusion_{name}.png", dpi=200)
        plt.close(fig)


def retrain_and_test(candidate, rows, cfg, args, device, output):
    set_seed(args.seed)
    train = FoldBearingDataset(args.images / f"{cfg.prefix}_cond1_full_images",
                               [r for r in rows if r["condition"] == 1], cfg.channels)
    backbone, mlp = FeatureBackbone(cfg.channels).to(device), BaseMLP().to(device)
    print("Final full-condition-1 pretraining", flush=True)
    train_network(backbone, mlp, make_loader(train, args, True), args, device, MAX_NEURONS, args.epochs)
    install_candidate(mlp, candidate)
    print("Final condition-1 input-layer fine-tuning (no condition-2 access)", flush=True)
    train_network(backbone, mlp, make_loader(train, args, True, args.seed+1), args, device,
                  candidate["active_n"], args.input_epochs, input_only=True)
    torch.save({"backbone": cpu_state(backbone), "mlp": cpu_state(mlp),
                "mlp_active_n": candidate["active_n"], "mlp_max_in_dim": MAX_NEURONS,
                "random_seed": args.seed, "feature_dim": FEATURE_DIM, "num_classes": NUM_CLASSES,
                "num_channels": cfg.channels, "gap_seconds": str(gap_decimal(args.gap)),
                "dataset": cfg.name}, output / "global_opt_final.pth")
    test = FoldBearingDataset(args.images / f"{cfg.prefix}_cond2_images_fixed",
                              [r for r in rows if r["condition"] == 2], cfg.channels)
    predictions = predict_images(backbone, mlp, make_loader(test, args), device, candidate["active_n"])
    write_csv(output / "condition2_predictions.csv", predictions, ["sample_id", "y_true", "y_pred"])
    metrics = classification_metrics([r["y_true"] for r in predictions], [r["y_pred"] for r in predictions])
    metrics.update(dataset=cfg.name, gap_seconds=str(gap_decimal(args.gap)), samples=len(test))
    save_json(output / "condition2_metrics.json", metrics)
    plot_confusions(metrics["confusion_matrix"], output)
    print(f"Condition 2: accuracy={metrics['accuracy']:.6f}, macro-F1={metrics['macro_f1']:.6f}", flush=True)


def run_one_dataset(tag, args, device):
    cfg = load_dataset_configs(args.indices, DATASET_NAMES[tag])[0]
    output = args.output / f"{tag}_{gap_tag(args.gap)}"
    output.mkdir()
    rows, datasets = validate_inputs(cfg, args)
    provenance = {"dataset": cfg.name, "gap_seconds": str(gap_decimal(args.gap)),
                  "stage": args.stage, "torch": str(torch.__version__), "numpy": str(np.__version__),
                  "device": str(device), "cv_settings": cv_settings(args),
                  "final_input_epochs": args.input_epochs,
                  "schema_version": 2, "block_points": cfg.points_per_block,
                  "segments_per_block": cfg.segments, "skip_rows": cfg.skip_rows,
                  "indices": str(args.indices.resolve()), "images": str(args.images.resolve()),
                  "normalization": "ycl.py retained-training channel minmax",
                  "index_sha256": sha256_file(args.indices / partition_name(cfg)),
                  "script_sha256": sha256_file(__file__),
                  "search": {k: getattr(args, k) for k in ("population", "generations", "alpha", "theta", "penalty", "lower", "upper")}}
    save_json(output / "run_config.json", provenance)
    if args.stage in ("cv", "all"):
        folds = [pre_train_single_fold(f, ds, cfg, args, device, output) for f, ds in enumerate(datasets, 1)]
        evaluator = FrozenFoldEvaluator(folds, args, device)
        candidate, score = AlphaEvolutionFull(evaluator, args, output).run()
        summary = export_cv(candidate, evaluator, folds, cfg, args, output)
        if not math.isclose(summary["fitness"], score["fitness"], abs_tol=1e-10):
            raise RuntimeError("Saved candidate does not reproduce its search score")
        torch.save({"candidate": clone_candidate(candidate), "dataset": cfg.name,
                    "schema_version": 2, "block_points": cfg.points_per_block,
                    "segments_per_block": cfg.segments,
                    "gap_seconds": str(gap_decimal(args.gap)), "cv_settings": cv_settings(args),
                    "summary": summary}, output / "selected_candidate.pt")
        print(f"CV selection: Accuracy={summary['accuracy_mean_percent']:.4f} +/- {summary['accuracy_std_percentage_points']:.4f}%; "
              f"Macro-F1={summary['macro_f1_mean_percent']:.4f} +/- {summary['macro_f1_std_percentage_points']:.4f}%", flush=True)
        del evaluator, folds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        selected = load_local_checkpoint(args.selection)
        if (selected.get("schema_version") != 2 or selected.get("block_points") != cfg.points_per_block
                or selected.get("segments_per_block") != cfg.segments):
            raise ValueError("Candidate comes from an old or incompatible segmentation; rerun CV")
        if selected["dataset"] != cfg.name or gap_decimal(selected["gap_seconds"]) != gap_decimal(args.gap):
            raise ValueError("Selected candidate belongs to a different dataset or gap")
        if selected["cv_settings"] != cv_settings(args):
            raise ValueError("Final settings differ from CV settings (seed/batch-size/epochs/lr)")
        candidate = selected["candidate"]
        provenance["selection_sha256"] = sha256_file(args.selection)
        save_json(output / "run_config.json", provenance)
    if args.stage in ("final", "all"):
        retrain_and_test(candidate, rows, cfg, args, device, output)
    save_json(output / "COMPLETE.json", {"status": "complete", "stage": args.stage})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["seu", "lab", "all"], default="seu")
    parser.add_argument("--indices", type=Path, default=Path("partitions_5040"))
    parser.add_argument("--images", type=Path, default=Path("images_5040"))
    parser.add_argument("--gap", default="1.0")
    parser.add_argument("--output", type=Path, default=Path("training_5040_1s"))
    parser.add_argument("--stage", choices=["cv", "final", "all"], default="cv")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument("--penalty", type=float, default=0.01)
    parser.add_argument("--lower", type=float, default=-0.3)
    parser.add_argument("--upper", type=float, default=0.3)
    args = parser.parse_args()
    gap_decimal(args.gap)
    if min(args.batch_size, args.epochs, args.input_epochs, args.population) < 1 or args.generations < 0:
        parser.error("Training counts must be positive; generations must be nonnegative")
    if not all(math.isfinite(v) for v in (args.lr, args.alpha, args.theta, args.penalty, args.lower, args.upper)):
        parser.error("Hyperparameters must be finite")
    if args.lr <= 0 or args.penalty < 0 or args.lower >= args.upper:
        parser.error("Invalid learning rate, penalty, or weight bounds")
    if args.stage == "final" and (args.selection is None or args.dataset == "all"):
        parser.error("Final-only mode requires --selection and one dataset")
    if args.stage != "final" and args.selection is not None:
        parser.error("--selection is only supported for --stage final")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be new or empty; use another --output to preserve previous results")
    return args


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for tag in (DATASET_NAMES if args.dataset == "all" else [args.dataset]):
        run_one_dataset(tag, args, device)


if __name__ == "__main__":
    main()
