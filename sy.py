import csv
import numpy as np
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- 1. Southeast University dataset configuration ----------------
class SoutheastConfig:
    # Disk file prefix (without _Gxx, without .csv)
    fault_files_cond1 = ["comb_20_0", "health_20_0", "outer_20_0", "ball_20_0", "inner_20_0"]  # D0~D4
    fault_files_cond2 = ["comb_30_2", "health_30_2", "outer_30_2", "ball_30_2", "inner_30_2"]
    fault_label_map = {"comb_20_0": "D0", "health_20_0": "D1", "outer_20_0": "D2", "ball_20_0": "D3", "inner_20_0": "D4",
                       "comb_30_2": "D0", "health_30_2": "D1", "outer_30_2": "D2", "ball_30_2": "D3", "inner_30_2": "D4"}
    subfile_per_raw = 16
    seg_per_subfile = 64
    fold_num = 5
    fold_sub_cnt = [4, 3, 3, 3, 3]   # Assign each fault G01‑G16 to 5‑fold
    skip_rows = 16   # Skip header comment rows
    keep_cols = 8    # Retain the first 8‑channel columns

# ---------------- 2. Self‑built laboratory dataset configuration ----------------
class LabDatasetConfig:
    fault_files_cond1 = ["Comb_1200_0", "Health_1200_0", "Outer_1200_0", "Ball_1200_0", "Inner_1200_0"]
    fault_files_cond2 = ["Comb_1800_50", "Health_1800_50", "Outer_1800_50", "Ball_1800_50", "Inner_1800_50"]
    fault_label_map = {"Comb_1200_0": "D0", "Health_1200_0": "D1", "Outer_1200_0": "D2", "Ball_1200_0": "D3", "Inner_1200_0": "D4",
                       "Comb_1800_50": "D0", "Health_1800_50": "D1", "Outer_1800_50": "D2", "Ball_1800_50": "D3", "Inner_1800_50": "D4"}
    subfile_per_raw = 10
    seg_per_subfile = 100
    fold_num = 5
    fold_sub_cnt = [2, 2, 2, 2, 2]
    skip_rows = 5
    keep_cols = 4

def generate_partition(config, output_csv_name):
    sample_list = []
    sample_id = 0
    subfile_fold_map = dict()
    fault_subfile_ranges = dict()  # raw_fname -> fold_idx:(g_start,g_end)

    for raw_fname in config.fault_files_cond1:
        sub_ids = list(range(1, config.subfile_per_raw + 1))  # G01..G16
        ptr = 0
        fold_range = dict()
        allocated_total = 0
        for f_idx, take in enumerate(config.fold_sub_cnt):
            seg = sub_ids[ptr:ptr + take]
            for s in seg:
                subfile_fold_map[(raw_fname, s - 1)] = f_idx   # Convert subfile_id to 0‑based index
            fold_range[f_idx] = (seg[0], seg[-1])
            ptr += take
            allocated_total += take
        # Verify the total allocated quantity
        assert allocated_total == config.subfile_per_raw, f"Error in sub‑file allocation for {raw_fname}! Allocated:{allocated_total}, Expected:{config.subfile_per_raw}"
        fault_subfile_ranges[raw_fname] = fold_range

    # Working condition 1: samples for 5‑fold cross‑validation training
    for raw_fname in config.fault_files_cond1:
        label = config.fault_label_map[raw_fname]
        for sub_id in range(config.subfile_per_raw):
            fold = subfile_fold_map[(raw_fname, sub_id)]
            for seg_idx in range(config.seg_per_subfile):
                sample_list.append({
                    "sample_id": sample_id,
                    "raw_file": raw_fname,
                    "subfile_id": sub_id,
                    "segment_idx": seg_idx,
                    "fault_label": label,
                    "group_id": f"{raw_fname}_{sub_id}",
                    "fold": fold,
                    "is_train": True
                })
                sample_id += 1

    # Working condition 2: cross‑condition pure test set, fold=-1
    for raw_fname in config.fault_files_cond2:
        label = config.fault_label_map[raw_fname]
        for sub_id in range(config.subfile_per_raw):
            for seg_idx in range(config.seg_per_subfile):
                sample_list.append({
                    "sample_id": sample_id,
                    "raw_file": raw_fname,
                    "subfile_id": sub_id,
                    "segment_idx": seg_idx,
                    "fault_label": label,
                    "group_id": f"{raw_fname}_{sub_id}",
                    "fold": -1,
                    "is_train": False
                })
                sample_id += 1

    headers = ["sample_id", "raw_file", "subfile_id", "segment_idx", "fault_label", "group_id", "fold", "is_train"]
    with open(output_csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(sample_list)

    c1 = sum(1 for s in sample_list if s["fold"] != -1)
    c2 = sum(1 for s in sample_list if s["fold"] == -1)
    print(f"✅ {output_csv_name} generated successfully, total number of samples: {len(sample_list)}")
    print(f"    Samples for working‑condition‑1 (5‑fold cross‑validation): {c1}")
    print(f"    Samples for working‑condition‑2 (pure test set): {c2}")
    print(f"    → Dataset loading parameters: skip_rows={config.skip_rows}, keep_cols={config.keep_cols}")
    return sample_list, fault_subfile_ranges

def write_dataset_config():
    """Generate unified parameter configuration file for dataset reading"""
    rows = [
        {"dataset": "southeast", "skip_rows": SoutheastConfig.skip_rows, "keep_cols": SoutheastConfig.keep_cols},
        {"dataset": "laboratory", "skip_rows": LabDatasetConfig.skip_rows, "keep_cols": LabDatasetConfig.keep_cols},
    ]
    with open("dataset_config.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "skip_rows", "keep_cols"])
        w.writeheader()
        w.writerows(rows)
    print("✅ dataset_config.csv has been generated (skip_rows / keep_cols)")

def gen_tables(se_ranges, se_cfg, lab_cfg):
    out = []
    out.append("**Table S1. Detailed grouping of the Southeast University and laboratory datasets**")
    out.append("| Raw file | Fault class | Sub-file → Fold |")
    out.append("|---|---|---|")
    for row, fname, cls in zip(["F1", "F2", "F3", "F4", "F5"],
                               se_cfg.fault_files_cond1, ["D0", "D1", "D2", "D3", "D4"]):
        parts = []
        for fi in range(se_cfg.fold_num):
            g0, g1 = se_ranges[fname][fi]
            parts.append(f"G{g0:02d}–G{g1:02d}→Fold {fi+1}")
        out.append(f"| {row} | {cls} | {','.join(parts)} |")
    out.append("")
    out.append("*Note: The full fold assignment of all 80 sub-files is provided in range notation; each group is uniquely determined by the raw-file number and the sub-file number.*")
    out.append("")
    out.append("**Table S2. Per‑fold composition of the grouped five‑fold cross‑validation (Southeast University, Working Condition 1).**")
    out.append("| Fold | Validation: sub-files | Validation samples | Validation sub‑files per class (D0/D1/D2/D3/D4) | Training: sub-files | Training samples |")
    out.append("|---|---|---|---|---|---|")
    perclass = ["4 / 3 / 3 / 3 / 3", "3 / 4 / 3 / 3 / 3", "3 / 3 / 4 / 3 / 3", "3 / 3 / 3 / 4 / 3", "3 / 3 / 3 / 3 / 4"]
    for fi in range(5):
        val_sub = 4 + 3 * 4
        val_samp = val_sub * se_cfg.seg_per_subfile
        train_sub = 80 - val_sub
        train_samp = train_sub * se_cfg.seg_per_subfile
        out.append(f"| {fi+1} | {val_sub} | {val_samp:,} | {perclass[fi]} | {train_sub} | {train_samp:,} |")
    out.append("")
    out.append("**Table S3. Per‑fold composition (laboratory dataset, Working Condition 1).**")
    out.append("| Fold | Validation: sub-files | Validation samples | Validation sub‑files per class (D0/D1/D2/D3/D4) | Training: sub-files | Training samples |")
    out.append("|---|---|---|---|---|---|")
    val_sub = 2 * 5
    val_samp = val_sub * lab_cfg.seg_per_subfile
    train_sub = 50 - val_sub
    train_samp = train_sub * lab_cfg.seg_per_subfile
    out.append(f"| 1–5 | {val_sub} | {val_samp:,} | 2 / 2 / 2 / 2 / 2 | {train_sub} | {train_samp:,} |")
    out.append("")
    text = "\n".join(out)
    with open("Table_S1_S2_S3.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n✅ Table_S1_S2_S3.md has been completely generated (including tables S1/S2/S3)")
    print("=" * 80)
    print(text)

if __name__ == "__main__":
    se_part, se_ranges = generate_partition(SoutheastConfig, "southeast_dataset_partition.csv")
    lab_part, lab_ranges = generate_partition(LabDatasetConfig, "lab_dataset_partition.csv")
    write_dataset_config()
    gen_tables(se_ranges, SoutheastConfig, LabDatasetConfig)
