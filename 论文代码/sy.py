import csv
import numpy as np
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------- 1. Southeast University Dataset Configuration ----------------
class SoutheastConfig:
    # Raw file prefix (without _Gxx, without .csv extension)
    fault_files_cond1 = ["comb_20_0", "health_20_0", "outer_20_0", "ball_20_0", "inner_20_0"]  # D0~D4
    fault_files_cond2 = ["comb_30_2", "health_30_2", "outer_30_2", "ball_30_2", "inner_30_2"]
    fault_label_map = {
        "comb_20_0": "D0", "health_20_0": "D1", "outer_20_0": "D2", "ball_20_0": "D3", "inner_20_0": "D4",
        "comb_30_2": "D0", "health_30_2": "D1", "outer_30_2": "D2", "ball_30_2": "D3", "inner_30_2": "D4"
    }
    subfile_per_raw = 16       # Number of sub-files per raw recording
    seg_per_subfile = 64       # Number of vibration segments per sub-file
    fold_num = 5
    fold_sub_cnt = [4, 3, 3, 3, 3]  # Number of sub-files allocated to each fold per fault class
    skip_rows = 16             # Header rows to skip in raw CSV
    keep_cols = 8              # Number of signal channels to keep

# ---------------- 2. Laboratory Dataset Configuration ----------------
class LabDatasetConfig:
    fault_files_cond1 = ["Comb_1200_0", "Health_1200_0", "Outer_1200_0", "Ball_1200_0", "Inner_1200_0"]
    fault_files_cond2 = ["Comb_1800_50", "Health_1800_50", "Outer_1800_50", "Ball_1800_50", "Inner_1800_50"]
    fault_label_map = {
        "Comb_1200_0": "D0", "Health_1200_0": "D1", "Outer_1200_0": "D2", "Ball_1200_0": "D3", "Inner_1200_0": "D4",
        "Comb_1800_50": "D0", "Health_1800_50": "D1", "Outer_1800_50": "D2", "Ball_1800_50": "D3", "Inner_1800_50": "D4"
    }
    subfile_per_raw = 10
    seg_per_subfile = 100
    fold_num = 5
    fold_sub_cnt = [2, 2, 2, 2, 2]
    skip_rows = 5
    keep_cols = 4

def generate_partition(config, output_csv_name):
    """
    Generate dataset partition index file with grouped 5-fold cross-validation.
    Sub-files from the same raw recording are assigned to the same fold to avoid data leakage.
    Fold indices are 1-based to align with downstream training code.
    """
    sample_list = []
    sample_id = 0
    subfile_fold_map = dict()
    fault_subfile_ranges = dict()  # raw_fname -> fold_idx: (g_start, g_end)

    # Allocate sub-files to each fold for condition 1
    for raw_fname in config.fault_files_cond1:
        sub_ids = list(range(1, config.subfile_per_raw + 1))
        ptr = 0
        fold_range = dict()
        allocated_total = 0

        for f_idx, take in enumerate(config.fold_sub_cnt):
            seg = sub_ids[ptr:ptr + take]
            for s in seg:
                # Fix C03: fold index starts from 1 to match downstream evaluation code
                subfile_fold_map[(raw_fname, s - 1)] = f_idx + 1
            fold_range[f_idx + 1] = (seg[0], seg[-1])
            ptr += take
            allocated_total += take

        assert allocated_total == config.subfile_per_raw, \
            f"{raw_fname} sub-file allocation error: allocated {allocated_total}, expected {config.subfile_per_raw}"
        fault_subfile_ranges[raw_fname] = fold_range

    # Generate samples for Working Condition 1 (training + cross-validation)
    for raw_fname in config.fault_files_cond1:
        label = config.fault_label_map[raw_fname]
        for subfile_id in range(config.subfile_per_raw):
            fold = subfile_fold_map[(raw_fname, subfile_id)]
            for seg_idx in range(config.seg_per_subfile):
                sample_list.append({
                    "sample_id": sample_id,
                    "raw_file": raw_fname,
                    "subfile_id": subfile_id,
                    "segment_idx": seg_idx,
                    "fault_label": label,
                    "group_id": f"{raw_fname}_{subfile_id}",
                    "fold": fold,
                    "is_train": True
                })
                sample_id += 1

    # Generate samples for Working Condition 2 (generalization test only, no fold assignment)
    for raw_fname in config.fault_files_cond2:
        label = config.fault_label_map[raw_fname]
        for subfile_id in range(config.subfile_per_raw):
            for seg_idx in range(config.seg_per_subfile):
                sample_list.append({
                    "sample_id": sample_id,
                    "raw_file": raw_fname,
                    "subfile_id": subfile_id,
                    "segment_idx": seg_idx,
                    "fault_label": label,
                    "group_id": f"{raw_fname}_{subfile_id}",
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
    print(f"✅ {output_csv_name} generated successfully, total samples: {len(sample_list)}")
    print(f"    Condition 1 (5-fold CV) samples: {c1}")
    print(f"    Condition 2 (test only) samples: {c2}")
    print(f"    → Dataset reading parameters: skip_rows={config.skip_rows}, keep_cols={config.keep_cols}")
    return sample_list, fault_subfile_ranges

def write_dataset_config():
    """Write global dataset reading configuration file"""
    rows = [
        {"dataset": "southeast", "skip_rows": SoutheastConfig.skip_rows, "keep_cols": SoutheastConfig.keep_cols},
        {"dataset": "laboratory", "skip_rows": LabDatasetConfig.skip_rows, "keep_cols": LabDatasetConfig.keep_cols},
    ]
    with open("dataset_config.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "skip_rows", "keep_cols"])
        w.writeheader()
        w.writerows(rows)
    print("✅ dataset_config.csv generated (skip_rows / keep_cols)")

def gen_tables(se_ranges, se_cfg, lab_ranges, lab_cfg):
    """Generate supplementary tables S1-S4 in markdown format"""
    out = []
    # ========== Table S1: Southeast University sub-file grouping ==========
    out.append("**Table S1. Detailed grouping of the Southeast University dataset**")
    out.append("| Raw file | Fault class | Sub-file → Fold |")
    out.append("|---|---|---|")
    for row, fname, cls in zip(
        ["F1", "F2", "F3", "F4", "F5"],
        se_cfg.fault_files_cond1,
        ["D0", "D1", "D2", "D3", "D4"]
    ):
        parts = []
        for fi in range(1, se_cfg.fold_num + 1):
            g0, g1 = se_ranges[fname][fi]
            parts.append(f"G{g0:02d}–G{g1:02d}→Fold {fi}")
        out.append(f"| {row} | {cls} | {','.join(parts)} |")
    out.append("")
    out.append("*Note: The full fold assignment of all 80 sub‑files is provided in range notation; "
               "each group is uniquely determined by the raw‑file number and the sub‑file number.*")
    out.append("")

    # ========== Table S2: Southeast 5-fold summary ==========
    out.append("**Table S2. Per‑fold composition of the grouped five‑fold cross‑validation (Southeast University, Working Condition 1).**")
    out.append("| Fold | Validation: sub‑files | Validation samples | Validation sub‑files per class (D0/D1/D2/D3/D4) | Training: sub‑files | Training samples |")
    out.append("|---|---|---|---|---|---|")
    fold_info_se = [
        {"val_sub": 5*4, "val_per_cls": "4 / 4 / 4 / 4 / 4"},
        {"val_sub": 5*3, "val_per_cls": "3 / 3 / 3 / 3 / 3"},
        {"val_sub": 5*3, "val_per_cls": "3 / 3 / 3 / 3 / 3"},
        {"val_sub": 5*3, "val_per_cls": "3 / 3 / 3 / 3 / 3"},
        {"val_sub": 5*3, "val_per_cls": "3 / 3 / 3 / 3 / 3"},
    ]
    total_se_sub = 5 * se_cfg.subfile_per_raw
    for fi, info in enumerate(fold_info_se):
        fold_idx = fi + 1
        val_sub = info["val_sub"]
        val_samp = val_sub * se_cfg.seg_per_subfile
        train_sub = total_se_sub - val_sub
        train_samp = train_sub * se_cfg.seg_per_subfile
        out.append(f"| {fold_idx} | {val_sub} | {val_samp:,} | {info['val_per_cls']} | {train_sub} | {train_samp:,} |")
    out.append("")
    out.append("*Note: Validation‑set sample sizes differ across folds. Group constraints are strictly obeyed: "
               "all sub‑segments from one sub‑file belong to the same fold and are never split between training and validation.*")
    out.append("")

    # ========== Table S3: Laboratory dataset sub-file grouping ==========
    out.append("**Table S3. Detailed grouping of the laboratory dataset**")
    out.append("| Raw file | Fault class | Sub-file → Fold |")
    out.append("|---|---|---|")
    for row, fname, cls in zip(
        ["L1", "L2", "L3", "L4", "L5"],
        lab_cfg.fault_files_cond1,
        ["D0", "D1", "D2", "D3", "D4"]
    ):
        parts = []
        for fi in range(1, lab_cfg.fold_num + 1):
            g0, g1 = lab_ranges[fname][fi]
            parts.append(f"G{g0:02d}–G{g1:02d}→Fold {fi}")
        out.append(f"| {row} | {cls} | {','.join(parts)} |")
    out.append("")
    out.append("*Note: The full fold assignment of all 50 sub‑files is provided in range notation; "
               "each group is uniquely determined by the raw‑file number and the sub‑file number.*")
    out.append("")

    # ========== Table S4: Laboratory 5-fold summary ==========
    out.append("**Table S4. Per‑fold composition of the grouped five‑fold cross‑validation (laboratory dataset, Working Condition 1).**")
    out.append("| Fold | Validation: sub‑files | Validation samples | Validation sub‑files per class (D0/D1/D2/D3/D4) | Training: sub‑files | Training samples |")
    out.append("|---|---|---|---|---|---|")
    lab_val_per_cls = "2 / 2 / 2 / 2 / 2"
    lab_val_sub = 5 * lab_cfg.fold_sub_cnt[0]
    lab_val_samp = lab_val_sub * lab_cfg.seg_per_subfile
    lab_total_sub = 5 * lab_cfg.subfile_per_raw
    lab_train_sub = lab_total_sub - lab_val_sub
    lab_train_samp = lab_train_sub * lab_cfg.seg_per_subfile
    for fi in range(1, 6):
        out.append(f"| {fi} | {lab_val_sub} | {lab_val_samp:,} | {lab_val_per_cls} | {lab_train_sub} | {lab_train_samp:,} |")
    out.append("")
    out.append("*Note: Each fold has equal‑sized validation set. "
               "All sub‑segments originating from the same sub‑file are assigned to one single fold.*")
    out.append("")

    text = "\n".join(out)
    with open("Table_S1_S2_S3_S4.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n✅ Table_S1_S2_S3_S4.md generated successfully")
    print("=" * 80)
    print(text)

if __name__ == "__main__":
    se_part, se_ranges = generate_partition(SoutheastConfig, "southeast_dataset_partition.csv")
    lab_part, lab_ranges = generate_partition(LabDatasetConfig, "lab_dataset_partition.csv")
    write_dataset_config()
    gen_tables(se_ranges, SoutheastConfig, lab_ranges, LabDatasetConfig)
