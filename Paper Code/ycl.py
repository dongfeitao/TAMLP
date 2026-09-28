import os
import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft
from PIL import Image
import matplotlib.pyplot as plt

# ================= File Name Mapping =================
se_mapping = {
    "C-20-0": "comb_20_0", "H-20-0": "health_20_0", "O-20-0": "outer_20_0",
    "B-20-0": "ball_20_0", "I-20-0": "inner_20_0",
    "C-30-2": "comb_30_2", "H-30-2": "health_30_2", "O-30-2": "outer_30_2",
    "B-30-2": "ball_30_2", "I-30-2": "inner_30_2",
}
lab_mapping = {
    "C-1200-0%": "Comb_1200_0", "H-1200-0%": "Health_1200_0", "O-1200-0%": "Outer_1200_0",
    "B-1200-0%": "Ball_1200_0", "I-1200-0%": "Inner_1200_0",
    "C-1800-50%": "Comb_1800_50", "H-1800-50%": "Health_1800_50", "O-1800-50%": "Outer_1800_50",
    "B-1800-50%": "Ball_1800_50", "I-1800-50%": "Inner_1800_50",
}
label_to_folder = {"D0": "C", "D1": "H", "D2": "O", "D3": "B", "D4": "I"}


def fix_partition_csv():
    def _fix_one(csv_path, mapping):
        df = pd.read_csv(csv_path)
        df.to_csv(csv_path + ".bak", index=False)
        df["raw_file_old"] = df["raw_file"].copy()
        def safe_map(x):
            return mapping[x] if x in mapping else x
        df["raw_file"] = df["raw_file"].apply(safe_map)
        nan_mask = df["raw_file"].isna()
        if nan_mask.any():
            print(f"\n!!!!!!!! ERROR: Unmatched raw_file values in {csv_path} !!!!!!!!")
            error_rows = df.loc[nan_mask, ["raw_file_old"]].drop_duplicates()
            print("Unmatched raw codes:")
            print(error_rows.to_string(index=False))
            raise SystemExit("Program terminated. Please add the above codes to the mapping dictionary.")
        df.drop(columns=["raw_file_old"], inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated {csv_path}, backup saved as {csv_path}.bak")
    _fix_one("southeast_dataset_partition.csv", se_mapping)
    _fix_one("lab_dataset_partition.csv", lab_mapping)
    se_df = pd.read_csv("southeast_dataset_partition.csv")
    lab_df = pd.read_csv("lab_dataset_partition.csv")
    print("\nSoutheast raw_file list:", sorted(se_df["raw_file"].unique()))
    print("Laboratory raw_file list:", sorted(lab_df["raw_file"].unique()))


def data_read(file_path, skip_rows, read_cols):
    try:
        df = pd.read_csv(file_path, header=None, sep='\t', engine='python', skip_blank_lines=True)
        df = df.iloc[skip_rows:, 0:read_cols].astype(float)
        if df.isnull().values.any():
            raise ValueError(f"NaN values detected in {file_path}")
        if not np.isfinite(df.values).all():
            raise ValueError(f"Inf values detected in {file_path}")
        if df.shape[1] != read_cols:
            raise ValueError(f"Channel mismatch in {file_path}: expected {read_cols}, got {df.shape[1]}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {str(e)}") from e


# ===================== Global Parameters =====================
SEG_LENGTH = 1024
IMG_SIZE = (128, 128)
EPS = 1e-12
WAVELET_NAME = "morl"
plt.switch_backend("Agg")

DATASET_CONFIGS = [
    {
        "name": "southeast",
        "partition_csv": "southeast_dataset_partition.csv",
        "raw_root": "/root/SJJ/data",
        "out_fold_root": "./se_fold_images",
        "cond1_img_path": "./se_cond1_full_images",
        "cond2_img_path": "./se_cond2_images_fixed",
        "fs": 5120,
        "primary_ch": 0,
        "skip_rows": 16,
        "read_cols": 8
    },
    {
        "name": "laboratory",
        "partition_csv": "lab_dataset_partition.csv",
        "raw_root": "/root/SJJ/data",
        "out_fold_root": "./lab_fold_images",
        "cond1_img_path": "./lab_cond1_full_images",
        "cond2_img_path": "./lab_cond2_images_fixed",
        "fs": 10240,
        "primary_ch": 0,
        "skip_rows": 5,
        "read_cols": 4
    }
]


class SignalPreprocessor:
    def __init__(self):
        self.channel_stats = dict()

    def fit_train_normalize(self, train_signal_dict):
        self.channel_stats.clear()
        for ch, sig in train_signal_dict.items():
            ch_min = float(np.min(sig))
            ch_max = float(np.max(sig))
            self.channel_stats[ch] = {"min": ch_min, "max": ch_max}

    def transform_normalize(self, signal_dict):
        out = dict()
        for ch, sig in signal_dict.items():
            if ch not in self.channel_stats:
                raise KeyError(f"Channel {ch} statistics not fitted. Call fit_train_normalize first.")
            s_min, s_max = self.channel_stats[ch]["min"], self.channel_stats[ch]["max"]
            norm_sig = np.zeros_like(sig) if abs(s_max - s_min) < 1e-10 else (sig - s_min) / (s_max - s_min)
            out[ch] = norm_sig
        return out

    def build_time_domain_grayscale(self, sig_1d: np.ndarray) -> Image.Image:
        sig_resample = np.interp(np.linspace(0, len(sig_1d)-1, IMG_SIZE[0]), np.arange(len(sig_1d)), sig_1d)
        img_arr = np.tile(sig_resample.reshape(1, -1), (IMG_SIZE[1], 1))
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_arr, mode="L")

    def build_log_spectrum_image(self, sig_1d: np.ndarray, fs: int) -> Image.Image:
        yf = rfft(sig_1d)
        amp = np.abs(yf)
        log_amp = np.log10(amp + EPS)
        log_resample = np.interp(np.linspace(0, len(log_amp)-1, IMG_SIZE[0]), np.arange(len(log_amp)), log_amp)
        img_arr = np.tile(log_resample.reshape(1, -1), (IMG_SIZE[1], 1))
        vmin, vmax = float(np.min(img_arr)), float(np.max(img_arr))
        if abs(vmax - vmin) > 1e-10:
            img_arr = ((img_arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            img_arr = np.zeros_like(img_arr, dtype=np.uint8)
        return Image.fromarray(img_arr, mode="L")

    def build_cwt_timefreq_image(self, sig_1d: np.ndarray, fs: int) -> Image.Image:
        scales = np.linspace(1, 128, 128)
        coeffs, _ = pywt.cwt(sig_1d, scales, wavelet=WAVELET_NAME, sampling_period=1.0/fs)
        coeff_abs = np.abs(coeffs)
        fig, ax = plt.subplots(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)
        ax.imshow(coeff_abs, cmap="viridis", aspect="auto")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), buf)
        img = img.resize(IMG_SIZE)
        plt.close(fig)
        return img


def read_single_segment(csv_path, subfile_id, seg_idx, seg_per_subfile, skip_rows, read_cols):
    df_raw = data_read(csv_path, skip_rows, read_cols)
    total_samples = len(df_raw)
    global_seg_idx = subfile_id * seg_per_subfile + seg_idx
    n_start = global_seg_idx * SEG_LENGTH
    n_end = n_start + SEG_LENGTH
    if n_end > total_samples:
        raise ValueError(f"Segment out of range in {csv_path}: subfile={subfile_id}, seg={seg_idx}, start={n_start}, end={n_end}, total={total_samples}")
    slice_data = df_raw.iloc[n_start:n_end, :].to_numpy()
    return {ch: slice_data[:, ch].astype(np.float64) for ch in range(slice_data.shape[1])}


def generate_fold_images(cfg):
    df_full = pd.read_csv(cfg["partition_csv"])
    cond1_df = df_full[df_full["fold"] != -1].reset_index(drop=True)
    seg_per_subfile = 64 if cfg["name"] == "southeast" else 100
    num_ch = cfg["read_cols"]
    primary_ch = cfg["primary_ch"]

    os.makedirs(cfg["out_fold_root"], exist_ok=True)
    for fold_idx in range(1, 6):
        print(f"\n===== Processing Fold {fold_idx}/5 for {cfg['name']} =====")
        fold_path = os.path.join(cfg["out_fold_root"], f"fold{fold_idx}_img")
        train_path = os.path.join(fold_path, "train")
        val_path = os.path.join(fold_path, "val")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        train_mask = cond1_df["fold"] != fold_idx
        val_mask = cond1_df["fold"] == fold_idx
        train_df = cond1_df[train_mask].reset_index(drop=True)
        val_df = cond1_df[val_mask].reset_index(drop=True)

        preproc = SignalPreprocessor()
        train_signals_by_channel = {ch: [] for ch in range(num_ch)}
        print("  Fitting normalization stats on all channels of training set...")
        for _, row in train_df.iterrows():
            sig_dict = read_single_segment(
                os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
                int(row["subfile_id"]), int(row["segment_idx"]),
                seg_per_subfile, cfg["skip_rows"], num_ch
            )
            for ch in range(num_ch):
                train_signals_by_channel[ch].append(sig_dict[ch])
        fit_dict = {ch: np.concatenate(sigs) for ch, sigs in train_signals_by_channel.items()}
        preproc.fit_train_normalize(fit_dict)
        del train_signals_by_channel

        print("  Generating training set images...")
        for idx, row in train_df.iterrows():
            sid = int(row["sample_id"])
            label = row["fault_label"]
            folder_label = label_to_folder[label]
            sig_dict = read_single_segment(
                os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
                int(row["subfile_id"]), int(row["segment_idx"]),
                seg_per_subfile, cfg["skip_rows"], num_ch
            )
            sig_norm = preproc.transform_normalize(sig_dict)
            save_dir = os.path.join(train_path, folder_label)
            os.makedirs(save_dir, exist_ok=True)

            # 时域：全通道分通道保存
            for ch in range(num_ch):
                preproc.build_time_domain_grayscale(sig_norm[ch]).save(
                    os.path.join(save_dir, f"sample_{sid:06d}_time_ch{ch}.png"))
            # 频域：全通道分通道保存
            for ch in range(num_ch):
                preproc.build_log_spectrum_image(sig_norm[ch], fs=cfg["fs"]).save(
                    os.path.join(save_dir, f"sample_{sid:06d}_freq_ch{ch}.png"))
            # 时频：仅第0通道单张保存
            preproc.build_cwt_timefreq_image(sig_norm[primary_ch], fs=cfg["fs"]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_tf.png"))

            if (idx + 1) % 500 == 0:
                print(f"     Train progress: {idx + 1}/{len(train_df)}")

        print("  Generating validation set images...")
        for idx, row in val_df.iterrows():
            sid = int(row["sample_id"])
            label = row["fault_label"]
            folder_label = label_to_folder[label]
            sig_dict = read_single_segment(
                os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
                int(row["subfile_id"]), int(row["segment_idx"]),
                seg_per_subfile, cfg["skip_rows"], num_ch
            )
            sig_norm = preproc.transform_normalize(sig_dict)
            save_dir = os.path.join(val_path, folder_label)
            os.makedirs(save_dir, exist_ok=True)

            for ch in range(num_ch):
                preproc.build_time_domain_grayscale(sig_norm[ch]).save(
                    os.path.join(save_dir, f"sample_{sid:06d}_time_ch{ch}.png"))
            for ch in range(num_ch):
                preproc.build_log_spectrum_image(sig_norm[ch], fs=cfg["fs"]).save(
                    os.path.join(save_dir, f"sample_{sid:06d}_freq_ch{ch}.png"))
            preproc.build_cwt_timefreq_image(sig_norm[primary_ch], fs=cfg["fs"]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_tf.png"))

            if (idx + 1) % 200 == 0:
                print(f"     Val progress: {idx + 1}/{len(val_df)}")
    print(f"\n✅ Fold-wise images generated for {cfg['name']}: {cfg['out_fold_root']}")


def generate_cond1_full_images(cfg):
    df_full = pd.read_csv(cfg["partition_csv"])
    cond1_df = df_full[df_full["fold"] != -1].reset_index(drop=True)
    seg_per_subfile = 64 if cfg["name"] == "southeast" else 100
    num_ch = cfg["read_cols"]
    primary_ch = cfg["primary_ch"]

    out_path = cfg["cond1_img_path"]
    os.makedirs(out_path, exist_ok=True)
    print(f"\n===== Generating Condition 1 full images for {cfg['name']} =====")

    preproc = SignalPreprocessor()
    all_signals_by_channel = {ch: [] for ch in range(num_ch)}
    print("  Fitting global normalization stats on all channels of Condition 1...")
    for _, row in cond1_df.iterrows():
        sig_dict = read_single_segment(
            os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
            int(row["subfile_id"]), int(row["segment_idx"]),
            seg_per_subfile, cfg["skip_rows"], num_ch
        )
        for ch in range(num_ch):
            all_signals_by_channel[ch].append(sig_dict[ch])
    fit_dict = {ch: np.concatenate(sigs) for ch, sigs in all_signals_by_channel.items()}
    preproc.fit_train_normalize(fit_dict)
    del all_signals_by_channel

    print(f"  Generating Condition 1 full images ({len(cond1_df)} samples)...")
    for idx, row in cond1_df.iterrows():
        sid = int(row["sample_id"])
        label = row["fault_label"]
        folder_label = label_to_folder[label]
        sig_dict = read_single_segment(
            os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
            int(row["subfile_id"]), int(row["segment_idx"]),
            seg_per_subfile, cfg["skip_rows"], num_ch
        )
        sig_norm = preproc.transform_normalize(sig_dict)
        save_dir = os.path.join(out_path, folder_label)
        os.makedirs(save_dir, exist_ok=True)

        for ch in range(num_ch):
            preproc.build_time_domain_grayscale(sig_norm[ch]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_time_ch{ch}.png"))
        for ch in range(num_ch):
            preproc.build_log_spectrum_image(sig_norm[ch], fs=cfg["fs"]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_freq_ch{ch}.png"))
        preproc.build_cwt_timefreq_image(sig_norm[primary_ch], fs=cfg["fs"]).save(
            os.path.join(save_dir, f"sample_{sid:06d}_tf.png"))

        if (idx + 1) % 500 == 0:
            print(f"     Progress: {idx + 1}/{len(cond1_df)}")
    print(f"✅ Condition 1 full images saved to: {out_path}")


def generate_cond2_images(cfg):
    df_full = pd.read_csv(cfg["partition_csv"])
    cond1_df = df_full[df_full["fold"] != -1].reset_index(drop=True)
    cond2_df = df_full[df_full["fold"] == -1].reset_index(drop=True)
    seg_per_subfile = 64 if cfg["name"] == "southeast" else 100
    num_ch = cfg["read_cols"]
    primary_ch = cfg["primary_ch"]

    print(f"\n===== Generating Condition 2 images for {cfg['name']} =====")
    preproc = SignalPreprocessor()
    train_signals_by_channel = {ch: [] for ch in range(num_ch)}
    print("  Fitting normalization stats on all channels of full Condition 1 training set...")
    for _, row in cond1_df.iterrows():
        sig_dict = read_single_segment(
            os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
            int(row["subfile_id"]), int(row["segment_idx"]),
            seg_per_subfile, cfg["skip_rows"], num_ch
        )
        for ch in range(num_ch):
            train_signals_by_channel[ch].append(sig_dict[ch])
    fit_dict = {ch: np.concatenate(sigs) for ch, sigs in train_signals_by_channel.items()}
    preproc.fit_train_normalize(fit_dict)
    del train_signals_by_channel

    out_path = cfg["cond2_img_path"]
    os.makedirs(out_path, exist_ok=True)
    print("  Generating Condition 2 test images...")
    for idx, row in cond2_df.iterrows():
        sid = int(row["sample_id"])
        label = row["fault_label"]
        folder_label = label_to_folder[label]
        sig_dict = read_single_segment(
            os.path.join(cfg["raw_root"], row["raw_file"] + ".csv"),
            int(row["subfile_id"]), int(row["segment_idx"]),
            seg_per_subfile, cfg["skip_rows"], num_ch
        )
        sig_norm = preproc.transform_normalize(sig_dict)
        save_dir = os.path.join(out_path, folder_label)
        os.makedirs(save_dir, exist_ok=True)

        for ch in range(num_ch):
            preproc.build_time_domain_grayscale(sig_norm[ch]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_time_ch{ch}.png"))
        for ch in range(num_ch):
            preproc.build_log_spectrum_image(sig_norm[ch], fs=cfg["fs"]).save(
                os.path.join(save_dir, f"sample_{sid:06d}_freq_ch{ch}.png"))
        preproc.build_cwt_timefreq_image(sig_norm[primary_ch], fs=cfg["fs"]).save(
            os.path.join(save_dir, f"sample_{sid:06d}_tf.png"))

        if (idx + 1) % 500 == 0:
            print(f"     Progress: {idx + 1}/{len(cond2_df)}")
    print(f"✅ Condition 2 images saved to: {out_path}")


if __name__ == "__main__":
    fix_partition_csv()
    for cfg in DATASET_CONFIGS:
        ds_name = cfg["name"]
        skip = cfg["skip_rows"]
        ncol = cfg["read_cols"]
        print(f"\n{'=' * 70}")
        print(f"===== Processing dataset: {ds_name} | skip_rows={skip}, channels={ncol} =====")
        print(f"{'=' * 70}")
        generate_fold_images(cfg)
        generate_cond1_full_images(cfg)
        generate_cond2_images(cfg)
    print("\n🎉 All preprocessing tasks completed!")
