import os
import shutil
import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft, rfftfreq
from PIL import Image
import matplotlib.pyplot as plt

# ================= File‑name mapping (raw_file keys inside the partition file) =================
se_mapping = {
    "C-20-0": "comb_20_0",
    "H-20-0": "health_20_0",
    "O-20-0": "outer_20_0",
    "B-20-0": "ball_20_0",
    "I-20-0": "inner_20_0",
    "C-30-2": "comb_30_2",
    "H-30-2": "health_30_2",
    "O-30-2": "outer_30_2",
    "B-30-2": "ball_30_2",
    "I-30-2": "inner_30_2",
}

lab_mapping = {
    "C-1200-0%": "Comb_1200_0",
    "H-1200-0%": "Health_1200_0",
    "O-1200-0%": "Outer_1200_0",
    "B-1200-0%": "Ball_1200_0",
    "I-1200-0%": "Inner_1200_0",
    "C-1800-50%": "Comb_1800_50",
    "H-1800-50%": "Health_1800_50",
    "O-1800-50%": "Outer_1800_50",
    "B-1800-50%": "Ball_1800_50",
    "I-1800-50%": "Inner_1800_50",
}


def fix_partition_csv():
    """
    Intelligent mapping: skip conversion if real file‑names already exist;
    perform mapping for code‑style identifiers; terminate with NaN‑error only for truly unrecognizable values
    """

    def _fix_one(csv_path, mapping):
        df = pd.read_csv(csv_path)
        df.to_csv(csv_path + ".bak", index=False)
        df["raw_file_old"] = df["raw_file"].copy()

        def safe_map(x):
            if x in mapping:
                return mapping[x]
            else:
                # Already converted real file‑name, pass through directly
                return x

        df["raw_file"] = df["raw_file"].apply(safe_map)

        nan_mask = df["raw_file"].isna()
        if nan_mask.any():
            print(f"\n!!!!!!!! ERROR: Unidentifiable raw_file value exists in {csv_path} !!!!!!!!")
            error_rows = df.loc[nan_mask, ["raw_file_old"]].drop_duplicates()
            print("Original identifiers unmatched in the mapping dictionary:")
            print(error_rows.to_string(index=False))
            raise SystemExit("Program terminated. Please add the above identifiers into the mapping dictionary!")

        df.drop(columns=["raw_file_old"], inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"✅Updated {csv_path}, backup saved as {csv_path}.bak")

    _fix_one("southeast_dataset_partition.csv", se_mapping)
    _fix_one("lab_dataset_partition.csv", lab_mapping)
    se_df = pd.read_csv("southeast_dataset_partition.csv")
    lab_df = pd.read_csv("lab_dataset_partition.csv")
    print("\nSoutheast raw_file list:", sorted(se_df["raw_file"].unique()))
    print("Laboratory raw_file list:", sorted(lab_df["raw_file"].unique()))


def data_read(file_path, skip_rows, read_cols):
    """
    Dual‑strategy fault‑tolerant reading for vibration CSV files
    :param file_path: path of source file
    :param skip_rows: number of header rows to skip
    :param read_cols: read the first N columns (8 for Southeast dataset, 4 for laboratory dataset)
    :return df: loaded data frame
    """
    try:
        df = pd.read_csv(file_path, header=None)
        df = df.iloc[skip_rows:, 0:read_cols]
        df = df.astype(float)
        return df
    except:
        try:
            df = pd.read_csv(file_path, header=None)
            df.columns = ['acc_data']
            df = df.iloc[skip_rows:, :]
            df['acc_data'] = df['acc_data'].astype(str)
            split_data = df['acc_data'].str.split('\t', expand=True)

            # Fill NaN if available columns are insufficient
            if split_data.shape[1] < read_cols:
                for i in range(split_data.shape[1], read_cols):
                    split_data[i] = np.nan
                df = split_data.iloc[:, 0:read_cols]
            else:
                df = split_data.iloc[:, 0:read_cols]

            df = df.astype(float)
            return df
        except Exception as e:
            print(f"All reading methods failed: {str(e)} , File:{file_path}")
            return pd.DataFrame(np.zeros((100, read_cols)))


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
        "out_global_img_root": "./se_cond1_global_images",
        "out_fold_root": "./se_fold_images",
        "fs": 5120,
        "primary_ch": 0,
        "skip_rows": 16,
        "read_cols": 8
    },
    {
        "name": "laboratory",
        "partition_csv": "lab_dataset_partition.csv",
        "raw_root": "/root/SJJ/data",
        "out_global_img_root": "./lab_cond1_global_images",
        "out_fold_root": "./lab_fold_images",
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
            stats = self.channel_stats[ch]
            s_min, s_max = stats["min"], stats["max"]
            if abs(s_max - s_min) < 1e-10:
                norm_sig = np.zeros_like(sig)
            else:
                norm_sig = (sig - s_min) / (s_max - s_min)
            out[ch] = norm_sig
        return out

    def build_time_domain_grayscale(self, sig_1d: np.ndarray) -> Image.Image:
        sig_resample = np.interp(np.linspace(0, len(sig_1d) - 1, IMG_SIZE[0]),
                                 np.arange(len(sig_1d)), sig_1d)
        img_arr = np.tile(sig_resample.reshape(1, -1), (IMG_SIZE[1], 1))
        img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_arr, mode="L")

    def build_log_spectrum_image(self, sig_1d: np.ndarray, fs: int) -> Image.Image:
        N = len(sig_1d)
        yf = rfft(sig_1d)
        amp = np.abs(yf)
        log_amp = np.log10(amp + EPS)
        log_resample = np.interp(np.linspace(0, len(log_amp) - 1, IMG_SIZE[0]),
                                 np.arange(len(log_amp)), log_amp)
        img_arr = np.tile(log_resample.reshape(1, -1), (IMG_SIZE[1], 1))
        vmin, vmax = float(np.min(img_arr)), float(np.max(img_arr))
        if abs(vmax - vmin) > 1e-10:
            img_arr = ((img_arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            img_arr = np.zeros_like(img_arr, dtype=np.uint8)
        return Image.fromarray(img_arr, mode="L")

    def build_cwt_timefreq_image(self, sig_1d: np.ndarray, fs: int) -> Image.Image:
        scales = np.linspace(1, 128, 128)
        coeffs, _ = pywt.cwt(sig_1d, scales, wavelet=WAVELET_NAME, sampling_period=1.0 / fs)
        coeff_abs = np.abs(coeffs)
        fig, ax = plt.subplots(figsize=(IMG_SIZE[0] / 100, IMG_SIZE[1] / 100), dpi=100)
        ax.imshow(coeff_abs, cmap="viridis", aspect="auto")
        ax.axis("off")
        # Remove canvas white margins to guarantee reproducible images
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        img = img.resize(IMG_SIZE)
        plt.close(fig)
        return img


def read_single_segment(csv_path, seg_idx, skip_rows, read_cols):
    df_raw = data_read(csv_path, skip_rows, read_cols)
    n_start = seg_idx * SEG_LENGTH
    n_end = n_start + SEG_LENGTH
    slice_data = df_raw.iloc[n_start:n_end, :].to_numpy()
    ch_dict = {}
    for ch in range(slice_data.shape[1]):
        ch_dict[ch] = slice_data[:, ch].astype(np.float64)
    return ch_dict


def export_global_images(df_subset, raw_root, out_dir, fs, skip_rows, read_cols, primary_ch=0, preprocessor=None):
    os.makedirs(out_dir, exist_ok=True)
    if preprocessor is None:
        preproc = SignalPreprocessor()
        sig_collect = []
        for _, row in df_subset.iterrows():
            fname = row["raw_file"]
            csv_file = os.path.join(raw_root, fname + ".csv")
            sig_dict = read_single_segment(csv_file, int(row["segment_idx"]), skip_rows, read_cols)
            sig_collect.append(sig_dict[primary_ch])
        fit_dict = {primary_ch: np.concatenate(sig_collect)}
        preproc.fit_train_normalize(fit_dict)
    else:
        preproc = preprocessor

    print("  >> Start generating global image library")
    for idx, row in df_subset.iterrows():
        sample_id = int(row["sample_id"])
        fault_label = row["fault_label"]
        fname = row["raw_file"]
        csv_file = os.path.join(raw_root, fname + ".csv")
        sig_raw_dict = read_single_segment(csv_file, int(row["segment_idx"]), skip_rows, read_cols)
        sig_norm_dict = preproc.transform_normalize(sig_raw_dict)
        sig = sig_norm_dict[primary_ch]

        save_sub = os.path.join(out_dir, f"sample_{sample_id:06d}_{fault_label}")
        os.makedirs(save_sub, exist_ok=True)

        img_t = preproc.build_time_domain_grayscale(sig)
        img_f = preproc.build_log_spectrum_image(sig, fs=fs)
        img_tf = preproc.build_cwt_timefreq_image(sig, fs=fs)

        img_t.save(os.path.join(save_sub, "time.png"))
        img_f.save(os.path.join(save_sub, "freq.png"))
        img_tf.save(os.path.join(save_sub, "tf.png"))

        if (idx + 1) % 200 == 0:
            print(f"     Progress {idx+1}/{len(df_subset)}")
    print(f"  >> ✅Global‑image generation finished, output path: {out_dir}")
    return preproc


def copy_global_to_fold_images(partition_csv, global_img_root, out_fold_root):
    """
    Only copy existing files without regenerating images, create fold1_img~fold5_img train‑validation folders
    """
    df = pd.read_csv(partition_csv)
    cond1_df = df[df["fold"] != -1].reset_index(drop=True)
    os.makedirs(out_fold_root, exist_ok=True)

    for fold_num in range(1, 6):
        fold_path = os.path.join(out_fold_root, f"fold{fold_num}_img")
        train_path = os.path.join(fold_path, "train")
        val_path = os.path.join(fold_path, "val")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)

        train_samples = cond1_df[cond1_df["fold"] != fold_num]
        val_samples = cond1_df[cond1_df["fold"] == fold_num]
        print(f"\nFold {fold_num}: Training samples {len(train_samples)} ,Validation samples {len(val_samples)}")

        # Copy training‑set samples
        for _, row in train_samples.iterrows():
            sid = int(row["sample_id"])
            label = row["fault_label"]
            src_folder = os.path.join(global_img_root, f"sample_{sid:06d}_{label}")
            dst_folder = os.path.join(train_path, label)
            if not os.path.exists(src_folder):
                print(f"Warning, missing source sample: {src_folder}")
                continue
            os.makedirs(dst_folder, exist_ok=True)
            for img_name in ["time.png", "freq.png", "tf.png"]:
                src_img = os.path.join(src_folder, img_name)
                dst_img = os.path.join(dst_folder, f"{label}_{img_name}")
                shutil.copy(src_img, dst_img)

        # Copy validation‑set samples
        for _, row in val_samples.iterrows():
            sid = int(row["sample_id"])
            label = row["fault_label"]
            src_folder = os.path.join(global_img_root, f"sample_{sid:06d}_{label}")
            dst_folder = os.path.join(val_path, label)
            if not os.path.exists(src_folder):
                print(f"Warning, missing source sample: {src_folder}")
                continue
            os.makedirs(dst_folder, exist_ok=True)
            for img_name in ["time.png", "freq.png", "tf.png"]:
                src_img = os.path.join(src_folder, img_name)
                dst_img = os.path.join(dst_folder, f"{label}_{img_name}")
                shutil.copy(src_img, dst_img)

    print(f"\n✅All five‑fold image folders have been generated, output directory:{out_fold_root}")


if __name__ == "__main__":
    fix_partition_csv()

    for cfg in DATASET_CONFIGS:
        ds_name = cfg["name"]
        skip = cfg["skip_rows"]
        ncol = cfg["read_cols"]
        print(f"\n{'=' * 70}")
        print(f"===== Processing dataset：{ds_name} | skip_rows={skip} ,read the first {ncol} columns =====")
        print(f"{'=' * 70}")

        df_full = pd.read_csv(cfg["partition_csv"])
        cond1_df = df_full[df_full["fold"] != -1].reset_index(drop=True)
        cond2_df = df_full[df_full["fold"] == -1].reset_index(drop=True)
        print(f"Working‑condition‑1 samples:{len(cond1_df)}, Working‑condition‑2 samples:{len(cond2_df)}")

        # 1. Generate global image library for working‑condition‑1
        final_preproc = export_global_images(
            df_subset=cond1_df,
            raw_root=cfg["raw_root"],
            out_dir=cfg["out_global_img_root"],
            fs=cfg["fs"],
            skip_rows=skip,
            read_cols=ncol,
            primary_ch=cfg["primary_ch"],
            preprocessor=None
        )

        # 2. Copy images from global library to create 5‑fold folders, compatible with legacy training code
        copy_global_to_fold_images(
            partition_csv=cfg["partition_csv"],
            global_img_root=cfg["out_global_img_root"],
            out_fold_root=cfg["out_fold_root"]
        )

        # 3. Generate generalization‑test‑set images for working‑condition‑2 (reuse normalization parameters from condition 1)
        cond2_global_out = cfg["out_global_img_root"] + "_cond2"
        export_global_images(
            df_subset=cond2_df,
            raw_root=cfg["raw_root"],
            out_dir=cond2_global_out,
            fs=cfg["fs"],
            skip_rows=skip,
            read_cols=ncol,
            primary_ch=cfg["primary_ch"],
            preprocessor=final_preproc
        )

    print("\n🎉All offline preprocessing + five‑fold image generation tasks finished!")
