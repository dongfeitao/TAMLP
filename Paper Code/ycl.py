import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sy import (CONFIGS, DEFAULT_GAPS, ROLE_FIELDS, gap_decimal, gap_tag,
                 make_samples, split_fold, write_csv, partition_name, load_dataset_configs)

IMG_SIZE = (128, 128)
EPS = 1e-12
LABEL_FOLDERS = {"D0": "C", "D1": "H", "D2": "O", "D3": "B", "D4": "I"}
INT_FIELDS = {"sample_id", "subfile_id", "segment_idx", "fold", "condition",
              "start_point", "end_point", "block_start_point", "block_end_point", "cv_fold", "gap_points"}


def read_manifest(path, cfg, fold, gap):
    """Reject missing, duplicated, edited, or incompatible manifest entries."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROLE_FIELDS:
            raise ValueError(f"Unexpected manifest schema: {path}; regenerate using sy.py")
        rows = []
        for row in reader:
            for field in INT_FIELDS:
                row[field] = int(row[field])
            if row["is_train"] not in ("True", "False"):
                raise ValueError("Invalid is_train value")
            row["is_train"] = row["is_train"] == "True"
            row["gap_seconds"] = str(gap_decimal(row["gap_seconds"]))
            rows.append(row)

    expected = split_fold(make_samples(cfg), cfg, fold, gap)
    # 1 and 1.0 describe the same gap.
    for actual, target in zip(rows, expected):
        if gap_decimal(actual["gap_seconds"]) != gap_decimal(target["gap_seconds"]):
            raise ValueError(f"Wrong gap in {path}")
        actual["gap_seconds"] = target["gap_seconds"]
    if rows != expected:
        raise ValueError(f"Manifest differs from the configured temporal split: {path}")
    return rows


class RawReader:
    """Cache at most two recordings and preserve absolute raw-point positions.

    Headers are skipped before parsing. Every remaining row must be numeric;
    blank/invalid rows raise rather than silently shifting temporal indices.
    """
    def __init__(self, cfg, root, separator=","):
        self.cfg, self.root, self.separator = cfg, Path(root), separator
        self.cache = OrderedDict()

    def _path(self, raw):
        canonical = self.root / (raw + ".csv")
        if canonical.exists():
            return canonical
        # Original repository also used C-20-0.csv / C-1200-0%.csv naming.
        condition = 1 if raw in self.cfg.cond1 else 2
        files = self.cfg.cond1 if condition == 1 else self.cfg.cond2
        label = files.index(raw)
        prefix = ("C", "H", "O", "B", "I")[label]
        suffix = (("20-0", "30-2") if self.cfg.name == "southeast"
                  else ("1200-0%", "1800-50%"))[condition - 1]
        alias = self.root / f"{prefix}-{suffix}.csv"
        if not alias.exists():
            raise FileNotFoundError(f"Missing raw file: {canonical} (also tried {alias})")
        return alias

    def recording(self, raw):
        if raw in self.cache:
            self.cache.move_to_end(raw)
            return self.cache[raw]

        cfg = self.cfg
        path = self._path(raw)
        print(
            f"文件={path}，分隔符优先={self.separator!r}，"
            f"跳过行数={cfg.skip_rows}，目标通道数={cfg.channels}",
            flush=True
        )

        # ========== 核心修改：多分隔符自动适配读取 ==========
        # 按优先级尝试分隔符：用户指定 → 逗号 → 制表符 → 任意空白符
        sep_candidates = [self.separator, ',', '\t', r'\s+']
        # 去重，保持顺序
        seen = set()
        sep_list = []
        for s in sep_candidates:
            if s not in seen:
                seen.add(s)
                sep_list.append(s)

        values = None
        last_error = None

        for sep in sep_list:
            try:
                # 先尝试直接指定usecols读取，效率最高
                frame = pd.read_csv(
                    path,
                    sep=sep,
                    header=None,
                    skiprows=cfg.skip_rows,
                    skip_blank_lines=False,
                    usecols=list(range(cfg.channels)),
                    dtype=np.float64
                )
                values = frame.to_numpy()
                break
            except Exception as e:
                last_error = e
                # 指定usecols失败，尝试全量读取后取前N列
                try:
                    frame_full = pd.read_csv(
                        path,
                        sep=sep,
                        header=None,
                        skiprows=cfg.skip_rows,
                        skip_blank_lines=False,
                        dtype=np.float64
                    )
                    if frame_full.shape[1] >= cfg.channels:
                        values = frame_full.iloc[:, :cfg.channels].to_numpy()
                        break
                except Exception:
                    continue

        # 兜底方案：整行读取为字符串再正则分割
        if values is None:
            try:
                df_raw = pd.read_csv(path, header=None, skiprows=cfg.skip_rows, dtype=str)
                split_data = df_raw.iloc[:, 0].str.split(r'[,\t\s]+', expand=True)
                if split_data.shape[1] >= cfg.channels:
                    values = split_data.iloc[:, :cfg.channels].astype(np.float64).to_numpy()
            except Exception as e:
                last_error = e

        if values is None:
            raise ValueError(
                f"{path}: 所有分隔符尝试均失败，无法读取{cfg.channels}列数据。"
                f"最后错误: {last_error}"
            )
        # ==================================================

        needed = cfg.required_points
        if values.shape[1] != cfg.channels or len(values) < needed:
            shortage = max(0, needed - len(values))
            raise ValueError(
                f"{path}: expected >= {needed} rows and {cfg.channels} channels; "
                f"got {values.shape} after skip_rows={cfg.skip_rows}; short by {shortage} rows."
                " Confirmed headers: 16 rows for Southeast, 5 for laboratory. "
                "Verify file integrity and config; no padding or automatic header changes are applied."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: nonfinite or blank rows; fix the source without silently dropping rows")
        if len(values) > needed:
            print(f"NOTE {raw}: using first {needed} signal rows; {len(values)-needed} trailing rows unused")
            values = values[:needed]

        self.cache[raw] = values
        if len(self.cache) > 2:
            self.cache.popitem(last=False)
        return values

    def segment(self, row):
        if not (row["block_start_point"] <= row["start_point"] < row["end_point"] <= row["block_end_point"]):
            raise ValueError(f"Window crosses a raw-block boundary: {row}")
        data = self.recording(row["raw_file"])[row["start_point"]:row["end_point"]]
        if len(data) != self.cfg.window:
            raise ValueError(f"Invalid segment bounds: {row}")
        return data


class SignalPreprocessor:
    def fit(self, rows, reader):
        if not rows:
            raise ValueError("Empty normalization training set")
        self.minimum = np.full(reader.cfg.channels, np.inf)
        self.maximum = np.full(reader.cfg.channels, -np.inf)
        for row in rows:
            data = reader.segment(row)
            self.minimum = np.minimum(self.minimum, data.min(axis=0))
            self.maximum = np.maximum(self.maximum, data.max(axis=0))
        return self

    def normalize(self, data):
        span = self.maximum - self.minimum
        result = (data - self.minimum) / np.where(np.abs(span) < 1e-10, 1.0, span)
        result[:, np.abs(span) < 1e-10] = 0.0
        return result

    def save_stats(self, path):
        Path(path).write_text(json.dumps({
            "method": "training_channel_minmax",
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist()
        }, indent=2), encoding="utf-8")

    @staticmethod
    def time_image(signal):
        line = np.interp(np.linspace(0, len(signal)-1, 128), np.arange(len(signal)), signal)
        return Image.fromarray((np.tile(line, (128, 1)) * 255).clip(0, 255).astype(np.uint8))

    @staticmethod
    def frequency_image(signal):
        log_amp = np.log10(np.abs(rfft(signal)) + EPS)
        line = np.interp(np.linspace(0, len(log_amp)-1, 128), np.arange(len(log_amp)), log_amp)
        array = np.tile(line, (128, 1))
        low, high = float(array.min()), float(array.max())
        if abs(high-low) > 1e-10:
            array = ((array-low)/(high-low)*255).astype(np.uint8)
        else:
            array = np.zeros_like(array, dtype=np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def time_frequency_image(signal, fs):
        coeffs, _ = pywt.cwt(signal, np.linspace(1, 128, 128), "morl", sampling_period=1.0/fs)
        fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
        try:
            ax.imshow(np.abs(coeffs), cmap="viridis", aspect="auto")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()
            return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).resize(IMG_SIZE)
        finally:
            plt.close(fig)


def save_images(rows, reader, preprocessor, destination):
    destination = Path(destination)
    for i, row in enumerate(rows, 1):
        signal = preprocessor.normalize(reader.segment(row))
        directory = destination / LABEL_FOLDERS[row["fault_label"]]
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"sample_{row['sample_id']:06d}"
        for ch in range(reader.cfg.channels):
            preprocessor.time_image(signal[:, ch]).save(directory / f"{stem}_time_ch{ch}.png")
            preprocessor.frequency_image(signal[:, ch]).save(directory / f"{stem}_freq_ch{ch}.png")
        preprocessor.time_frequency_image(signal[:, 0], reader.cfg.fs).save(directory / f"{stem}_tf.png")
        if i % 500 == 0:
            print(f"{destination}: {i}/{len(rows)}", flush=True)


def generate_fold_images(cfg, reader, manifest_root, output, gaps):
    for gap in gaps:
        root = output / f"{cfg.prefix}_fold_images" / gap_tag(gap)
        # Validate all five manifests before generating this experiment's images.
        manifests = [read_manifest(
            manifest_root / cfg.name / gap_tag(gap) / f"fold{f}.csv",
            cfg, f, gap
        ) for f in range(1, 6)]
        for fold, manifest in enumerate(manifests, 1):
            train = [r for r in manifest if r["role"] == "train"]
            val = [r for r in manifest if r["role"] == "validation"]
            folder = root / f"fold{fold}_img"
            folder.mkdir(parents=True, exist_ok=False)
            print(f"{cfg.name} {gap_tag(gap)} fold{fold}: train={len(train)}, validation={len(val)}", flush=True)
            # Fit AFTER exclusion; validation and excluded samples cannot affect stats.
            prep = SignalPreprocessor().fit(train, reader)
            prep.save_stats(folder / "normalization.json")
            write_csv(folder / "manifest.csv", manifest, ROLE_FIELDS)
            save_images(train, reader, prep, folder / "train")
            save_images(val, reader, prep, folder / "val")
            (folder / "COMPLETE.json").write_text(
                json.dumps({"train": len(train), "validation": len(val)}),
                encoding="utf-8"
            )


def generate_full_images(cfg, reader, output):
    # Final training uses ALL condition-1 samples, not a globally purged subset.
    rows = make_samples(cfg)
    train = [r for r in rows if r["condition"] == 1]
    test = [r for r in rows if r["condition"] == 2]
    prep = SignalPreprocessor().fit(train, reader)
    prep.save_stats(output / f"{cfg.prefix}_full_normalization.json")
    save_images(train, reader, prep, output / f"{cfg.prefix}_cond1_full_images")
    save_images(test, reader, prep, output / f"{cfg.prefix}_cond2_images_fixed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, default=Path("partitions_5040"))
    parser.add_argument("--output", type=Path, default=Path("images_5040"))
    parser.add_argument("--dataset", choices=["all", *CONFIGS], default="southeast")
    parser.add_argument("--gaps", nargs="+", default=list(DEFAULT_GAPS))
    parser.add_argument("--mode", choices=["cv", "full", "all"], default="all")
    parser.add_argument("--se-raw-root", type=Path, default=Path("/root/SJJ/data"))
    parser.add_argument("--lab-raw-root", type=Path, default=Path("/root/SJJ/data"))
    parser.add_argument("--sep", choices=["tab", "comma", "whitespace"], default="comma",
                        help="优先使用的分隔符，失败后会自动尝试其他分隔符")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate raw files with recorded header settings; generate no images")
    args = parser.parse_args()

    gaps = list(dict.fromkeys(gap_decimal(g) for g in args.gaps))
    if not args.check_only and args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output directory is not empty. Choose a NEW --output; old images are never reused.")

    configs = load_dataset_configs(args.indices, args.dataset)
    separator = {"tab": "\t", "comma": ",", "whitespace": r"\s+"}[args.sep]

    if args.check_only:
        for cfg in configs:
            root = args.se_raw_root if cfg.name == "southeast" else args.lab_raw_root
            reader = RawReader(cfg, root, separator)
            files = cfg.cond1 if args.mode == "cv" else cfg.cond1 + cfg.cond2
            for raw in files:
                values = reader.recording(raw)
                print(f"OK {raw}: shape={values.shape}, skip_rows={cfg.skip_rows}")
        print("Raw-file checks passed. No images were generated.")
        return

    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Choose a NEW --output; old images are never reused.")
    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "run_config.json").write_text(json.dumps({
        "datasets": [c.name for c in configs],
        "gaps": [str(g) for g in gaps],
        "mode": args.mode,
        "indices": str(args.indices.resolve()),
        "separator": args.sep,
        "selected_columns_zero_based": {c.name: list(range(c.channels)) for c in configs},
        "skip_rows": {c.name: c.skip_rows for c in configs},
        "schema_version": 2,
        "block_points": {c.name: c.points_per_block for c in configs},
        "segments_per_block": {c.name: c.segments for c in configs},
        "normalization": "retained-training-only channel minmax",
        "status": "started"
    }, indent=2), encoding="utf-8")

    for cfg in configs:
        root = args.se_raw_root if cfg.name == "southeast" else args.lab_raw_root
        print(f"{cfg.name}: using skip_rows={cfg.skip_rows} from experiment_config.json", flush=True)
        reader = RawReader(cfg, root, separator)
        if args.mode in ("cv", "all"):
            generate_fold_images(cfg, reader, args.indices, args.output, gaps)
        if args.mode in ("full", "all"):
            generate_full_images(cfg, reader, args.output)
        print(f"Training index_csv: {args.indices / partition_name(cfg)}")
        for gap in gaps:
            print(f"Training image_root ({gap}s): {args.output / (cfg.prefix + '_fold_images') / gap_tag(gap)}")

    (args.output / "COMPLETE.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
