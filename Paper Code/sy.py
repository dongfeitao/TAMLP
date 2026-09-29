import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    prefix: str
    cond1: tuple
    cond2: tuple
    subfiles: int
    segments: int
    fold_counts: tuple
    fs: int
    channels: int
    skip_rows: int
    window: int = 1024
    block_points: int = 0

    @property
    def points_per_block(self):
        return self.block_points or self.segments * self.window

    @property
    def required_points(self):
        return self.subfiles * self.points_per_block


CONFIGS = {
    "southeast": DatasetConfig(
        "southeast", "se",
        ("comb_20_0", "health_20_0", "outer_20_0", "ball_20_0", "inner_20_0"),
        ("comb_30_2", "health_30_2", "outer_30_2", "ball_30_2", "inner_30_2"),
        16, 63, (4, 3, 3, 3, 3), 5120, 8, 16, block_points=65535),
    "laboratory": DatasetConfig(
        "laboratory", "lab",
        ("Comb_1200_0", "Health_1200_0", "Outer_1200_0", "Ball_1200_0", "Inner_1200_0"),
        ("Comb_1800_50", "Health_1800_50", "Outer_1800_50", "Ball_1800_50", "Inner_1800_50"),
        10, 100, (2, 2, 2, 2, 2), 10240, 4, 5),
}
DEFAULT_GAPS = ("0", "0.2", "1.0", "2.0")
BASE_FIELDS = ["sample_id", "raw_file", "subfile_id", "segment_idx",
               "fault_label", "group_id", "fold", "is_train", "condition",
               "start_point", "end_point", "block_start_point", "block_end_point"]
ROLE_FIELDS = BASE_FIELDS + ["cv_fold", "gap_seconds", "gap_points", "role"]


def gap_decimal(value):
    gap = Decimal(str(value))
    if not gap.is_finite() or gap < 0:
        raise ValueError("Gap must be finite and nonnegative")
    return gap


def gap_tag(value):
    text = format(gap_decimal(value).normalize(), "f")
    return "gap_" + text.replace(".", "p") + "s"


def gap_points(value, fs):
    return int((gap_decimal(value) * fs).to_integral_value(rounding=ROUND_CEILING))


def partition_name(cfg):
    stem = "southeast" if cfg.name == "southeast" else "lab"
    return stem + "_dataset_partition.csv"


def load_dataset_configs(indices, dataset="all"):
    """Load the recorded header settings, never silently fall back to defaults."""
    path = Path(indices) / "experiment_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; regenerate indices with sy.py")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2:
        raise ValueError(f"Old index version in {path}; regenerate using the 5040-sample sy.py")
    entries = {entry["name"]: entry for entry in metadata["datasets"]}
    names = list(CONFIGS) if dataset == "all" else [dataset]
    configs = []
    for name in names:
        if name not in entries:
            raise ValueError(f"Dataset {name} was not generated in {path}")
        entry = entries[name]
        baseline = asdict(CONFIGS[name])
        for key, value in baseline.items():
            if key == "skip_rows":
                continue
            actual = entry.get(key)
            if isinstance(value, tuple):
                actual = tuple(actual) if isinstance(actual, list) else actual
            if actual != value:
                raise ValueError(f"Configuration mismatch for {name}.{key}; use matching sy.py")
        skip = entry.get("skip_rows")
        if type(skip) is not int or skip < 0:
            raise ValueError(f"Invalid skip_rows in {path}: {skip}")
        configs.append(replace(CONFIGS[name], skip_rows=skip))
    return configs


def make_samples(cfg):
    if sum(cfg.fold_counts) != cfg.subfiles or any(n <= 0 for n in cfg.fold_counts):
        raise ValueError("Invalid temporal block allocation")
    if cfg.segments != cfg.points_per_block // cfg.window:
        raise ValueError("segments must equal the number of complete windows per raw block")
    fold_of_group = [fold for fold, n in enumerate(cfg.fold_counts, 1) for _ in range(n)]
    rows = []
    for condition, files in ((1, cfg.cond1), (2, cfg.cond2)):
        for label, raw in enumerate(files):
            for group in range(cfg.subfiles):
                for segment in range(cfg.segments):
                    block_start = group * cfg.points_per_block
                    start = block_start + segment * cfg.window
                    rows.append(dict(
                        sample_id=len(rows), raw_file=raw, subfile_id=group,
                        segment_idx=segment, fault_label=f"D{label}",
                        group_id=f"{raw}_{group}",
                        fold=fold_of_group[group] if condition == 1 else -1,
                        # Legacy field: condition-1 eligibility, NOT a per-fold training mask.
                        is_train=condition == 1, condition=condition,
                        start_point=start, end_point=start + cfg.window,
                        block_start_point=block_start, block_end_point=block_start+cfg.points_per_block))
    return rows


def fold_intervals(rows, cv_fold):
    """Hull of actual validation windows in consecutive raw blocks.

    Unused tails may separate windows. No training block lies inside this hull.
    Gap distances refer to actual window supports, not compressed sample IDs.
    """
    grouped = {}
    for row in rows:
        if row["condition"] == 1 and row["fold"] == cv_fold:
            grouped.setdefault(row["raw_file"], []).append(row)
    result = {}
    for raw, samples in grouped.items():
        samples = sorted(samples, key=lambda r: r["start_point"])
        groups = sorted({r["subfile_id"] for r in samples})
        if groups != list(range(groups[0], groups[-1]+1)):
            raise ValueError("Expected consecutive validation blocks per recording")
        if any(a["end_point"] > b["start_point"] for a, b in zip(samples, samples[1:])):
            raise ValueError("Overlapping validation windows")
        result[raw] = (samples[0]["start_point"], samples[-1]["end_point"])
    return result


def split_fold(rows, cfg, cv_fold, gap):
    """Validation unchanged; exclude training windows intersecting its dilation.

    A window ending exactly gap_points before validation is retained. No gap
    is imposed between different files. Condition 2 never enters CV manifests.
    """
    padding = gap_points(gap, cfg.fs)
    intervals = fold_intervals(rows, cv_fold)
    manifest = []
    for row in rows:
        if row["condition"] != 1:
            continue
        left, right = intervals[row["raw_file"]]
        if row["fold"] == cv_fold:
            role = "validation"
        elif row["start_point"] < right + padding and row["end_point"] > left - padding:
            role = "excluded"
        else:
            role = "train"
        manifest.append(dict(row, cv_fold=cv_fold,
                             gap_seconds=str(gap_decimal(gap)), gap_points=padding, role=role))
    expected = {f"D{i}" for i in range(len(cfg.cond1))}
    for role in ("train", "validation"):
        if {r["fault_label"] for r in manifest if r["role"] == role} != expected:
            raise ValueError(f"Fold {cv_fold}: {role} is missing classes; gap may be too large")
    return manifest


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(cfg, output, gaps):
    rows = make_samples(cfg)
    write_csv(output / partition_name(cfg), rows, BASE_FIELDS)
    tails = []
    for condition, files in ((1, cfg.cond1), (2, cfg.cond2)):
        for raw in files:
            for group in range(cfg.subfiles):
                start = group * cfg.points_per_block + cfg.segments * cfg.window
                end = (group+1) * cfg.points_per_block
                if end > start:
                    tails.append(dict(raw_file=raw, condition=condition, subfile_id=group,
                                      start_point=start, end_point=end, discarded_points=end-start))
    write_csv(output / cfg.name / "discarded_tails.csv", tails,
              ["raw_file", "condition", "subfile_id", "start_point", "end_point", "discarded_points"])
    write_csv(output / cfg.name / "condition2_test.csv",
              [r for r in rows if r["condition"] == 2], BASE_FIELDS)
    summaries = []
    for gap in gaps:
        for fold in range(1, len(cfg.fold_counts) + 1):
            manifest = split_fold(rows, cfg, fold, gap)
            write_csv(output / cfg.name / gap_tag(gap) / f"fold{fold}.csv", manifest, ROLE_FIELDS)
            counts = Counter(r["role"] for r in manifest)
            for label in ["ALL"] + [f"D{i}" for i in range(len(cfg.cond1))]:
                selected = manifest if label == "ALL" else [r for r in manifest if r["fault_label"] == label]
                n = Counter(r["role"] for r in selected)
                summaries.append(dict(dataset=cfg.name, gap_seconds=str(gap_decimal(gap)),
                                      fold=fold, fault_label=label, train=n["train"],
                                      validation=n["validation"], excluded=n["excluded"]))
            print(f"{cfg.name} {gap_tag(gap)} fold{fold}: {dict(counts)}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("partitions_5040"))
    parser.add_argument("--dataset", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--gaps", nargs="+", default=list(DEFAULT_GAPS))
    parser.add_argument("--se-skip-rows", type=int, default=None,
                        help="Southeast header rows: 0 for headerless files; original default is 16")
    parser.add_argument("--lab-skip-rows", type=int, default=None,
                        help="Laboratory header rows: 0 for headerless files; original default is 5")
    args = parser.parse_args()
    if any(value is not None and value < 0 for value in (args.se_skip_rows, args.lab_skip_rows)):
        parser.error("Header row counts must be nonnegative")
    gaps = list(dict.fromkeys(gap_decimal(g) for g in args.gaps))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output to preserve prior runs")
    args.output.mkdir(parents=True, exist_ok=True)
    configs = list(CONFIGS.values()) if args.dataset == "all" else [CONFIGS[args.dataset]]
    configs = [replace(c, skip_rows=(args.se_skip_rows if c.name == "southeast" else args.lab_skip_rows))
               if (args.se_skip_rows if c.name == "southeast" else args.lab_skip_rows) is not None
               else c for c in configs]
    summary = []
    for cfg in configs:
        print(f"{cfg.name}: skip_rows={cfg.skip_rows}; confirm this matches your raw files")
        summary.extend(export_dataset(cfg, args.output, gaps))
    write_csv(args.output / "split_summary.csv", summary,
              ["dataset", "gap_seconds", "fold", "fault_label", "train", "validation", "excluded"])
    metadata = dict(schema_version=2, index_convention="zero-based half-open; after header",
                    gap_rule="training-side exclusion; validation unchanged",
                    tail_rule="discard incomplete tail within each equal-length raw block; never pad",
                    datasets=[asdict(c) for c in configs], gaps=[str(g) for g in gaps])
    (args.output / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Manifests saved to {args.output.resolve()}; no model metrics have been generated.")


if __name__ == "__main__":
    main()
