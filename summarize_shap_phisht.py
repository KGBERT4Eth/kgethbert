import argparse
import csv
import statistics
from pathlib import Path


FIELDS = ["value", "timeStamp", "IO", "gasUsed", "gas"]
DEFAULT_ROOT = Path("Eval_output") / "shap_phish"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize mean_abs_shap for selected fields from the last N "
            "shap_phish_logit output folders."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def get_latest_run_dirs(root, last_n):
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    selected = run_dirs[-last_n:]
    if len(selected) < last_n:
        raise ValueError(f"Only found {len(selected)} folders under {root}, need {last_n}.")
    return selected


def read_mean_abs_shap(csv_path):
    values = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row["field"]
            if field in FIELDS:
                values[field] = float(row["mean_abs_shap"])
    missing = [field for field in FIELDS if field not in values]
    if missing:
        raise ValueError(f"{csv_path} is missing fields: {', '.join(missing)}")
    return values


def summarize(run_dirs):
    field_values = {field: [] for field in FIELDS}
    for run_dir in run_dirs:
        csv_path = run_dir / "global_field_importance.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        values = read_mean_abs_shap(csv_path)
        for field in FIELDS:
            field_values[field].append(values[field])

    rows = []
    for field in FIELDS:
        values = field_values[field]
        rows.append(
            {
                "field": field,
                "n": len(values),
                "mean_abs_shap_mean": statistics.mean(values),
                "mean_abs_shap_sample_std": statistics.stdev(values),
            }
        )
    return rows


def write_summary(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, run_dirs):
    print("Used folders:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    print()
    print("field,n,mean_abs_shap_mean,mean_abs_shap_sample_std")
    for row in rows:
        print(
            f"{row['field']},{row['n']},"
            f"{row['mean_abs_shap_mean']:.12f},"
            f"{row['mean_abs_shap_sample_std']:.12f}"
        )


def main():
    args = parse_args()
    run_dirs = get_latest_run_dirs(args.root, args.last_n)
    rows = summarize(run_dirs)
    print_summary(rows, run_dirs)
    if args.output is not None:
        write_summary(rows, args.output)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
