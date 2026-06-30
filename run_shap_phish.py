import argparse
import subprocess
import sys


DEFAULT_SAMPLE_SEEDS = [42, 123, 2024, 3407]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shap_phish_logit.py multiple times with different sample seeds."
    )
    parser.add_argument(
        "--sample_seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SEEDS,
    )
    args, passthrough_args = parser.parse_known_args()
    return args, passthrough_args


def main():
    args, passthrough_args = parse_args()
    for sample_seed in args.sample_seeds:
        command = [
            sys.executable,
            "shap_phish_logit.py",
            "--sample_seed",
            str(sample_seed),
            *passthrough_args,
        ]
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
