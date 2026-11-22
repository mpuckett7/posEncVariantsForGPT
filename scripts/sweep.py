"""Launch every training run: 4 variants x 3 seeds (AI-scaffolded).

Usage (from the repo root, on the GPU desktop):
    python scripts/sweep.py                       # all 12 runs, sequentially
    python scripts/sweep.py --dry-run             # just print the commands
    python scripts/sweep.py --variants rope nope --seeds 0
    python scripts/sweep.py --max_iters 200       # unknown flags are forwarded to src.train
Finished runs (ckpt.pt exists) are skipped, so an interrupted sweep can simply be re-run.
Each run's console output goes to runs/<name>/stdout.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                  # allow `from src...` when run as a script

from src.config import VARIANTS  # noqa: E402  (import after sys.path tweak)


def main() -> None:
    # allow_abbrev=False: otherwise a forwarded flag like `--seed` would be swallowed as an
    # abbreviation of `--seeds` instead of being passed through to src.train.
    parser = argparse.ArgumentParser(description="Run the full positional-encoding sweep",
                                     allow_abbrev=False)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--force", action="store_true", help="re-run even if ckpt.pt exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    args, extra = parser.parse_known_args()    # `extra` is forwarded to src.train unchanged

    failures = []
    # Seeds are the OUTER loop so a partial sweep still has every variant for the early seeds.
    for seed in args.seeds:
        for variant in args.variants:
            name = f"{variant}_s{seed}"
            run_dir = ROOT / args.out_dir / name
            if (run_dir / "ckpt.pt").exists() and not args.force:
                print(f"[skip] {name}: ckpt.pt already exists")
                continue
            cmd = [sys.executable, "-m", "src.train",
                   "--variant", variant, "--seed", str(seed), "--out_dir", args.out_dir, *extra]
            print(f"[run ] {name}: {' '.join(cmd)}")
            if args.dry_run:
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            with open(run_dir / "stdout.log", "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            minutes = (time.time() - started) / 60
            if proc.returncode != 0:
                print(f"[FAIL] {name}: exit code {proc.returncode} after {minutes:.1f} min "
                      f"(see {run_dir / 'stdout.log'})")
                failures.append(name)
            else:
                print(f"[done] {name}: {minutes:.1f} min")

    if failures:
        print(f"\n{len(failures)} run(s) failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
