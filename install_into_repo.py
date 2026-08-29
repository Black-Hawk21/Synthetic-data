"""Copy this package's files into your local `aml-sim/` directory.

Usage:
    python install_into_repo.py /path/to/Synthetic-data/aml-sim

Or just copy the files yourself — this script does nothing clever, it only
saves you from putting them in the wrong place. Nothing in the original repo is
overwritten; if a name collides, the script stops and tells you.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

FILES = ["aml_models.py", "gnn_model.py", "train_models.py", "train_gnn.py",
         "predict.py", "predict_gnn.py", "compare_models.py",
         "requirements-models.txt", "MODELS_README.md"]
MODEL_FILES = ["account_model.joblib", "transaction_model.joblib",
               "gnn_account_model.pt", "gnn_transaction_model.pt"]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    target = Path(sys.argv[1]).expanduser().resolve()
    here = Path(__file__).parent.resolve()

    if not (target / "run.py").exists() or not (target / "amlgen").is_dir():
        print(f"error: {target} does not look like the aml-sim directory "
              f"(no run.py / amlgen/ found).")
        return 1

    clash = [f for f in FILES if (target / f).exists()]
    if clash:
        print("error: these files already exist in the target and would be "
              "overwritten:")
        for f in clash:
            print(f"  {f}")
        print("Move or delete them first, or copy the files manually.")
        return 1

    for f in FILES:
        src = here / f
        if src.exists():
            shutil.copy2(src, target / f)
            print(f"  + {f}")

    (target / "models").mkdir(exist_ok=True)
    for f in MODEL_FILES:
        src = here / "models" / f
        if src.exists():
            shutil.copy2(src, target / "models" / f)
            print(f"  + models/{f}")

    print(f"\nInstalled into {target}")
    print("Next:  cd", target)
    print("       python run.py all          # generate the dataset")
    print("       python predict_gnn.py      # classify with the GNN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
