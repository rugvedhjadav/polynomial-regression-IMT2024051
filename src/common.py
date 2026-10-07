"""Shared helpers: data loading and saving predictions."""
from pathlib import Path
import pandas as pd

ROLL = "IMT2024051"


def load(data_dir, var):
    """Return X_train, y_train, X_test (numpy) for problem `var` (1 or 2)."""
    d = Path(data_dir)
    tr = pd.read_csv(d / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(d / f"{ROLL}_test_var{var}.csv")
    cols = [c for c in tr.columns if c != "y"]
    return tr[cols].values, tr["y"].values, te[cols].values


def save_predictions(pred, out_dir, var):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ROLL}_pred_var{var}.csv"
    pd.DataFrame({"y": pred}).to_csv(path, index=False)
    print("wrote", path, "| rows:", len(pred))
