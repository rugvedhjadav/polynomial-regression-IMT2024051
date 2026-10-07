"""Shapes, ranges, clipping, and train-vs-test distribution check."""
import argparse
import numpy as np
from common import load

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
args = ap.parse_args()

for var in (1, 2):
    Xtr, ytr, Xte = load(args.data, var)
    print(f"\n=== var{var}: train {Xtr.shape}, test {Xte.shape} ===")
    print("y mean %.3f std %.3f min %.2f max %.2f" % (ytr.mean(), ytr.std(), ytr.min(), ytr.max()))
    print("feature std   train", Xtr.std(0).round(3), "test", Xte.std(0).round(3))
    print("share at +-1  train", (np.abs(Xtr) == 1).mean(0).round(3), "test", (np.abs(Xte) == 1).mean(0).round(3))
    print("avg clipped features/row  train %.2f  test %.2f"
          % ((np.abs(Xtr) == 1).sum(1).mean(), (np.abs(Xte) == 1).sum(1).mean()))
