"""Plain least-squares polynomial regression: 5-fold CV error vs degree."""
import argparse
from math import comb
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from common import load

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
args = ap.parse_args()

for var, maxdeg in ((1, 7), (2, 15)):
    X, y, _ = load(args.data, var)
    n = X.shape[1]
    print(f"\n=== var{var} ===\ndeg  terms  trainMSE   valMSE    valR2")
    for d in range(1, maxdeg + 1):
        tr, va, r2 = [], [], []
        for a, b in KFold(5, shuffle=True, random_state=42).split(X):
            pf = PolynomialFeatures(d, include_bias=False)
            A, B = pf.fit_transform(X[a]), pf.transform(X[b])
            m = LinearRegression().fit(A, y[a])
            tr.append(mean_squared_error(y[a], m.predict(A)))
            va.append(mean_squared_error(y[b], m.predict(B)))
            r2.append(r2_score(y[b], m.predict(B)))
        print(f"{d:>3} {comb(n + d, d):>6} {np.mean(tr):9.4f} {np.mean(va):10.4f} {np.mean(r2):9.4f}")
