"""var2 final model: degree-10 polynomial + standardisation + Ridge (alpha by 10-fold CV)."""
import argparse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from common import load, save_predictions

DEGREE = 10
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="predictions")
args = ap.parse_args()

X, y, Xte = load(args.data, 2)
alphas = np.logspace(np.log10(0.681) - 0.5, np.log10(0.681) + 0.5, 15)
cv_mse = np.zeros(len(alphas))
for a, b in KFold(10, shuffle=True, random_state=0).split(X):
    pf = PolynomialFeatures(DEGREE, include_bias=False)
    A, B = pf.fit_transform(X[a]), pf.transform(X[b])
    sc = StandardScaler().fit(A)
    A, B = sc.transform(A), sc.transform(B)
    for i, al in enumerate(alphas):
        cv_mse[i] += np.mean((y[b] - Ridge(alpha=al).fit(A, y[a]).predict(B)) ** 2) / 10
best = alphas[cv_mse.argmin()]
pipe = make_pipeline(PolynomialFeatures(DEGREE, include_bias=False), StandardScaler(),
                     Ridge(alpha=best)).fit(X, y)
print("alpha %.3g | 10-fold CV MSE %.4f | train MSE %.4f"
      % (best, cv_mse.min(), np.mean((y - pipe.predict(X)) ** 2)))
save_predictions(pipe.predict(Xte), args.out, 2)
