"""var1 final model: degree-5 polynomial + standardisation + Lasso (alpha by 10-fold CV)."""
import argparse
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from common import load, save_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="predictions")
args = ap.parse_args()

X, y, Xte = load(args.data, 1)
pipe = make_pipeline(
    PolynomialFeatures(5, include_bias=False),
    StandardScaler(),
    LassoCV(cv=KFold(10, shuffle=True, random_state=0), n_alphas=60, eps=1e-4,
            max_iter=50000, tol=1e-4, n_jobs=-1),
).fit(X, y)
lasso = pipe[-1]
print("alpha %.5f | non-zero terms %d | 10-fold CV MSE %.4f | train MSE %.4f"
      % (lasso.alpha_, (lasso.coef_ != 0).sum(), lasso.mse_path_.mean(1).min(),
         np.mean((y - pipe.predict(X)) ** 2)))
save_predictions(pipe.predict(Xte), args.out, 1)
