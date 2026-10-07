# Polynomial Regression Assignment (IMT2024051)

Two personalised regression problems solved with polynomial regression only.

| Problem | Features | Final model | 5/10-fold CV MSE | CV R² |
|---|---|---|---|---|
| var1 (turbine) | x1..x6 | degree 5 polynomial + standardisation + **Lasso** | ~0.315 | ~0.97 |
| var2 (thermal map) | x1..x3 | degree 10 polynomial + standardisation + **Ridge** | ~0.252 | ~0.994 |

## Layout
```
data/          put the 4 CSVs here (IMT2024051_{train,test}_var{1,2}.csv)
src/common.py          data loading / saving helpers
src/eda.py             shapes, ranges, clipping at +-1, train-vs-test shift
src/baseline_sweep.py  plain least squares, CV error vs degree (shows over-fitting)
src/var1_model.py      final var1 model -> predictions/IMT2024051_pred_var1.csv
src/var2_model.py      final var2 model -> predictions/IMT2024051_pred_var2.csv
predictions/   submission files
```

## Run
```
pip install -r requirements.txt
cd src
python eda.py --data ../data
python baseline_sweep.py --data ../data
python var1_model.py --data ../data --out ../predictions
python var2_model.py --data ../data --out ../predictions
```

## Method summary
1. Plain polynomial least squares overfits quickly: it peaks at degree 4 (var1) and degree 8 (var2) in CV.
2. Features are standardised and Ridge / Lasso are added, with the penalty chosen by cross-validation.
   This makes higher degrees usable. Lasso was best for var1 and Ridge for var2.
3. Degrees were confirmed with repeated CV over several random splits.
4. var1 test inputs are more clipped at +-1 than the train inputs (about 49% vs 31% of values), so
   degree choice was also checked with a test-like weighted validation. Degree 5 still won.
5. var2 train and test distributions match, so plain CV is a fair guide.
