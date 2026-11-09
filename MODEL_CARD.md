# Simulation model specification

## Intended use

Educational demonstration of initial cell discharge-capacity regression from synthetic manufacturing metadata. All records and outputs carry a synthetic-simulation label. No experimentally calibrated battery chemistry is claimed.

## Data and equation

Default configuration: 2,000 cells in 20 batches of 100 records; master seed 42. Inputs are independently uniform over:

| Feature | Meaning | Range |
|---|---|---|
| m | Dry cathode coating mass, g | 0.010–0.020 |
| f | Active material fraction | 0.85–0.95 |
| p | Electrode porosity fraction | 0.25–0.45 |
| c | Compression ratio | 0.10–0.30 |
| e | Normalized electrolyte fill | 0.80–1.20 |

`Q0 = m × f × 150`

`zp = (p − 0.35)/0.10; zc = (c − 0.20)/0.10; ze = (e − 1.00)/0.20`

`u = 0.92 × (1 − 0.12zp²) × (1 − 0.08zc²) × (1 − 0.10ze²)`

`Qsim = Q0 × clip(u + b + noise, 0.05, 1.00)`

The batch effect is Normal(0, 0.015) once per batch; record noise is Normal(0, 0.010). The noise-free reference is `Q0 × u`. At the default input centre this equals 1.863 mAh. Identifiers, targets, nominal capacity, batch effects, and noise never enter the predictor matrix.

## Learning and evaluation

Four complete batches are reserved using a separate deterministic split seed. Sixteen batches supply four-fold GroupKFold development evaluation. Every algorithm sees the same partitions. SVR scaling is fitted within each training fold through a pipeline.

- Mean baseline and Linear Regression: standard formulations.
- Random Forest: 100 trees; depth 5 or unlimited; minimum leaf sizes 1 or 5.
- RBF SVR: C 1 or 10; gamma `scale` or 0.1; epsilon 0.01 mAh. The solver's numerical tolerance remains its default; epsilon is the capacity error-tube width.
- Gradient Boosting: 100 estimators; learning rate 0.05 or 0.10; depth 2 or 3.

Fourteen parameter candidates are compared on development data. Lowest mean validation MAE selects the pipeline; numerical ties prefer simpler approaches. Final test targets do not affect selection. The selected pipeline is fitted to all 1,600 development records and evaluated once on 400 reserved records. The mean baseline also has a reserved-test reference score. Reusing the stored report does not repeat evaluation.

Development feature importance is exploratory permutation importance, not a causal explanation of a submitted scenario. Sensitivity scores use fresh seed-314159 records at noise standard deviations 0, 0.01, and 0.03 with the selected model fixed. They are reported separately from the reserved final test. No uncertainty interval is currently estimated.

## Traceability and limits

Configuration hashes, feature order, dependency versions, selected parameters, fold manifests, checksums, complete test predictions, training time, and process peak working set are retained. Memory is a process-lifetime peak including imports and training, not isolated fit memory.

The equation's assumed optima and independent input distributions are illustrative. Models learn that synthetic relationship; strong scores do not establish accuracy on real batteries. Use measured manufacturing records and an independent physical validation protocol before practical deployment.

Implementation references: [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), and [preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
