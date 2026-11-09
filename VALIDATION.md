# Engine validation

Completed 8 November 2026. This is a working local machine-learning simulation, replacing fixed interface examples.

## Evaluation

- Selected model: Support Vector Regression, chosen by grouped development MAE.
- Training/development: 1,600 synthetic cells from 16 batches, four grouped folds.
- Reserved final test: 400 synthetic cells from four separate batches.
- Final-test MAE: 0.021509 mAh.
- Final-test RMSE: 0.027495 mAh.
- Final-test R²: 0.992706 (not an accuracy percentage).
- Final-test MAPE: 1.2696%.
- Mean baseline final-test MAE: 0.273685 mAh.
- Full initialization/training: 62.429 seconds in the recorded run.
- Process-lifetime peak working set: 206.23 MB; includes imports and training.
- Model run: run-f7d2423dde724f7c.

The generator, parameters, noise assumptions, and evaluation partitions were fixed before final evaluation. Reusing a saved pipeline reuses its report rather than rerunning final evaluation. Scores apply only to the declared synthetic domain.

## Verification

All 11 automated tests passed. They cover the exact 1.863 mAh analytical reference, reproducibility and physical bounds, input and dataset validation, batch separation, undefined metrics, candidate selection and reserved evaluation, pipeline persistence and input sensitivity, artifact and schema checks, SQLite traceability, and local HTTP request boundaries.

The complete browser journey was exercised: first initialization trains the engine behind the waiting state; a page refresh resumes the same job; the computed result appears automatically; detailed metrics and three graphs load; full evaluation JSON exports; and saved results appear in database-backed history. Changing coating mass from 0.0150 g to 0.0110 g changed the prediction from approximately 1.874 mAh to 1.378 mAh without retraining.

Both reviewed themes remain available. This implementation uses the existing designed frontend with a local Python HTTP service instead of Streamlit. Compatible CSV validation is available as an engine utility, but a file-import interface is not part of this simplified initial release. Full cross-browser and assistive-technology verification remains future work.

## Start the installed app

Double-click Start Casimir.cmd in the project's Simulation App folder. Its isolated environment and offline dependency wheels are installed locally. Keep the app window open while using it.

The source ZIP contains source, pinned requirements, and the trained model/data artifacts; it excludes the machine-specific virtual environment and the large platform-specific wheel cache. Installing that ZIP on another computer requires the pinned libraries, or a compatible wheel cache.
