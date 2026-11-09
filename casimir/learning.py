"""Grouped model selection with a reserved final test partition."""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import ctypes
import json
from pathlib import Path
import platform
import re
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .simulation import Config, FEATURES, LABEL, generate, validate_dataset


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def peak_memory_mb():
    """Process-lifetime peak working set on Windows; not incremental fit memory."""
    if platform.system() != 'Windows':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
                    *[(name,ctypes.c_size_t) for name in [
                        'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                        'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                        'PagefileUsage','PeakPagefileUsage']]]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    current=ctypes.windll.kernel32.GetCurrentProcess
    current.restype=ctypes.c_void_p
    get_info=ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if get_info(current(),ctypes.byref(counters),counters.cb):
        return round(counters.PeakWorkingSetSize/1024**2,2)
    return None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def scores(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    return {'mae': float(mean_absolute_error(y, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction)) if len(y)>1 and np.ptp(y)>0 else None,
            'mape': float(np.mean(np.abs((y-prediction)/y))*100) if (y != 0).all() else None}


def partitions(frame, seed=42):
    groups = np.array(sorted(frame.batch_id.unique()))
    if len(groups) != 20:
        raise ValueError('The experiment requires exactly 20 complete batches.')
    test_groups = sorted(np.random.default_rng(seed+1).choice(groups, 4, replace=False).tolist())
    test_mask = frame.batch_id.isin(test_groups).to_numpy()
    development = np.flatnonzero(~test_mask)
    final_test = np.flatnonzero(test_mask)
    folds = list(GroupKFold(n_splits=4).split(development, groups=frame.iloc[development].batch_id))
    manifest = {'development_batches': sorted(set(groups)-set(test_groups)),
                'final_test_batches': test_groups, 'development_records': len(development),
                'final_test_records': len(final_test), 'folds': []}
    for train, validation in folds:
        train_batches = sorted(frame.iloc[development[train]].batch_id.unique().tolist())
        val_batches = sorted(frame.iloc[development[validation]].batch_id.unique().tolist())
        if set(train_batches) & set(val_batches) or set(test_groups) & set(train_batches+val_batches):
            raise AssertionError('Batch overlap detected.')
        manifest['folds'].append({'training_batches': train_batches, 'validation_batches': val_batches})
    return development, final_test, folds, manifest


def candidates():
    return [
        ('Mean baseline', DummyRegressor(strategy='mean'), {}),
        ('Linear Regression', LinearRegression(), {}),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
         {'max_depth': [5, None], 'min_samples_leaf': [1, 5]}),
        ('Support Vector Regression', make_pipeline(StandardScaler(), SVR(kernel='rbf', epsilon=.01)),
         {'svr__C': [1., 10.], 'svr__gamma': ['scale', .1]}),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42),
         {'learning_rate': [.05, .10], 'max_depth': [2, 3]})]


def train(root, progress=lambda message: None):
    started = time.perf_counter()
    root = Path(root)
    config = Config()
    progress('Initializing your simulation. This first run may take a little longer.')
    frame, audit = generate(config)
    validate_dataset(frame, config)
    development, test, folds, manifest = partitions(frame, config.seed)
    x = frame.loc[:, list(FEATURES)].to_numpy()
    y = frame.reference_capacity.to_numpy()
    xd, yd = x[development], y[development]
    comparison, search = [], []
    selected_estimator, selected_folds, selected_oof = None, None, None
    selected_row, selection_key = None, None
    for complexity, (name, estimator, grid) in enumerate(candidates()):
        progress('Preparing your simulation. Please keep this window open.')
        algorithm_best = None
        for params in ParameterGrid(grid):
            fold_scores, fitted = [], []
            oof = np.empty_like(yd)
            for training, validation in folds:
                model = clone(estimator).set_params(**params)
                model.fit(xd[training], yd[training])
                predicted = model.predict(xd[validation])
                oof[validation] = predicted
                fold_scores.append(scores(yd[validation], predicted))
                fitted.append(model)
            cv_mae = float(np.mean([s['mae'] for s in fold_scores]))
            row = {'algorithm': name, 'parameters': params, **scores(yd, oof),
                   'cv_mae_mean': cv_mae,
                   'cv_mae_std': float(np.std([s['mae'] for s in fold_scores])),
                   'fold_metrics': fold_scores}
            search.append(row)
            if algorithm_best is None or cv_mae < algorithm_best[0]['cv_mae_mean']:
                algorithm_best = (row, clone(estimator).set_params(**params), fitted, oof.copy())
        row, winner, fitted, oof = algorithm_best
        comparison.append(row)
        # Test labels never enter this decision. Complexity breaks numerical ties only.
        key = (round(row['cv_mae_mean'], 12), complexity)
        if selection_key is None or key < selection_key:
            selection_key, selected_row = key, row
            selected_estimator, selected_folds, selected_oof = winner, fitted, oof

    # Exploratory importance uses only development-validation portions.
    progress('Finalizing your simulation setup.')
    importance = []
    for (_, validation), fitted in zip(folds, selected_folds):
        result = permutation_importance(fitted, xd[validation], yd[validation],
                                        n_repeats=5, random_state=42,
                                        scoring='neg_mean_absolute_error', n_jobs=1)
        importance.append(result.importances_mean)
    importance = [{'feature': name, 'mae_increase': float(value)}
                  for name, value in zip(FEATURES, np.mean(importance, axis=0))]
    selected_estimator.fit(xd, yd)
    # Exactly one prediction pass over the reserved final test for this model run.
    test_prediction = selected_estimator.predict(x[test])
    test_scores = scores(y[test], test_prediction)
    baseline = DummyRegressor(strategy='mean').fit(xd, yd)
    baseline_test = scores(y[test], baseline.predict(x[test]))
    test_frame = frame.iloc[test].copy()
    test_frame['predicted_capacity'] = test_prediction
    test_frame['residual'] = y[test]-test_prediction
    batch_metrics = [{'batch': group, 'records': len(part),
                      **scores(part.reference_capacity, part.predicted_capacity)}
                     for group, part in test_frame.groupby('batch_id')]
    sensitivity = []
    for noise in [0., .01, .03]:
        fresh, _ = generate(replace(config, seed=314159, noise_std=noise))
        sensitivity.append({'noise_std': noise, 'records': len(fresh),
                            **scores(fresh.reference_capacity,
                                     selected_estimator.predict(fresh.loc[:, list(FEATURES)].to_numpy()))})

    run_id = 'run-' + uuid.uuid4().hex[:16]
    run_dir = root / 'models' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(run_dir/'dataset.csv', index=False, float_format='%.17g')
    audit.to_csv(run_dir/'generator-audit.csv', index=False, float_format='%.17g')
    test_frame.to_csv(run_dir/'test-predictions.csv', index=False, float_format='%.17g')
    joblib.dump(selected_estimator, run_dir/'pipeline.joblib')
    report = {'run_id': run_id, 'created_at': timestamp(), 'label': LABEL,
              'config_id': config.config_id, 'config': config.manifest(),
              'feature_order': list(FEATURES), 'units': 'mAh',
              'software': {'python': platform.python_version(), 'numpy': np.__version__,
                           'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                           'joblib': joblib.__version__},
              'selected_algorithm': selected_row['algorithm'],
              'selected_parameters': selected_row['parameters'],
              'selection_rule': 'Lowest four-fold grouped development MAE; numerical ties prefer simpler models.',
              'development_comparison': comparison, 'candidate_search': search,
              'test_metrics': test_scores, 'baseline_test_metrics': baseline_test,
              'batch_metrics': batch_metrics, 'split_manifest': manifest,
              'feature_importance': importance,
              'importance_context': 'Exploratory development-fold permutation importance; not causal effects.',
              'sensitivity': sensitivity,
              'sensitivity_context': 'Fresh synthetic records, fixed selected model; separate from reserved testing.',
              'test_points': test_frame[['reference_capacity', 'predicted_capacity', 'residual', 'batch_id']].to_dict('records'),
              'training_seconds': round(time.perf_counter()-started, 3),
              'process_peak_working_set_mb': peak_memory_mb(),
              'memory_context': 'Process-lifetime peak working set, including imports and training; not isolated model memory.',
              'checksums': {name: sha256(run_dir/name) for name in
                            ['dataset.csv', 'generator-audit.csv', 'test-predictions.csv', 'pipeline.joblib']}}
    write_json(run_dir/'report.json', report)
    write_json(root/'latest.json', {'run_id': run_id, 'report_sha256': sha256(run_dir/'report.json')})
    return selected_estimator, report


def load(root):
    root = Path(root)
    latest = json.loads((root/'latest.json').read_text(encoding='utf-8'))
    run_id = latest['run_id']
    if not re.fullmatch(r'run-[a-f0-9]{16}', run_id):
        raise ValueError('Invalid saved model identifier.')
    run_dir = root/'models'/run_id
    if sha256(run_dir/'report.json') != latest['report_sha256']:
        raise ValueError('Saved model report failed its integrity check.')
    report = json.loads((run_dir/'report.json').read_text(encoding='utf-8'))
    if report['config_id'] != Config().config_id or report['feature_order'] != list(FEATURES):
        raise ValueError('Saved model has an incompatible simulation version or feature order.')
    if report['software']['scikit-learn'] != sklearn.__version__:
        raise ValueError('Saved model requires its recorded scikit-learn version.')
    for package, installed in [('numpy',np.__version__),('pandas',pd.__version__),('joblib',joblib.__version__)]:
        if report['software'][package] != installed:
            raise ValueError(f'Saved model requires its recorded {package} version.')
    for name in ['dataset.csv', 'generator-audit.csv', 'test-predictions.csv', 'pipeline.joblib']:
        if sha256(run_dir/name) != report['checksums'][name]:
            raise ValueError(f'{name} failed its integrity check.')
    # Never accept a model path from an upload or API request.
    return joblib.load(run_dir/'pipeline.joblib'), report
