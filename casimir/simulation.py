"""Versioned synthetic domain from the corrected research specification."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd

FEATURES = ('m', 'f', 'p', 'c', 'e')
RANGES = {'m': (.010, .020), 'f': (.85, .95), 'p': (.25, .45),
          'c': (.10, .30), 'e': (.80, 1.20)}
LABEL = 'SYNTHETIC SIMULATION — not measured battery data'


@dataclass(frozen=True)
class Config:
    version: str = '1.0.0'
    seed: int = 42
    batches: int = 20
    records_per_batch: int = 100
    specific_capacity: float = 150.0
    utilization: float = .92
    porosity_penalty: float = .12
    compression_penalty: float = .08
    filling_penalty: float = .10
    batch_std: float = .015
    noise_std: float = .010

    def manifest(self):
        return {**asdict(self), 'ranges': RANGES, 'feature_order': FEATURES,
                'centres': [.35, .20, 1.0], 'normalizers': [.10, .10, .20],
                'clip': [.05, 1.0], 'sampling': 'independent uniform',
                'formula': 'm*f*qs*clip(u+b+noise,.05,1); '
                           'u=.92*(1-.12*zp^2)*(1-.08*zc^2)*(1-.10*ze^2)',
                'label': LABEL}

    @property
    def config_id(self):
        encoded = json.dumps(self.manifest(), sort_keys=True).encode()
        return 'sim-' + hashlib.sha256(encoded).hexdigest()[:16]


def validate_inputs(values):
    if not isinstance(values, dict) or set(values) != set(FEATURES):
        raise ValueError('Provide exactly the five scenario values: m, f, p, c, e.')
    clean = {}
    for name in FEATURES:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number.')
        lo, hi = RANGES[name]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}.')
        clean[name] = float(value)
    return clean


def components(x, config=Config()):
    x = np.asarray(x, dtype=float)
    m, f, p, c, e = x.T
    nominal = m * f * config.specific_capacity
    u = (config.utilization * (1-config.porosity_penalty*((p-.35)/.10)**2)
         * (1-config.compression_penalty*((c-.20)/.10)**2)
         * (1-config.filling_penalty*((e-1)/.20)**2))
    return nominal, u


def analytical_reference(values, config=Config()):
    clean = validate_inputs(values)
    nominal, u = components([[clean[k] for k in FEATURES]], config)
    return float((nominal * u)[0])


def generate(config=Config()):
    if config.batches < 8 or config.records_per_batch < 2:
        raise ValueError('Generation needs at least 8 batches and 2 cells per batch.')
    rng = np.random.default_rng(config.seed)
    records, audit = [], []
    for batch in range(config.batches):
        batch_effect = rng.normal(0, config.batch_std)
        x = np.column_stack([rng.uniform(*RANGES[k], config.records_per_batch)
                             for k in FEATURES])
        noise = rng.normal(0, config.noise_std, config.records_per_batch)
        nominal, u = components(x, config)
        targets = nominal * np.clip(u+batch_effect+noise, .05, 1.)
        for i in range(config.records_per_batch):
            cell_id = f'CELL-{batch+1:02d}-{i+1:04d}'
            records.append({'cell_id': cell_id, 'batch_id': f'BATCH-{batch+1:02d}',
                            'config_id': config.config_id,
                            **dict(zip(FEATURES, x[i])),
                            'reference_capacity': targets[i]})
            audit.append({'cell_id': cell_id, 'nominal_capacity': nominal[i],
                          'utilization': u[i], 'batch_effect': batch_effect,
                          'record_noise': noise[i], 'analytical_reference': nominal[i]*u[i]})
    return pd.DataFrame(records), pd.DataFrame(audit)


def validate_dataset(frame, config=Config()):
    required = {'cell_id', 'batch_id', 'config_id', *FEATURES, 'reference_capacity'}
    if not required.issubset(frame.columns):
        raise ValueError('Dataset is missing required columns.')
    if frame[list(required)].isna().any().any():
        raise ValueError('Dataset has missing required values.')
    if frame.cell_id.duplicated().any():
        raise ValueError('Dataset cell identifiers must be unique.')
    if not frame.config_id.eq(config.config_id).all():
        raise ValueError('Dataset configuration is incompatible.')
    for key in (*FEATURES, 'reference_capacity'):
        if not pd.api.types.is_numeric_dtype(frame[key]) or not np.isfinite(frame[key]).all():
            raise ValueError(f'{key} must contain finite numerical values.')
    for key, (lo, hi) in RANGES.items():
        if not frame[key].between(lo, hi).all():
            raise ValueError(f'{key} is outside the supported range.')
    if (frame.reference_capacity <= 0).any():
        raise ValueError('Reference capacities must be positive.')
    if frame[list(FEATURES)+['reference_capacity']].duplicated().any():
        raise ValueError('Dataset contains repeated copies of a record.')
    return frame
