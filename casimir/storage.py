"""Local, traceable storage. All connections enforce foreign keys."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import uuid

from .learning import timestamp
from .simulation import Config


class Repository:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS SimulationConfig (
              config_id TEXT PRIMARY KEY, version TEXT NOT NULL, specification TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS Batch (
              batch_id TEXT PRIMARY KEY, config_id TEXT NOT NULL REFERENCES SimulationConfig,
              partition_name TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS CellRecord (
              record_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES Batch,
              source_kind TEXT NOT NULL, inputs TEXT NOT NULL, reference_capacity REAL);
            CREATE TABLE IF NOT EXISTS ModelRun (
              model_id TEXT PRIMARY KEY, config_id TEXT NOT NULL REFERENCES SimulationConfig,
              algorithm TEXT NOT NULL, artifact_path TEXT NOT NULL, report TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS Prediction (
              prediction_id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES CellRecord,
              model_id TEXT NOT NULL REFERENCES ModelRun, predicted_capacity REAL NOT NULL,
              units TEXT NOT NULL, created_at TEXT NOT NULL, saved INTEGER NOT NULL DEFAULT 0,
              result TEXT NOT NULL);
            ''')
            config = Config()
            db.execute('INSERT OR IGNORE INTO SimulationConfig VALUES (?,?,?)',
                       (config.config_id, config.version, json.dumps(config.manifest())))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        try:
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db
        finally:
            db.close()

    def register_model(self, report, dataset):
        config_id = report['config_id']
        with self.connect() as db:
            for batch, partition in [('development_batches','development'), ('final_test_batches','test')]:
                db.executemany('INSERT OR IGNORE INTO Batch VALUES (?,?,?)',
                               [(config_id+':'+b, config_id, partition)
                                for b in report['split_manifest'][batch]])
            db.executemany('INSERT OR IGNORE INTO CellRecord VALUES (?,?,?,?,?)',
                           [(config_id+':'+r['cell_id'], config_id+':'+r['batch_id'], 'synthetic',
                             json.dumps({k:r[k] for k in ['m','f','p','c','e']}), r['reference_capacity'])
                            for r in dataset.to_dict('records')])
            db.execute('INSERT OR IGNORE INTO ModelRun VALUES (?,?,?,?,?)',
                       (report['run_id'], config_id, report['selected_algorithm'],
                        'models/'+report['run_id']+'/pipeline.joblib', json.dumps(report)))

    def add_prediction(self, result):
        config_id = result['config_id']
        batch_id = config_id+':SCENARIOS'
        record_id = 'scenario-'+uuid.uuid4().hex
        prediction_id = 'prediction-'+uuid.uuid4().hex
        result = {**result, 'prediction_id': prediction_id}
        with self.connect() as db:
            model = db.execute('SELECT config_id FROM ModelRun WHERE model_id=?',
                               (result['model_run_id'],)).fetchone()
            if not model or model[0] != config_id:
                raise ValueError('Scenario and model configurations are incompatible.')
            db.execute('INSERT OR IGNORE INTO Batch VALUES (?,?,?)', (batch_id, config_id, 'scenario'))
            db.execute('INSERT INTO CellRecord VALUES (?,?,?,?,NULL)',
                       (record_id, batch_id, 'scenario', json.dumps(result['inputs'])))
            db.execute('INSERT INTO Prediction VALUES (?,?,?,?,?,?,0,?)',
                       (prediction_id, record_id, result['model_run_id'], result['capacity'],
                        'mAh', result['createdAt'], json.dumps(result)))
        return result

    def save(self, prediction_id):
        with self.connect() as db:
            updated = db.execute('UPDATE Prediction SET saved=1 WHERE prediction_id=?', (prediction_id,))
            if not updated.rowcount:
                raise ValueError('This result could not be found.')

    def history(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT result FROM Prediction WHERE saved=1 ORDER BY created_at DESC LIMIT 500')]

    def report(self, model_id):
        with self.connect() as db:
            row = db.execute('SELECT report FROM ModelRun WHERE model_id=?', (model_id,)).fetchone()
            if not row:
                raise ValueError('Model report not found.')
            return json.loads(row[0])
