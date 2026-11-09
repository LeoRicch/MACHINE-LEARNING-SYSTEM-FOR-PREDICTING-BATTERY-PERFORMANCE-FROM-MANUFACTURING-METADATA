"""One user action orchestrates initialization, fitting/loading, and prediction."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import uuid

import numpy as np
import pandas as pd

from .learning import load, timestamp, train
from .simulation import Config, FEATURES, LABEL, analytical_reference, validate_inputs
from .storage import Repository


class Engine:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repository = Repository(self.root/'casimir.sqlite3')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='casimir')
        self.lock = threading.Lock()
        self.jobs = {}
        self.model = None
        self.report = None

    def initialize(self, values):
        clean = validate_inputs(values)
        with self.lock:
            if sum(j['state'] in ('queued','running') for j in self.jobs.values()) >= 3:
                raise ValueError('A simulation is already in progress. Please wait for it to finish.')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'state':'queued', 'message':'Initializing your simulation.'}
            # Keep completed in-memory jobs bounded; persisted predictions remain in SQLite.
            completed = [key for key,value in self.jobs.items() if value['state'] in ('complete','failed')]
            for key in completed[:-100]:
                del self.jobs[key]
        self.executor.submit(self._run, job_id, clean)
        return {'job_id':job_id}

    def _update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def _run(self, job_id, inputs):
        self._update(job_id, state='running', message='Initializing your simulation.')
        try:
            if self.model is None:
                if (self.root/'latest.json').exists():
                    model, report = load(self.root)
                else:
                    model, report = train(self.root,
                        lambda message:self._update(job_id,message=message))
                dataset = pd.read_csv(self.root/'models'/report['run_id']/'dataset.csv')
                self.repository.register_model(report, dataset)
                self.model, self.report = model, report
            self._update(job_id, message='Preparing your result.')
            capacity = float(self.model.predict(np.array([[inputs[k] for k in FEATURES]]))[0])
            if not np.isfinite(capacity) or capacity <= 0:
                raise ValueError('The model could not produce a supported capacity for this scenario.')
            reference = analytical_reference(inputs)
            result = {'capacity':capacity, 'reference':reference, 'inputs':inputs,
                      'nominal_capacity':inputs['m']*inputs['f']*Config().specific_capacity,
                      'model_run_id':self.report['run_id'],
                      'model':self.report['selected_algorithm'], 'config_id':Config().config_id,
                      'simulation_version':Config().version, 'kind':'synthetic_prediction',
                      'createdAt':timestamp(), 'units':'mAh', 'label':LABEL,
                      'test_metrics':self.report['test_metrics'],
                      'test_records':self.report['split_manifest']['final_test_records']}
            result = self.repository.add_prediction(result)
            self._update(job_id, state='complete', message='Your result is ready.', result=result)
        except Exception:
            import logging
            logging.exception('Simulation job failed')
            self._update(job_id, state='failed',
                         message='We couldn’t initialize this simulation. Please check the app log or restart the app.')

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError('Simulation session not found. Please initialize again.')
            return dict(self.jobs[job_id])

    def close(self):
        self.executor.shutdown(wait=True)
