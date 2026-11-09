import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import numpy as np

from casimir.learning import load, partitions, scores, sha256, train, write_json
from casimir.service import Engine
from casimir.simulation import Config, FEATURES, analytical_reference, generate, validate_dataset, validate_inputs
from casimir.storage import Repository


DEFAULT={'m':.015,'f':.90,'p':.35,'c':.20,'e':1.00}


class SimulationTests(unittest.TestCase):
    def test_documented_zero_noise_example(self):
        self.assertAlmostEqual(analytical_reference(DEFAULT),1.863,places=12)

    def test_reproducibility_and_physical_bounds(self):
        frame,audit=generate()
        repeated,repeated_audit=generate()
        self.assertTrue(frame.equals(repeated))
        self.assertTrue(audit.equals(repeated_audit))
        self.assertEqual(len(frame),2000)
        self.assertTrue((frame.reference_capacity>0).all())
        self.assertTrue((frame.reference_capacity.to_numpy()<=audit.nominal_capacity.to_numpy()).all())
        self.assertEqual(list(frame[list(FEATURES)].columns),list(FEATURES))

    def test_input_validation(self):
        for value in ['',None,True,float('nan'),float('inf'),.03]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_inputs({**DEFAULT,'m':value})
        with self.assertRaises(ValueError):validate_inputs({**DEFAULT,'extra':0})
        with self.assertRaises(ValueError):validate_inputs({'m':.015})

    def test_dataset_validation(self):
        frame,_=generate()
        validate_dataset(frame)
        with self.assertRaises(ValueError):validate_dataset(frame.drop(columns=['m']))
        with self.assertRaises(ValueError):validate_dataset(frame.assign(cell_id='duplicate'))
        broken=frame.copy();broken.loc[0,'f']=float('nan')
        with self.assertRaises(ValueError):validate_dataset(broken)

    def test_partition_integrity(self):
        frame,_=generate()
        development,test,folds,manifest=partitions(frame)
        self.assertEqual((len(development),len(test)),(1600,400))
        self.assertFalse(set(frame.iloc[development].batch_id)&set(frame.iloc[test].batch_id))
        visits=np.zeros(len(development),dtype=int)
        for training,validation in folds:
            self.assertFalse(set(frame.iloc[development[training]].batch_id)&set(frame.iloc[development[validation]].batch_id))
            visits[validation]+=1
        np.testing.assert_array_equal(visits,np.ones(len(development)))

    def test_undefined_metrics_are_explicit(self):
        self.assertIsNone(scores([1,1],[1,1])['r2'])
        self.assertIsNone(scores([0,1],[0,1])['mape'])


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(os.environ.get('CASIMIR_TEST_DATA',cls.temp.name))
        if os.environ.get('CASIMIR_TEST_DATA') and (cls.root/'latest.json').exists():
            cls.model,cls.report=load(cls.root)
        else:
            cls.model,cls.report=train(cls.root,lambda m:print(m,flush=True))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_accuracy_on_reserved_batches_and_search(self):
        report=self.report
        self.assertEqual(len(report['development_comparison']),5)
        self.assertEqual(len(report['candidate_search']),14)
        self.assertEqual(len(report['test_points']),400)
        self.assertLess(report['test_metrics']['mae'],report['baseline_test_metrics']['mae'])
        self.assertLess(report['test_metrics']['mae'],.12)
        expected=min(report['development_comparison'],key=lambda r:r['cv_mae_mean'])
        self.assertEqual(expected['algorithm'],report['selected_algorithm'])
        self.assertEqual(len(report['sensitivity']),3)

    def test_reloaded_model_and_input_response(self):
        model,report=load(self.root)
        x=np.array([[DEFAULT[k] for k in FEATURES]])
        np.testing.assert_allclose(model.predict(x),self.model.predict(x),atol=1e-12)
        lighter=x.copy();lighter[0,0]=.011
        self.assertGreater(model.predict(x)[0],model.predict(lighter)[0])

    def test_schema_and_artifact_integrity(self):
        report_path=self.root/'models'/self.report['run_id']/'report.json'
        latest_path=self.root/'latest.json'
        report_bytes,latest_bytes=report_path.read_bytes(),latest_path.read_bytes()
        try:
            changed=json.loads(report_bytes);changed['feature_order']=list(reversed(FEATURES))
            write_json(report_path,changed)
            write_json(latest_path,{'run_id':self.report['run_id'],'report_sha256':sha256(report_path)})
            with self.assertRaises(ValueError):load(self.root)
        finally:
            report_path.write_bytes(report_bytes);latest_path.write_bytes(latest_bytes)
        pipeline=self.root/'models'/self.report['run_id']/'pipeline.joblib'
        original=pipeline.read_bytes()
        try:
            pipeline.write_bytes(original+b'changed')
            with self.assertRaises(ValueError):load(self.root)
        finally:pipeline.write_bytes(original)

    def test_service_and_sqlite_roundtrip(self):
        engine=Engine(self.root)
        try:
            job=engine.initialize(DEFAULT)['job_id']
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                status=engine.status(job)
                if status['state'] in ('complete','failed'):break
                time.sleep(.05)
            self.assertEqual(status['state'],'complete',status)
            result=status['result']
            engine.repository.save(result['prediction_id'])
            saved=engine.repository.history()
            self.assertEqual(saved[0]['inputs'],DEFAULT)
            self.assertEqual(saved[0]['units'],'mAh')
            self.assertEqual(saved[0]['model_run_id'],self.report['run_id'])
            self.assertAlmostEqual(result['reference'],1.863)
            with engine.repository.connect() as db:
                self.assertEqual(db.execute('PRAGMA foreign_keys').fetchone()[0],1)
                self.assertEqual(db.execute('SELECT count(*) FROM CellRecord WHERE source_kind="synthetic"').fetchone()[0],2000)
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
        finally:engine.close()

    def test_http_boundaries(self):
        from server import Handler
        engine=Engine(self.root)
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,engine=engine))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(url+'/api/health') as response:
                self.assertEqual(json.load(response)['status'],'ready')
            request=urllib.request.Request(url+'/api/initialize',data=json.dumps({'inputs':{**DEFAULT,'m':.1}}).encode(),headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code,400)
            error.exception.close()
            foreign=urllib.request.Request(url+'/api/initialize',data=json.dumps({'inputs':DEFAULT}).encode(),headers={'Content-Type':'application/json','Origin':'https://foreign.example'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(foreign)
            self.assertEqual(error.exception.code,403)
            error.exception.close()
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url+'/data/latest.json')
            error.exception.close()
        finally:
            server.shutdown();server.server_close();engine.close();worker.join()


if __name__=='__main__':unittest.main(verbosity=2)
