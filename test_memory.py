"""Small CPU-only synthetic verification of memory.py; never starts the pilot."""
from __future__ import annotations

import io
import unittest

import numpy as np
import torch

import linear
import memory


def histories():
    cfg = linear.Config(horizon=90, change_tick=34)
    rows = [linear.rollout(cfg,linear.generate_tape(cfg,regime,2,100+i))
            for i,regime in enumerate(('nominal','q_up','r_up','ab_switch'))]
    return cfg,rows


class CausalMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.rows = histories()
        cls.mem_cfg = memory.MemoryConfig(optimizer_steps=3,sequence_length=20,burn_in=4,
                                          startup_ticks=7,batch_size=4)
        cls.callbacks = []
        cls.models,cls.receipt = memory.train_pair(cls.rows,cls.mem_cfg,103,
            callback=lambda step,total: cls.callbacks.append((step,total)))

    def test_equal_training_and_checkpoint_roundtrip(self):
        self.assertEqual(self.callbacks,[(1,3),(2,3),(3,3)])
        self.assertEqual(self.receipt['optimizer_updates_total'],6)
        self.assertTrue(self.receipt['same_initial_shared_parameters'])
        self.assertLessEqual(self.receipt['active_parameter_count'][memory.ADAPTIVE] /
                             self.receipt['active_parameter_count'][memory.STATIC],1.05)
        for model in self.models.values():
            payload = memory.checkpoint(model)
            stream = io.BytesIO()
            torch.save(payload,stream)
            stream.seek(0)
            reloaded = memory.restore_checkpoint(torch.load(stream,weights_only=True,map_location='cpu'))
            np.testing.assert_array_equal(memory.predict_many(model,self.rows),memory.predict_many(reloaded,self.rows))

    def test_batch_predictions_match_causal_streaming_and_do_not_read_truth(self):
        row = self.rows[1]
        untouched = row['raw'].copy()
        for model in self.models.values():
            stream = memory.new_observer(model)
            values = np.array([k.astype(float)+stream(x,k) for x,k in zip(row['raw'],row['kf'],strict=True)])
            testonly = [{'raw':row['raw'],'kf':row['kf']}] * 2
            batched = memory.predict_many(model,testonly)
            self.assertEqual(batched.shape,(2,len(row['raw']),2))
            np.testing.assert_allclose(batched[0],values,atol=1e-7,rtol=1e-6)
            np.testing.assert_array_equal(batched[0,:7],row['kf'][:7])
            np.testing.assert_array_equal(row['raw'], untouched)
            taus = np.array(stream.tau_history)
            self.assertTrue(np.all((taus >= .015)&(taus <= 1.5)))
            if model.arm == memory.STATIC:
                np.testing.assert_allclose(taus, np.broadcast_to(taus[0],taus.shape),atol=0,rtol=0)

    def test_symmetric_perturbation_first_appears_at_delivery(self):
        tape = linear.generate_tape(self.cfg,'q_up',4,819)
        result = memory.effective_memory(self.cfg,tape,self.models,self.cfg.change_tick+12,horizon_ticks=20)
        self.assertEqual(result['arrival_tick'],result['source_tick']+4)
        self.assertEqual(result['arrival_age_seconds'],.08)
        for arm,metrics in result['methods'].items():
            self.assertTrue(metrics['future_actions_fixed'])
            self.assertTrue(metrics['prearrival_unchanged'])
            self.assertEqual(metrics['lag_ticks'],list(range(20)))
            self.assertGreater(abs(metrics['arrival_position_sensitivity']),.001)
            self.assertAlmostEqual(metrics['physical_age_seconds'][-1],.46)
            if arm in memory.ARMS:
                self.assertEqual(metrics['baseline_internal_tau']['ticks'],20)


if __name__ == '__main__':
    unittest.main()
