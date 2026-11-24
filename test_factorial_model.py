from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

import factorial_model as f
import linear
from memory import _train_norm


def scenario_rows():
    c = linear.Config(horizon=55, change_tick=20)
    rows = []
    for i, tag in enumerate(linear.REGIMES):
        r = linear.rollout(c, linear.generate_tape(c, tag, 2, 86581+i))
        event = np.zeros((c.horizon,len(f.EVENT_FEATURES)),dtype=np.float32)
        for row in r["event_records"]:
            t = row["arrival_tick"]
            event[t] = (1.,row["innovation"],row["innovation_var"],
                        (t-row["source_tick"])*c.dt,0.,0.)
        rows.append({**r,"event":event})
    return rows


class FactorialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = scenario_rows()
        cls.cfg = f.FactorialConfig(optimizer_steps=3,batch_size=2,sequence_length=20,burn_in=4)
        cls.norm = _train_norm(*(np.stack([r[k] for r in cls.rows]) for k in ("raw","kf","truth")),cls.cfg)

    def model(self, arm):
        torch.manual_seed(10020)
        return f.FactorialObserver(arm,self.cfg,self.norm)

    def test_exact_zero_init_and_four_arm_determinism(self):
        seen=[]
        models,r = f.train_four(self.rows,25100,self.cfg,lambda a,b:seen.append((a,b)))
        twin,s = f.train_four(self.rows,25100,self.cfg)
        self.assertEqual(r["active_parameter_count"],{f.ARMS[0]:1514,f.ARMS[1]:1519,f.ARMS[2]:1523,f.ARMS[3]:1528})
        self.assertEqual(r["optimizer_batch_index_sha256"],s["optimizer_batch_index_sha256"])
        self.assertEqual(r["training_array_sha256"],s["training_array_sha256"])
        self.assertEqual(seen,[(1,3),(2,3),(3,3)])
        for arm in f.ARMS:
            for k,v in models[arm].state_dict().items():
                self.assertTrue(torch.equal(v,twin[arm].state_dict()[k]))
            init = f.predict_many(self.model(arm),self.rows)
            np.testing.assert_array_equal(init,np.stack([x["kf"] for x in self.rows]))
        self.assertGreater(torch.count_nonzero(models[f.ARMS[3]].rate_context_head.weight), 0)
        self.assertGreater(torch.count_nonzero(models[f.ARMS[3]].gain_context_head.weight), 0)

    def test_step_event_free_propagation_and_zero_innovation_null(self):
        m=self.model(f.ARMS[3])
        with torch.no_grad():
            m.rate_context_head.weight.fill_(.5)
            m.gain_context_head.weight.fill_(.3)
            m.readout.weight.fill_(.15)
        row=self.rows[0]
        raw=torch.from_numpy(row["raw"][28:29])
        kf=torch.from_numpy(row["kf"][28:29])
        c=m.initial(1)
        _,c,_=m.step(raw,kf,torch.zeros((1,6)),c)
        e1=torch.tensor([[1.,.06,.004,.04,0.,0.]])
        e2=e1.clone(); e2[0,1]=.12
        nil=e1.clone(); nil[0,1]=0.
        _,c1,d1=m.step(raw,kf,e1,c)
        _,_,d2=m.step(raw,kf+.03,e2,c)
        _,cn,dn=m.step(raw,kf,nil,c)
        _,cz,dz=m.step(raw,kf,torch.zeros_like(e1),c)
        self.assertTrue(torch.equal(d1["tau"],d2["tau"]))
        self.assertFalse(torch.equal(d1["gain"],d2["gain"]))
        self.assertEqual(torch.count_nonzero(dn["innovation_update"]),0)
        self.assertEqual(torch.count_nonzero(dz["innovation_update"]),0)
        self.assertTrue(torch.equal(cn[0],cz[0]))
        _,_,future1=m.step(raw,kf,torch.zeros_like(e1),c1)
        _,_,futuren=m.step(raw,kf,torch.zeros_like(e1),cn)
        self.assertFalse(torch.equal(future1["tau"],futuren["tau"]))

    def test_gate_batch_online_parity_and_bounds(self):
        m=self.model(f.ARMS[3])
        with torch.no_grad():
            m.readout.weight.fill_(.7)
        r=self.rows[1]
        online=f.new_residual(m)
        delta=np.stack([online(x,k,e) for x,k,e in zip(r["raw"],r["kf"],r["event"],strict=True)])
        self.assertTrue(np.all(delta[:self.cfg.startup_ticks]==0.))
        self.assertTrue(np.any(delta[self.cfg.startup_ticks:]!=0.))
        self.assertTrue(np.all(np.abs(delta)<=m.residual_cap.numpy().astype(np.float64)))
        np.testing.assert_allclose(f.predict_many(m,[r])[0],r["kf"]+delta,rtol=1e-6,atol=2e-7)

    def test_bad_event_cannot_carry_hidden_information(self):
        bad=self.rows[0]["event"].copy()
        bad[0,1]=.05
        with self.assertRaises(ValueError):
            f.train_four([{**self.rows[0],"event":bad}],31,replace(self.cfg,optimizer_steps=1))


if __name__ == "__main__":
    unittest.main()
