from __future__ import annotations

import json
import unittest

import numpy as np

from linear import Config, DelayedKalman, Packet, calibration_report, causal_attribution, dynamics, generate_tape, rollout


class CausalLinearPilot(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(horizon=96, change_tick=40)

    def test_onset_variance_matched_but_oracle_gains_differ(self):
        exp = calibration_report(self.cfg)
        a = rollout(self.cfg, generate_tape(self.cfg, "q_up", 4, 901), "oracle_kf")
        b = rollout(self.cfg, generate_tape(self.cfg, "r_up", 4, 901), "oracle_kf")
        acq, arrive = self.cfg.change_tick, self.cfg.change_tick + 4
        self.assertEqual(int(a["arrived_source_tick"][arrive]), acq)
        self.assertAlmostEqual(a["innovation_var"][arrive], exp["position_innovation_variance_q_up"], 14)
        self.assertAlmostEqual(a["innovation_var"][arrive], b["innovation_var"][arrive], 14)
        self.assertGreater(a["position_gain"][arrive], exp["oracle_position_gains"]["nominal"])
        self.assertLess(b["position_gain"][arrive], exp["oracle_position_gains"]["nominal"])
        np.testing.assert_array_equal(a["action"], b["action"])
        np.testing.assert_array_equal(a["truth"][:acq], b["truth"][:acq])
        self.assertTrue(a["oracle_parameters_privileged"])

    def test_observation_impulse_first_visible_at_delivery(self):
        t = generate_tape(self.cfg, "nominal", 4, 902)
        control = rollout(self.cfg, t)
        alt = rollout(self.cfg, t, perturb_source_tick=22, perturb_delta=.01)
        np.testing.assert_array_equal(control["estimate"][:26], alt["estimate"][:26])
        np.testing.assert_array_equal(control["truth"], alt["truth"])
        self.assertGreater(np.max(np.abs(control["estimate"][26]-alt["estimate"][26])), 0)
        self.assertFalse(alt["raw"][:4,1].any())
        self.assertEqual(float(alt["raw"][4,3]), np.float32(.08))

    def test_closed_loop_is_not_fixed_trajectory_replay(self):
        t = generate_tape(self.cfg, "r_up", 2, 803)
        baseline = rollout(self.cfg, t, closed_loop=True)
        changed = rollout(self.cfg, t, residual=lambda _x, _kf: [.04,0.], closed_loop=True)
        self.assertIsNotNone(baseline["post_change_tracking_rmse"])
        self.assertNotEqual(float(baseline["action"][0]), float(changed["action"][0]))
        self.assertNotEqual(float(baseline["truth"][1,1]), float(changed["truth"][1,1]))
        for r in (baseline, changed):
            np.testing.assert_array_equal(r["raw"][1:,5], r["action"][:-1].astype(np.float32))
        self.assertEqual([(r["source_tick"],r["arrival_tick"]) for r in baseline["event_records"]],
                         [(r["source_tick"],r["arrival_tick"]) for r in changed["event_records"]])

    def test_out_of_order_does_not_replace_latest_packet(self):
        cfg = Config(horizon=80, change_tick=40, delay_jitter_max=5)
        t = generate_tape(cfg, "nominal", 2, 9011)
        r = rollout(cfg, t)
        times = [row["source_tick"] for row in r["event_records"]]
        self.assertTrue(any(i > j for i,j in zip(times,times[1:])))
        self.assertTrue(np.all(np.diff(r["source_tick"]) >= 0))
        for e in r["event_records"]:
            self.assertLessEqual(e["source_tick"], e["arrival_tick"])
        self.assertTrue(np.isfinite(r["estimate"]).all())

    def test_delivered_packet_is_never_read_from_future(self):
        f = DelayedKalman(self.cfg)
        with self.assertRaises(ValueError):
            f.step(0, 0., [Packet(0, 2, 1.)])
        t = generate_tape(self.cfg, "nominal", 0, 99)
        with self.assertRaises(ValueError):
            rollout(self.cfg, t, "oracle_kf", residual=lambda _a,_b: [0,0])

    def test_oracle_same_as_fixed_in_nominal_and_ab_only_changes_dynamics(self):
        t = generate_tape(self.cfg, "nominal", 2, 811)
        np.testing.assert_array_equal(rollout(self.cfg,t)["estimate"], rollout(self.cfg,t,"oracle_kf")["estimate"])
        f, b, q, r = dynamics(self.cfg)
        fa, ba, qa, ra = dynamics(self.cfg, "ab_switch")
        self.assertFalse(np.array_equal(f,fa))
        self.assertFalse(np.array_equal(b,ba))
        np.testing.assert_array_equal(q,qa)
        self.assertEqual(r,ra)

    def test_attribution_is_json_and_callbacks_use_disjoint_groups(self):
        seen = []
        result = causal_attribution(self.cfg, range(810, 815), range(830, 835), 2,
                                    event_count=6, callback=lambda d,n:seen.append((d,n)))
        self.assertEqual(seen[-1], (20,20))
        self.assertEqual(len(seen), 20)
        self.assertEqual(result["scores"]["causal_history"]["post_shift_packets_used"],6)
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            causal_attribution(self.cfg, range(810,816),range(815,821),2)


if __name__ == "__main__":
    unittest.main()
