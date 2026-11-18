#!/usr/bin/env python3
"""Do v3.1 passengers sit INSIDE the bulkhead rectangle B, and does that drive the dP / t_blind results?
Re-runs the v3.1 tipping-point sweep with and without resampling positions that fall inside B (CRN-paired)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TrialModelBus"))

import math
import numpy as np
from dataclasses import replace
import upd_montecarlo_bus_standart as q

def inB(x, y, r): return (x >= r[0]) & (x <= r[1]) & (y >= r[2]) & (y <= r[3])

def sim(vc, policy, p, n, rng, excl):
    L = vc.length; door_x = np.asarray(vc.door_frac) * L; nd, k = door_x.size, p.n_inspectors; idx = np.arange(n)
    right = rng.random(n) < p.p_right_seat
    xa = rng.uniform(0.1 * L, 0.9 * L, n)
    ya = np.where(right, rng.uniform(0.3, 0.9, n), rng.uniform(1.65, 2.25, n))
    if excl:   # resample inside-B positions (same draw count in every scenario -> CRN preserved)
        bad = inB(xa, ya, vc.partition)
        for _ in range(50):
            if not bad.any(): break
            xr = rng.uniform(0.1 * L, 0.9 * L, n); xa = np.where(bad, xr, xa); bad = inB(xa, ya, vc.partition)
    doors = q.assign_doors(policy, n, k, nd, rng); dxj = door_x[doors]
    occ_pass = q.segment_hits_rect(xa[:, None], ya[:, None], dxj, q.PLATFORM_Y, vc.partition) & p.partition_on & p.block_passenger_view
    mu = math.log(p.react_mean) - p.react_sigma ** 2 / 2
    t_react = rng.lognormal(mu, p.react_sigma, n); lead = rng.uniform(0.0, p.lead_max, n)
    t_occ = rng.uniform(p.occ_lo, p.occ_hi, (n, k))
    t_det = np.where(occ_pass, t_react[:, None] + t_occ, np.maximum(0.0, t_react - lead)[:, None]).min(axis=1)
    rho = rng.uniform(p.rho_lo, p.rho_hi, n)
    blocked = np.zeros((n, nd), bool); blocked[idx[:, None], doors] = True
    v_agent = np.clip(rng.normal(p.agent_speed, 0.2, n), 0.5, 2.5); v_eff = v_agent * q.speed_factor(rho, p, "agent")
    dist_all = np.hypot(xa[:, None] - door_x[None, :], ya[:, None])
    t_egress = np.where(blocked, np.inf, dist_all / v_eff[:, None]).min(axis=1); t_escape = t_det + t_egress
    d_ins = np.hypot(xa[:, None] - dxj, ya[:, None])
    v_ins = np.clip(rng.normal(p.insp_speed, 0.15, (n, k)), 0.3, 1.5) * q.speed_factor(rho, p, "inspector")[:, None]
    queue = rng.exponential(p.queue_mean, (n, k)); t_dwell = rng.uniform(p.dwell_lo, p.dwell_hi, n)
    blind = rng.uniform(p.insp_blind_lo, p.insp_blind_hi, (n, k))
    occ_i = q.segment_hits_rect(dxj, 0.0, xa[:, None], ya[:, None], vc.partition) & p.partition_on & p.block_inspector_view
    t_check = (d_ins / v_ins + queue + np.where(occ_i, blind, 0.0)).min(axis=1)
    return ((t_escape < t_check) & (t_escape <= t_dwell)).astype(float), inB(xa, ya, vc.partition)

def run(ci, vc, p, n, seed, excl):
    return sim(vc, "single_door", p, n, np.random.default_rng([seed, ci, 0]), excl)

def tip(vals):
    for (t0, v0), (t1, v1) in zip(vals, vals[1:]):
        if v0 <= 0 < v1: return t0 + (0 - v0) / (v1 - v0) * (t1 - t0)

if __name__ == "__main__":
    n, seed = 400_000, 20261009
    grid = np.arange(0.0, 5.01, 0.25); p0 = q.Params()
    print(f"{'Class':<12}{'L (m)':>7}{'inside B %':>12}{'dP base (pp)':>14}{'tip (s)':>9} | {'dP base excl':>13}{'tip excl (s)':>13}")
    for ci, vc in enumerate(q.CLASSES):
        out = []
        for excl in (False, True):
            off, ins = run(ci, vc, replace(p0, partition_on=False), n, seed, excl)
            on, _ = run(ci, vc, p0, n, seed, excl)
            curve = []
            for tb in grid:
                s, _ = run(ci, vc, replace(p0, insp_blind_lo=tb, insp_blind_hi=tb), n, seed, excl)
                curve.append((tb, float((s - off).mean())))
            out.append((100 * float((on - off).mean()), tip(curve), float(ins.mean())))
        (db, tp, fi), (db2, tp2, _) = out
        print(f"{vc.name.split()[0]:<12}{vc.length:>7.2f}{100*fi:>12.2f}{db:>14.2f}{tp:>9.2f} | {db2:>13.2f}{tp2:>13.2f}")