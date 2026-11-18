#!/usr/bin/env python3
"""
Qoyanshyq Table I -- Monte Carlo coverage-audit harness (v3.2).

P_slip = probability that an inspection fails to verify a non-compliant
benchmark passenger during one stop, per vehicle class and INSPECTION POLICY.
It is an audit failure rate: the transit authority wants it low.

What v3.2 adds over v3
1. EMPIRICAL FLEET GEOMETRY (Mercedes-Benz Citaro / eCitaro series, Daimler Buses):
     Class-I    : Citaro K 10.63 m, 2 doors  (x/L = 0.13, 0.58)
     Class-IIa  : eCitaro 12.14 m, 2 doors   (x/L = 0.11, 0.53)
     Class-IIb  : eCitaro 12.14 m, 3 doors   (x/L = 0.11, 0.53, 0.91)
     Class-IIIa : eCitaro G 18.13 m, 3 doors (x/L = 0.07, 0.36, 0.68)
     Class-IIIb : eCitaro G 18.13 m, 4 doors (x/L = 0.07, 0.36, 0.68, 0.91)
   Cabin width updated to standard European 2.55 m.
2. t_blind TIPPING-POINT SWEEP (--tblind-sweep):
   Sweeps deterministic t_blind from 0.0 s to 5.0 s (step 0.5 s) using paired
   Common Random Numbers (CRN) to pinpoint where dP_total flips sign.
3. INTERLOCKED BOARDING SIMULATION (--interlock):
   Evaluates sequenced aperture control D_open(t) where secondary doors (k > 0)
   remain locked until t = tau_interlock while inspectors board through door 0.
4. FIVE-SEED ROBUSTNESS CHECK (multiseed_results.csv):
   Evaluates all 15 baseline cells across 5 independent master seeds (20261009..20261013).

Usage:
  python qoyanshyq_monte_carlo.py                       # full suite (weidmann)
  python qoyanshyq_monte_carlo.py --friction alpha      # previous friction law
  python qoyanshyq_monte_carlo.py --n 50000 --sensitivity --zones
Requires only NumPy. Writes table1_results.csv, ablation_results.csv,
tblind_sweep.csv, interlock_results.csv, and multiseed_results.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace

import numpy as np

Z95 = 1.959963984540054
PLATFORM_Y = -2.0
W_V0, W_GAMMA, W_RHOMAX = 1.34, 1.913, 5.4   # Weidmann (1993)


# --------------------------------------------------------------------------- #
# Configuration (Daimler Buses / Mercedes-Benz eCitaro specs + behavioral assumptions)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VehicleClass:
    name: str
    length: float
    door_frac: tuple     # door x-positions / length; index 0 = ingress door
    partition: tuple     # opaque partition rectangle (x0, x1, y0, y1), metres


CLASSES = (
    VehicleClass("Class-I (Citaro K 10.63m 2D)", 10.63, (0.13, 0.58), (0.0, 2.0, 0.95, 2.55)),
    VehicleClass("Class-IIa (eCitaro 12.14m 2D)", 12.14, (0.11, 0.53), (0.0, 2.0, 0.95, 2.55)),
    VehicleClass("Class-IIb (eCitaro 12.14m 3D)", 12.14, (0.11, 0.53, 0.91), (0.0, 2.0, 0.95, 2.55)),
    VehicleClass("Class-IIIa (eCitaro G 18.13m 3D)", 18.13, (0.07, 0.36, 0.68), (0.0, 2.0, 0.95, 2.55)),
    VehicleClass("Class-IIIb (eCitaro G 18.13m 4D)", 18.13, (0.07, 0.36, 0.68, 0.91), (0.0, 2.0, 0.95, 2.55)),
)

POLICIES = ("single_door", "independent_random", "distinct_random")
MODES = ("baseline", "crowded", "crowded_coupled")

SCENARIOS = {
    "OFF":       dict(partition_on=False),
    "ON":        dict(partition_on=True),
    "pass_only": dict(partition_on=True, block_inspector_view=False),
    "insp_only": dict(partition_on=True, block_passenger_view=False),
}


@dataclass(frozen=True)
class Params:
    n_inspectors: int = 2
    width: float = 2.55
    p_right_seat: float = 0.5
    # partition switches
    partition_on: bool = True
    block_passenger_view: bool = True
    block_inspector_view: bool = True
    # passenger-side detection
    react_mean: float = 1.0
    react_sigma: float = 0.4
    lead_max: float = 3.0
    occ_lo: float = 1.0          # s, passenger delay when the platform is occluded
    occ_hi: float = 3.0
    # inspector-side blindness (ASSUMPTION -- swept in tblind_sweep)
    insp_blind_lo: float = 2.0   # s, inspector re-acquisition delay when blocked
    insp_blind_hi: float = 5.0
    # operational countermeasure: sequenced door interlock
    tau_interlock: float = 0.0   # s, secondary doors (k > 0) stay closed until t = tau_interlock
    # kinematics
    agent_speed: float = 1.3     # m/s free walking speed (agent)
    insp_speed: float = 0.8      # m/s inspector speed in a loaded cabin
    queue_mean: float = 6.0
    # crowd friction
    friction: str = "weidmann"   # "weidmann" | "alpha"
    alpha: float = 1.0           # alpha-law friction on the agent
    alpha_insp: float = 0.0      # alpha-law friction on inspectors
    couple_inspectors: bool = False   # weidmann: slow inspectors too
    rho_lo: float = 0.0
    rho_hi: float = 2.0
    dwell_lo: float = 12.0
    dwell_hi: float = 30.0


def apply_mode(p: Params, mode: str) -> Params:
    if mode == "baseline":
        return p
    if mode == "crowded":
        return replace(p, rho_lo=3.0, rho_hi=3.5)
    if mode == "crowded_coupled":
        return replace(p, rho_lo=3.0, rho_hi=3.5, alpha_insp=p.alpha, couple_inspectors=True)
    raise ValueError(mode)


@dataclass
class Trial:
    slip: np.ndarray
    right: np.ndarray
    frac: np.ndarray
    egress: np.ndarray


# --------------------------------------------------------------------------- #
# Crowd friction
# --------------------------------------------------------------------------- #
def weidmann_factor(rho):
    rho = np.maximum(np.asarray(rho, float), 1e-6)
    f = 1.0 - np.exp(-W_GAMMA * (1.0 / rho - 1.0 / W_RHOMAX))
    return np.clip(f, 0.02, 1.0)


def speed_factor(rho, p: Params, who: str):
    """Multiplier on free speed for 'agent' or 'inspector' at density rho."""
    if p.friction == "weidmann":
        return weidmann_factor(rho) if (who == "agent" or p.couple_inspectors) else np.ones_like(rho)
    a = p.alpha if who == "agent" else p.alpha_insp
    return 1.0 / (1.0 + a * rho)


def print_calibration(p: Params):
    print("\nCROWD-FRICTION CALIBRATION: walking speed (m/s) for a 1.3 m/s free walker")
    print("-" * 70)
    print(f"{'rho (ped/m^2)':>14}{'Weidmann 1993':>18}{'1/(1+alpha*rho)':>20}")
    for r in (0.5, 1.0, 2.0, 3.0, 3.5, 4.5):
        w = p.agent_speed * float(weidmann_factor(r))
        a = p.agent_speed / (1.0 + p.alpha * r)
        print(f"{r:>14.1f}{w:>18.3f}{a:>20.3f}")
    print(f"Active friction model: {p.friction}")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def segment_hits_rect(p0x, p0y, p1x, p1y, rect):
    """Vectorised slab test: does segment p0->p1 intersect the axis-aligned rectangle?"""
    x0, x1, y0, y1 = rect
    p0x, p0y, p1x, p1y = np.broadcast_arrays(*(np.asarray(a, float) for a in (p0x, p0y, p1x, p1y)))
    dx, dy = p1x - p0x, p1y - p0y
    tmin, tmax = np.zeros(dx.shape), np.ones(dx.shape)
    ok = np.ones(dx.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p0, d, lo, hi in ((p0x, dx, x0, x1), (p0y, dy, y0, y1)):
            parallel = np.abs(d) < 1e-12
            t1, t2 = (lo - p0) / d, (hi - p0) / d
            tlo = np.where(parallel, -np.inf, np.minimum(t1, t2))
            thi = np.where(parallel, np.inf, np.maximum(t1, t2))
            ok &= ~parallel | ((p0 >= lo) & (p0 <= hi))
            tmin, tmax = np.maximum(tmin, tlo), np.minimum(tmax, thi)
    return ok & (tmin <= tmax)


def assign_doors(policy, n, k, n_doors, rng):
    if policy == "single_door":
        return np.zeros((n, k), dtype=int)
    if policy == "independent_random":
        return rng.integers(0, n_doors, size=(n, k))
    if policy == "distinct_random":
        perm = np.argsort(rng.random((n, n_doors)), axis=1)
        return perm[:, np.arange(k) % n_doors]
    raise ValueError(policy)


# --------------------------------------------------------------------------- #
# Simulation core. The random-draw sequence does not depend on any partition
# switch, so scenarios run with the same seed are matched trial by trial.
# --------------------------------------------------------------------------- #
def simulate(vc: VehicleClass, policy: str, p: Params, n: int, rng) -> Trial:
    L = vc.length
    door_x = np.asarray(vc.door_frac) * L
    n_doors, k = door_x.size, p.n_inspectors
    idx = np.arange(n)

    right = rng.random(n) < p.p_right_seat
    xa = rng.uniform(0.1 * L, 0.9 * L, n)
    ya = np.where(right, rng.uniform(0.3, 0.9, n), rng.uniform(1.65, 2.25, n))

    doors = assign_doors(policy, n, k, n_doors, rng)
    dxj = door_x[doors]

    # passenger channel: seat -> platform spot (early warning), eq. (1)
    occ_pass = segment_hits_rect(xa[:, None], ya[:, None], dxj, PLATFORM_Y, vc.partition)
    occ_pass = occ_pass & p.partition_on & p.block_passenger_view
    mu = math.log(p.react_mean) - p.react_sigma ** 2 / 2
    t_react = rng.lognormal(mu, p.react_sigma, n)
    lead = rng.uniform(0.0, p.lead_max, n)
    t_occ = rng.uniform(p.occ_lo, p.occ_hi, (n, k))
    t_det = np.where(occ_pass,
                     t_react[:, None] + t_occ,
                     np.maximum(0.0, t_react - lead)[:, None]).min(axis=1)

    rho = rng.uniform(p.rho_lo, p.rho_hi, n)

    # egress: nearest door not held by an inspector, subject to tau_interlock
    blocked = np.zeros((n, n_doors), dtype=bool)
    blocked[idx[:, None], doors] = True
    v_agent = np.clip(rng.normal(p.agent_speed, 0.2, n), 0.5, 2.5)
    v_eff = v_agent * speed_factor(rho, p, "agent")

    dist_all = np.hypot(xa[:, None] - door_x[None, :], ya[:, None])
    t_walk_all = np.where(blocked, np.inf, dist_all / v_eff[:, None])
    t_egress = t_walk_all.min(axis=1)

    if p.tau_interlock > 0.0:
        # Door 0 (ingress) opens at t = 0; secondary doors (k >= 1) open at t = tau_interlock
        door_open_time = np.zeros((1, n_doors))
        door_open_time[:, 1:] = p.tau_interlock
        t_escape_per_door = np.maximum(t_det[:, None] + t_walk_all, door_open_time)
        t_escape = t_escape_per_door.min(axis=1)
    else:
        t_escape = t_det + t_egress

    # inspector reach time
    d_ins = np.hypot(xa[:, None] - dxj, ya[:, None])
    v_ins = np.clip(rng.normal(p.insp_speed, 0.15, (n, k)), 0.3, 1.5)
    v_ins = v_ins * speed_factor(rho, p, "inspector")[:, None]
    queue = rng.exponential(p.queue_mean, (n, k))
    t_dwell = rng.uniform(p.dwell_lo, p.dwell_hi, n)
    blind = rng.uniform(p.insp_blind_lo, p.insp_blind_hi, (n, k))   # drawn last: older streams unchanged

    # inspector channel: boarding point -> seat blocked by the partition?
    occ_insp = segment_hits_rect(dxj, 0.0, xa[:, None], ya[:, None], vc.partition)
    occ_insp = occ_insp & p.partition_on & p.block_inspector_view
    t_check = (d_ins / v_ins + queue + np.where(occ_insp, blind, 0.0)).min(axis=1)

    slip = (t_escape < t_check) & (t_escape <= t_dwell)
    return Trial(slip, right, xa / L, t_egress)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def summarize(slip):
    n = slip.size
    p_hat = float(slip.mean())
    sd = float(slip.std(ddof=1))
    se = sd / math.sqrt(n)
    denom = 1 + Z95 ** 2 / n
    centre = (p_hat + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p_hat * (1 - p_hat) / n + Z95 ** 2 / (4 * n * n)) / denom
    bm = [b.mean() for b in np.array_split(slip, 10)]
    return dict(n=n, p=p_hat, sd=sd, se=se, lo=centre - half, hi=centre + half,
                batch_sd=float(np.std(bm, ddof=1)))


def mean_ci(d):
    d = np.asarray(d, float)
    m = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(d.size)
    return m, m - Z95 * se, m + Z95 * se


def diff_ci(a, b):
    d = b["p"] - a["p"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return d, d - Z95 * se, d + Z95 * se


def run_cell(ci, pi, vc, policy, params, n, seed) -> Trial:
    return simulate(vc, policy, params, n, np.random.default_rng([seed, ci, pi]))


def pct(x, w=6):
    return f"{100 * x:{w}.2f}%"


def pp(x):
    return f"{100 * x:+6.2f}"


def ci_txt(lo, hi):
    return f"[{pp(lo).strip()}, {pp(hi).strip()}]"


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def main_table(params, n, seed, csv_path):
    results, rows = {}, []
    print(f"\nTABLE I -- P_slip by class and policy (baseline density, partition ON in both channels, "
          f"friction = {params.friction}, n = {n:,}/cell, seed = {seed})")
    print("-" * 109)
    print(f"{'Class':<32}{'Policy':<20}{'P_slip':>9}{'SD':>8}{'SE':>8}{'95% CI (Wilson)':>22}{'batch SD':>10}")
    print("-" * 109)
    for ci, vc in enumerate(CLASSES):
        for pi, pol in enumerate(POLICIES):
            s = summarize(run_cell(ci, pi, vc, pol, params, n, seed).slip)
            results[(ci, pol)] = s
            txt = f"[{pct(s['lo']).strip()}, {pct(s['hi']).strip()}]"
            print(f"{vc.name:<32}{pol:<20}{pct(s['p']):>9}{s['sd']:8.4f}{s['se']:8.4f}{txt:>22}{s['batch_sd']:10.4f}")
            rows.append(dict(vehicle_class=vc.name, doors=len(vc.door_frac), policy=pol,
                             friction=params.friction, **s))
        print()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {csv_path}")

    print("\n2-Door vs 3-Door Comparison on 12.14 m eCitaro (Class-IIa vs Class-IIb):")
    for pol in POLICIES:
        d, lo, hi = diff_ci(results[(1, pol)], results[(2, pol)])
        print(f"  {pol:<20}: 3D minus 2D = {pp(d)} pp   {ci_txt(lo, hi)}")

    print("\n3-Door vs 4-Door Comparison on 18.13 m eCitaro G (Class-IIIa vs Class-IIIb):")
    for pol in POLICIES:
        d, lo, hi = diff_ci(results[(3, pol)], results[(4, pol)])
        print(f"  {pol:<20}: 4D minus 3D = {pp(d)} pp   {ci_txt(lo, hi)}")

    print("\nCountermeasure effect: distinct_random minus single_door (pp, 95% CI)")
    for ci, vc in enumerate(CLASSES):
        d, lo, hi = diff_ci(results[(ci, "single_door")], results[(ci, "distinct_random")])
        print(f"  {vc.name:<32}{pp(d)} pp   {ci_txt(lo, hi)}")


def collect_ablation(params, mode, n, seed):
    base = apply_mode(params, mode)
    rows = []
    for ci, vc in enumerate(CLASSES):
        for pi, pol in enumerate(POLICIES):
            t = {name: run_cell(ci, pi, vc, pol, replace(base, **kw), n, seed) for name, kw in SCENARIOS.items()}
            assert all(np.array_equal(t["OFF"].right, x.right) for x in t.values())
            s = {k: v.slip.astype(float) for k, v in t.items()}
            drv, ker = ~t["OFF"].right, t["OFF"].right
            tot = mean_ci(s["ON"] - s["OFF"])
            pas = mean_ci(s["pass_only"] - s["OFF"])
            ins = mean_ci(s["insp_only"] - s["OFF"])
            itr = mean_ci(s["ON"] - s["pass_only"] - s["insp_only"] + s["OFF"])
            d_tot = mean_ci((s["ON"] - s["OFF"])[drv])
            d_ins = mean_ci((s["insp_only"] - s["OFF"])[drv])
            rows.append(dict(
                mode=mode, friction=params.friction, vehicle_class=vc.name, policy=pol,
                p_off=s["OFF"].mean(), p_pass_only=s["pass_only"].mean(),
                p_insp_only=s["insp_only"].mean(), p_on=s["ON"].mean(),
                d_total=tot[0], d_total_lo=tot[1], d_total_hi=tot[2],
                d_pass=pas[0], d_pass_lo=pas[1], d_pass_hi=pas[2],
                d_insp=ins[0], d_insp_lo=ins[1], d_insp_hi=ins[2],
                interaction=itr[0], interaction_lo=itr[1], interaction_hi=itr[2],
                kerb_on=s["ON"][ker].mean(), kerb_off=s["OFF"][ker].mean(),
                drv_on=s["ON"][drv].mean(), drv_off=s["OFF"][drv].mean(),
                drv_d_total=d_tot[0], drv_d_total_lo=d_tot[1], drv_d_total_hi=d_tot[2],
                drv_d_insp=d_ins[0], drv_d_insp_lo=d_ins[1], drv_d_insp_hi=d_ins[2]))
    return rows


def print_ablation(rows, mode, friction):
    print(f"\nPARTITION ABLATION -- mode: {mode}, friction: {friction}   "
          f"(dP > 0: partition helps the evader; paired 95% CI)")
    print("-" * 137)
    print(f"{'Class':<32}{'Policy':<20}{'P OFF':>9}{'P ON':>9}{'dP total':>10}{'95% CI (pp)':>20}"
          f"{'dP pass':>10}{'dP insp':>10}{'interact':>10}")
    print("-" * 137)
    for r in rows:
        print(f"{r['vehicle_class']:<32}{r['policy']:<20}{pct(r['p_off']):>9}{pct(r['p_on']):>9}"
              f"{pp(r['d_total']):>10}{ci_txt(r['d_total_lo'], r['d_total_hi']):>20}"
              f"{pp(r['d_pass']):>10}{pp(r['d_insp']):>10}{pp(r['interaction']):>10}")


def print_asymmetry(rows, mode, friction):
    print(f"\nRHT ASYMMETRY -- mode: {mode}, friction: {friction}   kerb (y 0.3-0.9) vs driver side (y 1.65-2.25)")
    print("-" * 129)
    print(f"{'Class':<32}{'Policy':<20}{'Kerb ON':>9}{'Kerb OFF':>10}{'Drv ON':>9}{'Drv OFF':>10}"
          f"{'dP drv':>9}{'95% CI (pp)':>20}{'dP drv insp':>13}")
    print("-" * 129)
    for r in rows:
        print(f"{r['vehicle_class']:<32}{r['policy']:<20}{pct(r['kerb_on']):>9}{pct(r['kerb_off']):>10}"
              f"{pct(r['drv_on']):>9}{pct(r['drv_off']):>10}{pp(r['drv_d_total']):>9}"
              f"{ci_txt(r['drv_d_total_lo'], r['drv_d_total_hi']):>20}{pp(r['drv_d_insp']):>13}")


def bottleneck_report(params, n, seed, modes):
    print(f"\nCROSS-CABIN BOTTLENECK -- single_door, partition ON, friction: {params.friction}")
    print("-" * 109)
    print(f"{'Class':<32}{'Mode':<17}{'t_egress kerb':>14}{'t_egress drv':>14}{'ratio':>7}{'P kerb':>10}{'P drv':>10}")
    print("-" * 109)
    for ci, vc in enumerate(CLASSES):
        for mode in modes:
            t = run_cell(ci, 0, vc, "single_door", apply_mode(params, mode), n, seed)
            ker, drv = t.right, ~t.right
            tk, td = float(t.egress[ker].mean()), float(t.egress[drv].mean())
            print(f"{vc.name:<32}{mode:<17}{tk:>12.2f} s{td:>12.2f} s{td / tk:>7.2f}"
                  f"{pct(t.slip[ker].mean()):>10}{pct(t.slip[drv].mean()):>10}")
        print()


def tblind_sweep_report(params, n, seed, out_csv="tblind_sweep.csv"):
    """Sweep deterministic t_blind from 0.0 to 5.0 s to find exact sign-flip tipping point."""
    print(f"\nSENSITIVITY SWEEP: t_blind TIPPING POINT (single_door, baseline density, CRN paired)")
    print("-" * 93)
    print(f"{'Class':<32}{'t_blind (s)':>12}{'dP_total (pp)':>15}{'95% CI (pp)':>22}{'Sign':>10}")
    print("-" * 93)

    grid = np.arange(0.0, 5.5, 0.5)
    rows = []
    for ci, vc in enumerate(CLASSES):
        s_off = run_cell(ci, 0, vc, "single_door", replace(params, partition_on=False), n, seed).slip.astype(float)
        prev_val, prev_tb = None, None
        tipping_str = "never flips"

        for tb in grid:
            q = replace(params, partition_on=True, insp_blind_lo=float(tb), insp_blind_hi=float(tb))
            s_on = run_cell(ci, 0, vc, "single_door", q, n, seed).slip.astype(float)
            m, lo, hi = mean_ci(s_on - s_off)
            sign_lbl = "EVADER (+)" if m > 0 else "INSP (-)"
            print(f"{vc.name:<32}{tb:>12.1f}{pp(m):>15}{ci_txt(lo, hi):>22}{sign_lbl:>10}")
            rows.append(dict(
                vehicle_class=vc.name, t_blind=float(tb),
                d_total=m, d_total_lo=lo, d_total_hi=hi
            ))
            if prev_val is not None and prev_val <= 0 < m:
                zero_tb = prev_tb + (0.0 - prev_val) / (m - prev_val) * (tb - prev_tb)
                tipping_str = f"~{zero_tb:.2f} s"
            prev_val, prev_tb = m, tb

        print(f"  >> Tipping point for {vc.name}: dP_total crosses 0 at t_blind = {tipping_str}\n")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


def interlock_report(params, n, seed, out_csv="interlock_results.csv"):
    """Evaluate Sequenced Aperture Control (Interlocked Boarding) across tau_interlock delays."""
    print(f"\nINTERLOCKED BOARDING EVALUATION (single_door, baseline density, partition ON)")
    print("-" * 97)
    print(f"{'Class':<32}{'tau_delay (s)':>14}{'P_slip':>10}{'95% Wilson CI':>22}{'Delta vs 0s (pp)':>18}")
    print("-" * 97)

    taus = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    rows = []
    for ci, vc in enumerate(CLASSES):
        base_s = None
        for tau in taus:
            q = replace(params, tau_interlock=tau)
            s = summarize(run_cell(ci, 0, vc, "single_door", q, n, seed).slip)
            if tau == 0.0:
                base_s = s
                delta_str = "baseline"
                d_val = 0.0
            else:
                d_val, lo, hi = diff_ci(base_s, s)
                delta_str = f"{pp(d_val).strip()} {ci_txt(lo, hi)}"
            txt = f"[{pct(s['lo']).strip()}, {pct(s['hi']).strip()}]"
            print(f"{vc.name:<32}{tau:>14.1f}{pct(s['p']):>10}{txt:>22}{delta_str:>18}")
            rows.append(dict(
                vehicle_class=vc.name, tau_interlock=tau,
                p_slip=s["p"], ci_lo=s["lo"], ci_hi=s["hi"], delta_vs_0=d_val
            ))
        print()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


def multiseed_report(params, n, base_seed, n_seeds=5, out_csv="multiseed_results.csv"):
    """Verify invariance to master seed across all 15 baseline cells."""
    seeds = [base_seed + i for i in range(n_seeds)]
    print(f"\nMULTI-SEED ROBUSTNESS CHECK ({n_seeds} seeds: {seeds[0]}..{seeds[-1]}, {n*n_seeds:,} trials/cell)")
    print("-" * 82)
    print(f"{'Class':<32}{'Policy':<20}{'Mean (%)':>12}{'SD (pp)':>12}")
    print("-" * 82)
    rows = []
    for ci, vc in enumerate(CLASSES):
        for pi, pol in enumerate(POLICIES):
            vals = [float(run_cell(ci, pi, vc, pol, params, n, s).slip.mean()) * 100.0 for s in seeds]
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            print(f"{vc.name:<32}{pol:<20}{mean_v:>11.2f}%{sd_v:>11.2f}")
            rows.append(dict(vehicle_class=vc.name, policy=pol, n_seeds=n_seeds,
                             total_trials=n * n_seeds, mean_pct=mean_v, sd_pp=sd_v))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}")


def zone_table(params, n, seed):
    print("\nCoverage-gap map (baseline, single_door, partition ON): P_slip by seat zone")
    print("-" * 75)
    print(f"{'Class':<32}{'Side':<8}{'Front':>10}{'Mid':>10}{'Rear':>10}")
    for ci, vc in enumerate(CLASSES):
        t = run_cell(ci, 0, vc, "single_door", params, n, seed)
        zone = np.digitize(t.frac, [0.3667, 0.6333])
        for name, mask in (("kerb", t.right), ("driver", ~t.right)):
            cells = [pct(t.slip[mask & (zone == z)].mean()) if (mask & (zone == z)).sum() > 30 else "   n/a"
                     for z in range(3)]
            print(f"{vc.name:<32}{name:<8}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}")


def sensitivity(params, n, seed):
    """dP_total (ON - OFF, pp) for single_door under +/-25% parameter changes."""
    knobs = ["insp_blind", "react_mean", "lead_max", "agent_speed", "insp_speed", "queue_mean", "dwell"]
    if params.friction == "alpha":
        knobs.append("alpha")
    print("\nSensitivity of dP_total (single_door, baseline density): parameter x0.75 / x1.25  (pp)")
    print("-" * 142)
    print(f"{'Parameter':<14}" + "".join(f"{c.name.split()[0]:>25}" for c in CLASSES))

    def dtot(i, vc, q):
        on = run_cell(i, 0, vc, "single_door", replace(q, partition_on=True), n, seed).slip.astype(float)
        off = run_cell(i, 0, vc, "single_door", replace(q, partition_on=False), n, seed).slip.astype(float)
        return float((on - off).mean())

    print(f"{'(baseline)':<14}" + "".join(f"{pp(dtot(i, vc, params)) + ' pp':>25}" for i, vc in enumerate(CLASSES)))
    for knob in knobs:
        cells = []
        for i, vc in enumerate(CLASSES):
            vals = []
            for mult in (0.75, 1.25):
                if knob == "dwell":
                    q = replace(params, dwell_lo=params.dwell_lo * mult, dwell_hi=params.dwell_hi * mult)
                elif knob == "insp_blind":
                    q = replace(params, insp_blind_lo=params.insp_blind_lo * mult,
                                insp_blind_hi=params.insp_blind_hi * mult)
                else:
                    q = replace(params, **{knob: getattr(params, knob) * mult})
                vals.append(dtot(i, vc, q))
            cells.append(f"{pp(vals[0]).strip()} / {pp(vals[1]).strip()}")
        print(f"{knob:<14}" + "".join(f"{c:>25}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--inspectors", type=int, default=2)
    ap.add_argument("--friction", choices=("weidmann", "alpha"), default="weidmann")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--out", default="table1_results.csv")
    ap.add_argument("--ablation-out", default="ablation_results.csv")
    ap.add_argument("--no-tblind-sweep", action="store_true")
    ap.add_argument("--no-interlock", action="store_true")
    ap.add_argument("--no-multiseed", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--zones", action="store_true")
    args = ap.parse_args()

    params = Params(n_inspectors=args.inspectors, friction=args.friction)
    print_calibration(params)
    main_table(params, args.n, args.seed, args.out)

    all_rows = []
    for mode in args.modes:
        rows = collect_ablation(params, mode, args.n, args.seed)
        all_rows += rows
        print_ablation(rows, mode, args.friction)
        print_asymmetry(rows, mode, args.friction)
    with open(args.ablation_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {args.ablation_out}")

    bottleneck_report(params, args.n, args.seed, args.modes)

    if not args.no_tblind_sweep:
        tblind_sweep_report(params, args.n, args.seed)

    if not args.no_interlock:
        interlock_report(params, args.n, args.seed)

    if not args.no_multiseed:
        multiseed_report(params, args.n, args.seed, n_seeds=5)

    if args.zones:
        zone_table(params, args.n, args.seed)
    if args.sensitivity:
        sensitivity(params, args.n, args.seed)

    print("\nReminder: behavioral parameters (including t_blind) remain assumptions; "
          "report results as model-conditional estimates anchored to Daimler Buses eCitaro geometry.")


if __name__ == "__main__":
    main()