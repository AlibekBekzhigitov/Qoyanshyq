#!/usr/bin/env python3
"""
Qoyanshyq v4a -- multi-evader, single-stop coverage audit (extends v3.1).
Bound directly to TrialModelBus/upd_montecarlo_bus_standart.py (real Mercedes-Benz Citaro specs).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "TrialModelBus", ROOT / "media", ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import upd_montecarlo_bus_standart as q31
except ModuleNotFoundError:
    import MonteCarlo as q31

Z95 = q31.Z95
PLATFORM_Y = q31.PLATFORM_Y

# Real Mercedes-Benz Citaro family seating plans
SEATS = {
    "Class-I": 26,      # Citaro K (10.63m 2D)
    "Class-IIa": 32,    # eCitaro (12.14m 2D)
    "Class-IIb": 29,    # eCitaro (12.14m 3D)
    "Class-IIIa": 45,   # eCitaro G (18.13m 3D)
    "Class-IIIb": 42,   # eCitaro G (18.13m 4D)
}
KERB_Y, DRV_Y = 0.60, 1.95
STAND_Y = (0.90, 1.65)
AISLE_W = 0.75


@dataclass(frozen=True)
class Profile:
    name: str
    react_mean: float   # s
    lead_max: float     # s, how early the platform is watched
    p_miss: float       # probability of never perceiving the inspectors
    beta: float         # seat-choice sharpness


PROFILES = {
    "novice":          Profile("novice", 2.0, 0.5, 0.20, 0.0),
    "typical":         Profile("typical", 1.0, 3.0, 0.00, 0.0),
    "vigilant":        Profile("vigilant", 0.7, 3.0, 0.00, 3.0),
    "vigilant_noseat": Profile("vigilant_noseat", 0.7, 3.0, 0.00, 0.0),
}


@dataclass(frozen=True)
class Cfg:
    n_evaders: int = 3
    load: float = 0.5
    positions: str = "seat"
    exclude_B: bool = False
    background: bool = True
    service_mean: float = 5.0
    service_sigma: float = 0.5
    check_policy: str = "nearest"
    mix: tuple = (("typical", 1.0),)


def in_B(x, y, rect):
    x0, x1, y0, y1 = rect
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def seat_table(vc, n_seats, p):
    L = vc.length
    n_k = n_seats // 2
    xs, ys = [], []
    for n_side, y in ((n_k, KERB_Y), (n_seats - n_k, DRV_Y)):
        for x in np.linspace(0.1 * L, 0.9 * L, n_side):
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    keep = ~in_B(xs, ys, vc.partition)
    xs, ys = xs[keep], ys[keep]
    door_x = np.asarray(vc.door_frac) * L
    vis = np.zeros(xs.size)
    for dx in door_x:
        hit = q31.segment_hits_rect(xs, ys, dx, PLATFORM_Y, vc.partition) if p.partition_on \
            else np.zeros(xs.size, bool)
        vis += ~hit
    vis /= door_x.size
    near = np.min(np.abs(xs[:, None] - door_x[None, :]), axis=1)
    score = vis + np.clip(1.0 - near / (L / 2.0), 0.0, 1.0)
    return xs, ys, score


def pick_target(cand, cx, cy, policy, rng, px, py):
    if policy == "random":
        return int(cand[rng.integers(cand.size)])
    if policy == "sweep":
        return int(cand[np.argmin(px[cand])])
    return int(cand[np.argmin(np.hypot(px[cand] - cx, py[cand] - cy))])


def one_trial(rng, vc, door_x, policy, p, cfg, seats, profs, probs):
    L, nd, k, E = vc.length, door_x.size, p.n_inspectors, cfg.n_evaders
    sx, sy, sscore = seats
    rho = rng.uniform(p.rho_lo, p.rho_hi)
    px, py = [], []

    free = np.ones(sx.size, bool)
    if cfg.background:
        occ = rng.random(sx.size) < cfg.load
        free &= ~occ
        px += list(sx[occ])
        py += list(sy[occ])
        n_st = int(round(rho * AISLE_W * L))
        if n_st > 0:
            px += list(rng.uniform(2.0, 0.95 * L, n_st))
            py += list(rng.uniform(*STAND_Y, n_st))
    n_bg = len(px)

    pidx = rng.choice(len(profs), size=E, p=probs)
    for e in range(E):
        pr = profs[pidx[e]]
        if cfg.positions == "continuous":
            while True:
                right = rng.random() < p.p_right_seat
                x = rng.uniform(0.1 * L, 0.9 * L)
                y = rng.uniform(0.3, 0.9) if right else rng.uniform(1.65, 2.25)
                if not (cfg.exclude_B and in_B(x, y, vc.partition)):
                    break
        else:
            idx = np.flatnonzero(free)
            if idx.size:
                w = np.exp(pr.beta * (sscore[idx] - sscore[idx].max()))
                w /= w.sum()
                j = int(idx[rng.choice(idx.size, p=w)])
                free[j] = False
                x, y = sx[j], sy[j]
            else:
                x, y = rng.uniform(2.0, 0.95 * L), rng.uniform(*STAND_Y)
        px.append(x)
        py.append(y)
    px, py = np.array(px), np.array(py)
    M = px.size
    ev_ids = np.arange(n_bg, n_bg + E)

    doors = q31.assign_doors(policy, 1, k, nd, rng)[0]
    held = np.zeros(nd, bool)
    held[doors] = True
    dxj = door_x[doors]
    v_ins = np.clip(rng.normal(p.insp_speed, 0.15, k), 0.3, 1.5) * float(q31.speed_factor(rho, p, "inspector"))
    queue = rng.exponential(p.queue_mean, k)
    t_dwell = rng.uniform(p.dwell_lo, p.dwell_hi)
    f_agent = float(q31.speed_factor(rho, p, "agent"))
    open_t = np.zeros(nd)
    open_t[1:] = p.tau_interlock

    t_esc_all = np.full(M, np.inf)
    for e in range(E):
        pr = profs[pidx[e]]
        i = ev_ids[e]
        mu = math.log(pr.react_mean) - p.react_sigma ** 2 / 2
        t_r = rng.lognormal(mu, p.react_sigma)
        lead = rng.uniform(0.0, pr.lead_max)
        t_occ = rng.uniform(p.occ_lo, p.occ_hi, k)
        miss = rng.random() < pr.p_miss
        v = np.clip(rng.normal(p.agent_speed, 0.2), 0.5, 2.5) * f_agent
        if miss:
            continue
        occ_p = q31.segment_hits_rect(px[i], py[i], dxj, PLATFORM_Y, vc.partition) \
            & p.partition_on & p.block_passenger_view
        t_det = float(np.where(occ_p, t_r + t_occ, max(0.0, t_r - lead)).min())
        w = np.hypot(px[i] - door_x, py[i]) / v
        w[held] = np.inf
        if p.tau_interlock > 0.0:
            t_esc_all[i] = float(np.maximum(t_det + w, open_t).min())
        else:
            t_esc_all[i] = t_det + float(w.min())

    arrive = np.full(M, np.inf)
    pos = [(float(dxj[j]), 0.0) for j in range(k)]
    ready = [float(q) for q in queue]
    active = [True] * k
    smu = math.log(cfg.service_mean) - cfg.service_sigma ** 2 / 2 if cfg.service_mean > 0 else 0.0
    while True:
        js = [j for j in range(k) if active[j] and ready[j] <= t_dwell]
        if not js:
            break
        j = min(js, key=lambda jj: ready[jj])
        t0, (cx, cy) = ready[j], pos[j]
        aboard = t_esc_all > t0
        cand = np.flatnonzero(aboard & np.isinf(arrive))
        if cand.size == 0:
            cl = np.flatnonzero(aboard & np.isfinite(arrive))
            cand = cl[t0 + np.hypot(px[cl] - cx, py[cl] - cy) / v_ins[j] < arrive[cl]] if cl.size else cl
        if cand.size == 0:
            active[j] = False
            continue
        tg = pick_target(cand, cx, cy, cfg.check_policy, rng, px, py)
        d = math.hypot(px[tg] - cx, py[tg] - cy)
        occ_i = bool(q31.segment_hits_rect(cx, cy, px[tg], py[tg], vc.partition)) \
            and p.partition_on and p.block_inspector_view
        blind = rng.uniform(p.insp_blind_lo, p.insp_blind_hi)
        ta = t0 + d / v_ins[j] + (blind if occ_i else 0.0)
        if ta >= arrive[tg]:
            active[j] = False
            continue
        arrive[tg] = ta
        left = t_esc_all[tg] < ta
        s = 0.0 if (left or cfg.service_mean <= 0) else rng.lognormal(smu, cfg.service_sigma)
        ready[j] = ta + s
        pos[j] = (float(px[tg]), float(py[tg]))

    te, ar = t_esc_all[ev_ids], arrive[ev_ids]
    slip = (te <= t_dwell) & (te < ar)
    chk = (ar <= t_dwell) & (ar <= te)
    n_checked = int(np.sum((arrive <= t_dwell) & (arrive <= t_esc_all)))
    return slip, chk, ~slip & ~chk, pidx, n_checked, M


def run_cell(ci, pi, vc, policy, p, cfg, n, seed, tag=0):
    rng = np.random.default_rng([seed, ci, pi, tag])
    door_x = np.asarray(vc.door_frac) * vc.length
    n_seats = SEATS[vc.name.split()[0]]
    seats = seat_table(vc, n_seats, p)
    names = [m[0] for m in cfg.mix]
    probs = np.array([m[1] for m in cfg.mix], float)
    probs /= probs.sum()
    profs = [PROFILES[nm] for nm in names]
    E = cfg.n_evaders
    S, C, U = (np.zeros((n, E), bool) for _ in range(3))
    P = np.zeros((n, E), int)
    NC, MM = np.zeros(n), np.zeros(n)
    for t in range(n):
        S[t], C[t], U[t], P[t], NC[t], MM[t] = one_trial(rng, vc, door_x, policy, p, cfg, seats, profs, probs)
    return dict(slip=S, chk=C, abd=U, prof=P, nchk=NC, M=MM, names=names)


def cluster_ci(per_trial):
    m = float(np.mean(per_trial))
    se = float(np.std(per_trial, ddof=1)) / math.sqrt(len(per_trial))
    return m, m - Z95 * se, m + Z95 * se


def summarize(res):
    slip, chk, abd = res["slip"], res["chk"], res["abd"]
    m, lo, hi = cluster_ci(slip.mean(axis=1))
    return dict(p_slip=m, lo=lo, hi=hi, p_checked=float(chk.mean()), p_aboard=float(abd.mean()),
                p_any_slip=float(slip.any(axis=1).mean()), p_all_caught=float((~slip).all(axis=1).mean()),
                mean_checks=float(res["nchk"].mean()), mean_pass=float(res["M"].mean()))


def pct(x):
    return f"{100 * x:6.2f}"


def regression(args):
    print("\nREGRESSION: v4a engine (N=1, no background, continuous positions, service=0) vs v3.1 simulate()")
    print("-" * 100)
    print(f"{'Class':<14}{'Policy':<20}{'Partition':<10}{'v3.1 %':>9}{'v4a %':>9}{'diff pp':>9}{'z':>7}{'ok':>5}")
    cfg = Cfg(n_evaders=1, positions="continuous", background=False, service_mean=0.0)
    n_ok = n_tot = 0
    for part in (True, False):
        p = q31.Params(partition_on=part)
        for ci, vc in enumerate(q31.CLASSES):
            for pi, pol in enumerate(q31.POLICIES):
                a = q31.run_cell(ci, pi, vc, pol, p, args.n, args.seed).slip.astype(float)
                res = run_cell(ci, pi, vc, pol, p, cfg, args.n, args.seed, tag=7)
                b = res["slip"][:, 0].astype(float)
                ma, mb = a.mean(), b.mean()
                se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
                z = (mb - ma) / se if se > 0 else 0.0
                ok = abs(z) < 3.0
                n_ok += ok
                n_tot += 1
                print(f"{vc.name.split()[0]:<14}{pol:<20}{'ON' if part else 'OFF':<10}{pct(ma):>9}{pct(mb):>9}"
                      f"{100 * (mb - ma):>+9.2f}{z:>7.2f}{'yes' if ok else 'NO':>5}")
    print(f"\nRegression: {n_ok}/{n_tot} cells within |z| < 3.")
    return n_ok == n_tot


def parse_mix(s):
    return tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in s.split(","))


def write_csv(path, rows):
    out_path = Path(__file__).resolve().parent / path
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_path}")


def base_params(args):
    return q31.apply_mode(q31.Params(), args.mode)


def exp_main(args):
    p = base_params(args)
    mix = parse_mix(args.mix)
    print(f"\nMAIN: E={args.evaders} evaders, mix={args.mix}, mode={args.mode}, policy={args.check_policy}, n={args.n}")
    print("-" * 112)
    print(f"{'Class':<12}{'Door policy':<19}{'Load':>5}{'P_slip %':>10}{'95% CI':>18}{'checked %':>10}{'aboard %':>9}"
          f"{'P(any slip)':>12}{'checks/stop':>12}")
    rows = []
    for ci, vc in enumerate(q31.CLASSES):
        for pi, pol in enumerate(("single_door", "distinct_random")):
            for load in args.loads:
                cfg = Cfg(n_evaders=args.evaders, load=load, mix=mix, check_policy=args.check_policy,
                          service_mean=args.service)
                s = summarize(run_cell(ci, pi, vc, pol, p, cfg, args.n, args.seed))
                print(f"{vc.name.split()[0]:<12}{pol:<19}{load:>5.1f}{pct(s['p_slip']):>10}"
                      f"{'[' + pct(s['lo']).strip() + ', ' + pct(s['hi']).strip() + ']':>18}{pct(s['p_checked']):>10}"
                      f"{pct(s['p_aboard']):>9}{pct(s['p_any_slip']):>12}{s['mean_checks']:>12.1f}")
                rows.append(dict(vehicle_class=vc.name, door_policy=pol, load=load, evaders=args.evaders,
                                 mix=args.mix, mode=args.mode, check_policy=args.check_policy, n=args.n, **s))
    write_csv("v4a_main.csv", rows)


def exp_profiles(args):
    p = base_params(args)
    print(f"\nPROFILES: pure populations, E={args.evaders}, load={args.loads[0]}, mode={args.mode}, "
          f"door policy=single_door, n={args.n}")
    print("-" * 80)
    print(f"{'Class':<12}{'Profile':<18}{'P_slip %':>10}{'95% CI':>18}{'checked %':>10}{'aboard %':>9}")
    rows = []
    for ci, vc in enumerate(q31.CLASSES):
        for nm in ("novice", "typical", "vigilant_noseat", "vigilant"):
            cfg = Cfg(n_evaders=args.evaders, load=args.loads[0], mix=((nm, 1.0),),
                      check_policy=args.check_policy, service_mean=args.service)
            s = summarize(run_cell(ci, 0, vc, "single_door", p, cfg, args.n, args.seed))
            print(f"{vc.name.split()[0]:<12}{nm:<18}{pct(s['p_slip']):>10}"
                  f"{'[' + pct(s['lo']).strip() + ', ' + pct(s['hi']).strip() + ']':>18}"
                  f"{pct(s['p_checked']):>10}{pct(s['p_aboard']):>9}")
            rows.append(dict(vehicle_class=vc.name, profile=nm, load=args.loads[0], evaders=args.evaders,
                             mode=args.mode, n=args.n, **s))
    write_csv("v4a_profiles.csv", rows)


def exp_policies(args):
    p = base_params(args)
    print(f"\nINSPECTOR CHECK POLICY: E={args.evaders}, load={args.loads[0]}, mix={args.mix}, mode={args.mode}, "
          f"door policy=single_door, n={args.n}")
    print("-" * 80)
    print(f"{'Class':<12}{'Check policy':<14}{'P_slip %':>10}{'95% CI':>18}{'checked %':>10}{'checks/stop':>12}")
    rows = []
    mix = parse_mix(args.mix)
    for ci, vc in enumerate(q31.CLASSES):
        for cp in ("nearest", "random", "sweep"):
            cfg = Cfg(n_evaders=args.evaders, load=args.loads[0], mix=mix, check_policy=cp,
                      service_mean=args.service)
            s = summarize(run_cell(ci, 0, vc, "single_door", p, cfg, args.n, args.seed))
            print(f"{vc.name.split()[0]:<12}{cp:<14}{pct(s['p_slip']):>10}"
                  f"{'[' + pct(s['lo']).strip() + ', ' + pct(s['hi']).strip() + ']':>18}"
                  f"{pct(s['p_checked']):>10}{s['mean_checks']:>12.1f}")
            rows.append(dict(vehicle_class=vc.name, check_policy=cp, load=args.loads[0], mix=args.mix,
                             mode=args.mode, n=args.n, **s))
    write_csv("v4a_check_policies.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--evaders", type=int, default=3)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--mix", default="novice=0.3,typical=0.5,vigilant=0.2")
    ap.add_argument("--mode", choices=q31.MODES, default="baseline")
    ap.add_argument("--check-policy", choices=("nearest", "random", "sweep"), default="nearest")
    ap.add_argument("--service", type=float, default=5.0, help="mean s per ticket check")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--profiles", action="store_true")
    ap.add_argument("--check-policies", action="store_true")
    args = ap.parse_args()

    if not (args.regression or args.main or args.profiles or args.check_policies):
        args.main = True
        args.profiles = True
        args.check_policies = True

    print(f"Running Qoyanshyq v4a | Citaro SEATS={SEATS} | service_mean={args.service}s")
    if args.regression:
        regression(args)
    if args.main:
        exp_main(args)
    if args.profiles:
        exp_profiles(args)
    if args.check_policies:
        exp_policies(args)


if __name__ == "__main__":
    main()