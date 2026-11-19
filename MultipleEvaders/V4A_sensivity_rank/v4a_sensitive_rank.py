#!/usr/bin/env python3
"""
Qoyanshyq v4a -- sensitivity ranking v2 (tornado with error bars + bulkhead panel).

What changed versus the first ranking script
--------------------------------------------
1. NOISE IS SHOWN.  Every bar end carries a 95% error bar (cluster-robust, per trial).  A swing
   whose 95% CI includes 0 is drawn hatched and tagged "n.s." -- it is NOT ranked as a driver.
2. COMPARABLE RANGES.  Continuous parameters are all perturbed by -30% / +30% around the baseline;
   shares move by +-0.10; evader count 2 vs 4.  (A wide range on one parameter guarantees it "wins".)
3. THE 'SKILL MIX' LEVER IS SPLIT into its real components: novice share, vigilant share,
   novice p_miss, reaction-time scale, and seat-choice sharpness (beta).
4. TWO OUTPUTS, TWO RANKINGS: P_slip (escaped at this stop) and P_checked (checked at this stop)
   are ranked separately, because the drivers differ.
5. BULKHEAD PANEL: the sensitivity of the PARTITION EFFECT dP = P(ON) - P(OFF) to t_blind, which
   P_slip alone hides.
6. Runs in parallel (macOS-safe 'spawn'); use --workers 1 to debug.

Place next to qoyanshyq_v4a_newscale.py and upd_montecarlo_bus_standart.py.
  python sensitivity_ranking_v2.py --n 10000                  # ~ 8 min on 2 cores
  python sensitivity_ranking_v2.py --class IIIa --policy distinct_random --n 10000
Outputs (next to this script, or --outdir): v4a_sens_v2_<class>_<policy>.csv / .png,
                                            v4a_bulkhead_dP_<class>_<policy>.csv / .png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (
    HERE,
    HERE.parent / "V4A_slipscale",
    HERE.parent,
    HERE.parents[1],
    HERE.parents[1] / "TrialModelBus",
    HERE.parents[1] / "media",
):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import qoyanshyq_v4a_newscale as v4a
except ModuleNotFoundError:
    try:
        import qoyanshyq_v4a_newscale_fixed as v4a
    except ModuleNotFoundError:
        import qoyanshyq_v4a as v4a
q31 = v4a.q31

Z95 = 1.959963984540054
CAT = {
    "ops":   ("Operational logs (APC/AVL, aggregated; low privacy risk)", "#2980b9"),
    "depot": ("Depot timing trial (volunteers + consent)", "#27ae60"),
    "lit":   ("Literature-anchored (published walking speeds)", "#8e44ad"),
    "behav": ("Behavioural assumption (not directly observable)", "#e67e22"),
}

# key, label, category, low value, high value   (value meaning is defined in apply())
PERTURBATIONS = [
    ("dwell",        "Stop dwell window",              "ops",   0.70, 1.30),   # multiplier
    ("load",         "Seated load factor",             "ops",   0.35, 0.65),   # absolute (base 0.50)
    ("service",      "Ticket-check service time",      "depot", 0.70, 1.30),   # multiplier
    ("queue",        "Inspector boarding delay",       "depot", 0.70, 1.30),   # multiplier
    ("insp_speed",   "Inspector walking speed",        "depot", 0.70, 1.30),   # multiplier
    ("t_blind",      "Inspector blind delay t_blind",  "depot", 0.70, 1.30),   # multiplier on [lo, hi]
    ("walk_speed",   "Evader free walking speed",      "lit",   0.70, 1.30),   # multiplier
    ("n_evaders",    "Evaders per stop",               "behav", 2, 4),          # absolute (base 3)
    ("novice_share", "Novice share of evaders",        "behav", 0.20, 0.40),   # absolute (base 0.30)
    ("vig_share",    "Vigilant share of evaders",      "behav", 0.10, 0.30),   # absolute (base 0.20)
    ("p_miss",       "Novice miss probability",        "behav", 0.14, 0.26),   # absolute (base 0.20)
    ("react",        "Evader reaction-time scale",     "behav", 0.70, 1.30),   # multiplier, all profiles
    ("beta",         "Vigilant seat-choice sharpness", "behav", 2.1, 3.9),     # absolute (base 3.0)
]


def apply(key, val, p, cfg):
    """Return (params, cfg, profile_overrides) for one perturbation."""
    ov = {}
    if key == "dwell":
        p = replace(p, dwell_lo=p.dwell_lo * val, dwell_hi=p.dwell_hi * val)
    elif key == "load":
        cfg = replace(cfg, load=val)
    elif key == "service":
        cfg = replace(cfg, service_mean=cfg.service_mean * val)
    elif key == "queue":
        p = replace(p, queue_mean=p.queue_mean * val)
    elif key == "insp_speed":
        p = replace(p, insp_speed=p.insp_speed * val)
    elif key == "t_blind":
        p = replace(p, insp_blind_lo=p.insp_blind_lo * val, insp_blind_hi=p.insp_blind_hi * val)
    elif key == "walk_speed":
        p = replace(p, agent_speed=p.agent_speed * val)
    elif key == "n_evaders":
        cfg = replace(cfg, n_evaders=int(val))
    elif key == "novice_share":
        cfg = replace(cfg, mix=(("novice", val), ("typical", 1.0 - val - 0.20), ("vigilant", 0.20)))
    elif key == "vig_share":
        cfg = replace(cfg, mix=(("novice", 0.30), ("typical", 1.0 - 0.30 - val), ("vigilant", val)))
    elif key == "p_miss":
        ov["novice"] = replace(v4a.PROFILES["novice"], p_miss=val)
    elif key == "react":
        for nm in ("novice", "typical", "vigilant"):
            ov[nm] = replace(v4a.PROFILES[nm], react_mean=v4a.PROFILES[nm].react_mean * val)
    elif key == "beta":
        ov["vigilant"] = replace(v4a.PROFILES["vigilant"], beta=val)
    else:
        raise KeyError(key)
    return p, cfg, ov


def stats(res):
    s, c = res["slip"].mean(axis=1), res["chk"].mean(axis=1)
    n = len(s)
    return dict(slip=100 * s.mean(), slip_se=100 * s.std(ddof=1) / math.sqrt(n),
                chk=100 * c.mean(), chk_se=100 * c.std(ddof=1) / math.sqrt(n),
                aboard=100 * float(res["abd"].mean()))


def run_task(task):
    """Worker: one engine run.  Must be a module-level function (picklable)."""
    name, key, val, p, cfg, ci, pi, policy, n, seed, tag = task
    ov = {}
    if key is not None:
        p, cfg, ov = apply(key, val, p, cfg)
    saved = dict(v4a.PROFILES)
    try:
        v4a.PROFILES.update(ov)
        vc = q31.CLASSES[ci]
        res = v4a.run_cell(ci, pi, vc, policy, p, cfg, n=n, seed=seed, tag=tag)
        return name, stats(res)
    finally:
        v4a.PROFILES.clear()
        v4a.PROFILES.update(saved)


def execute(tasks, workers):
    out = {}
    if workers <= 1:
        for t in tasks:
            nm, st = run_task(t)
            out[nm] = st
            print(f"  done {nm:<22} slip {st['slip']:6.2f}  chk {st['chk']:5.2f}", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nm, st in ex.map(run_task, tasks):
            out[nm] = st
            print(f"  done {nm:<22} slip {st['slip']:6.2f}  chk {st['chk']:5.2f}", flush=True)
    return out


# --------------------------------------------------------------------------- #
def swing(lo, hi, m):
    d = hi[m] - lo[m]
    se = math.hypot(lo[m + "_se"], hi[m + "_se"])
    return d, Z95 * se, abs(d) > Z95 * se


def tornado(ax, rows, m, base, title, xlabel):
    rows = sorted(rows, key=lambda r: abs(r[m + "_swing"]), reverse=True)
    ypos = np.arange(len(rows))[::-1]
    allv = [base[m] - Z95 * base[m + "_se"], base[m] + Z95 * base[m + "_se"]]
    for y, r in zip(ypos, rows):
        col = CAT[r["cat"]][1]
        sig = r[m + "_sig"]
        lo, hi = r[m + "_lo"], r[m + "_hi"]
        for v, a in ((lo, 0.45), (hi, 0.95)):
            ax.barh(y, v - base[m], left=base[m], height=0.6, color=col if sig else "#bdbdbd",
                    alpha=a if sig else 0.55, edgecolor=col if sig else "#888888", linewidth=1.1,
                    hatch=None if sig else "///")
        ax.errorbar([lo, hi], [y, y], xerr=[Z95 * r[m + "_lo_se"], Z95 * r[m + "_hi_se"]], fmt="none",
                    ecolor="#222222", capsize=2.5, linewidth=1)
        a, b = lo - Z95 * r[m + "_lo_se"], hi + Z95 * r[m + "_hi_se"]
        left, right = min(a, b), max(lo + Z95 * r[m + "_lo_se"], hi + Z95 * r[m + "_hi_se"])
        allv += [left, right]
        pad = 0.012 * (max(allv) - min(allv) + 1)
        ax.text(min(lo, hi) - Z95 * r[m + "_lo_se"] - pad * 4, y, f"{min(lo, hi):.1f}", va="center", ha="right", fontsize=8)
        ax.text(max(lo, hi) + Z95 * r[m + "_hi_se"] + pad * 4, y, f"{max(lo, hi):.1f}", va="center", ha="left", fontsize=8)
    ax.axvspan(base[m] - Z95 * base[m + "_se"], base[m] + Z95 * base[m + "_se"], color="#999999", alpha=0.3, lw=0)
    ax.axvline(base[m], color="#111111", linestyle="--", linewidth=1.1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['label']}  [{r['range']}]" + ("" if r[m + "_sig"] else "  (n.s.)") for r in rows], fontsize=8.8)
    span = max(allv) - min(allv)
    ax.set_xlim(min(allv) - 0.24 * span, max(allv) + 0.20 * span)   # room for the value labels
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return rows


def make_figures(rows, base, scen, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.8), dpi=250)
    tornado(a1, rows, "slip", base, "A. P_slip: escaped at this stop", "P_slip (%)")
    tornado(a2, rows, "chk", base, "B. P_checked: checked at this stop", "P_checked (%)")
    handles = [Patch(facecolor=c, edgecolor=c, label=lab) for lab, c in CAT.values()]
    handles += [Patch(facecolor="#bdbdbd", edgecolor="#888888", hatch="///", label="n.s.: swing CI includes 0"),
                Patch(facecolor="#999999", alpha=0.3, label="baseline 95% CI (grey band)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.075, "Light bar = low setting, dark bar = high setting; error bars = 95% CI of each run.",
             ha="center", fontsize=8.5, style="italic")
    fig.suptitle(f"Qoyanshyq v4a sensitivity ranking: {scen}", fontsize=12.5, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved {out_png}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] saved {path}")


def bulkhead_panel(args, base_p, base_cfg, ci, pi, policy, scen, workers):
    grid = [0.0, 1.0, 2.0, 3.5, 5.0]
    tasks = [("OFF", None, None, replace(base_p, partition_on=False), base_cfg, ci, pi, policy, args.n, args.seed, 900),
             ("ON_default", None, None, base_p, base_cfg, ci, pi, policy, args.n, args.seed, 901)]
    for i, t in enumerate(grid):
        tasks.append((f"ON_tblind={t}", None, None, replace(base_p, insp_blind_lo=t, insp_blind_hi=t), base_cfg,
                      ci, pi, policy, args.n, args.seed, 910 + i))
    print("\nBULKHEAD PANEL (dP = P_ON - P_OFF vs deterministic t_blind)")
    out = execute(tasks, workers)
    off = out["OFF"]
    rows = []
    for nm in ["ON_default"] + [f"ON_tblind={t}" for t in grid]:
        d = out[nm]["slip"] - off["slip"]
        ci95 = Z95 * math.hypot(out[nm]["slip_se"], off["slip_se"])
        rows.append(dict(setting=nm, dP_pp=round(d, 3), ci95_pp=round(ci95, 3), p_on=round(out[nm]["slip"], 2),
                         p_off=round(off["slip"], 2)))
        print(f"  {nm:<16} dP = {d:+6.2f} pp  +-{ci95:4.2f}")
    write_csv(args.outdir / f"v4a_bulkhead_dP_{scen_tag(args)}.csv", rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=250)
    xs = grid
    ys = [r["dP_pp"] for r in rows[1:]]
    es = [r["ci95_pp"] for r in rows[1:]]
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#c0392b", capsize=3, linewidth=1.6, label="dP at fixed t_blind")
    ax.axhline(0, color="#111111", linewidth=1)
    d0, e0 = rows[0]["dP_pp"], rows[0]["ci95_pp"]
    ax.axhspan(d0 - e0, d0 + e0, color="#2980b9", alpha=0.25, label=f"default U(2,5) s: {d0:+.2f} pp")
    ax.axvspan(2, 5, color="#999999", alpha=0.12, label="default t_blind range")
    ax.set_xlabel("Deterministic inspector re-acquisition delay t_blind (s)", fontweight="bold")
    ax.set_ylabel("Bulkhead effect dP (pp)  [>0: partition raises P_slip]", fontweight="bold")
    ax.set_title(f"Bulkhead effect vs t_blind: {scen}", fontsize=10.5, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    out_png = args.outdir / f"v4a_bulkhead_dP_{scen_tag(args)}.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved {out_png}")


def scen_tag(args):
    return f"{args.cls}_{args.policy}_{args.mode}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10_000, help="trials per run (10000 ~ SD 0.17 pp; 20000 ~ 0.12 pp)")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--class", dest="cls", default="IIb", help="IIb, IIIa, ... (substring of the class name)")
    ap.add_argument("--policy", default="single_door", choices=list(q31.POLICIES))
    ap.add_argument("--mode", default="baseline", choices=list(q31.MODES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument("--skip-bulkhead", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ci = [i for i, c in enumerate(q31.CLASSES) if f"Class-{args.cls}" == c.name.split()[0]][0]
    pi = list(q31.POLICIES).index(args.policy)
    vc = q31.CLASSES[ci]
    base_p = q31.apply_mode(q31.Params(partition_on=True), args.mode)
    base_cfg = v4a.Cfg(n_evaders=3, load=0.50, positions="seat", background=True, service_mean=5.0,
                       service_sigma=0.5, check_policy="nearest",
                       mix=(("novice", 0.3), ("typical", 0.5), ("vigilant", 0.2)))
    scen = f"{vc.name}, {args.policy}, {args.mode}, load 0.50, E=3, n={args.n:,}/run"
    print(f"Scenario: {scen}\nSEATS used: {v4a.SEATS}\nWorkers: {args.workers}\n")

    tasks = [("BASE", None, None, base_p, base_cfg, ci, pi, args.policy, args.n, args.seed, 0)]
    for i, (key, label, cat, lo, hi) in enumerate(PERTURBATIONS):
        tasks.append((f"{key}:low", key, lo, base_p, base_cfg, ci, pi, args.policy, args.n, args.seed, 100 + 2 * i))
        tasks.append((f"{key}:high", key, hi, base_p, base_cfg, ci, pi, args.policy, args.n, args.seed, 101 + 2 * i))
    out = execute(tasks, args.workers)
    base = out["BASE"]
    print(f"\nBaseline: P_slip {base['slip']:.2f}% (+-{Z95 * base['slip_se']:.2f}), "
          f"P_checked {base['chk']:.2f}% (+-{Z95 * base['chk_se']:.2f}), still aboard {base['aboard']:.2f}%")

    rows = []
    for key, label, cat, lo, hi in PERTURBATIONS:
        L, H = out[f"{key}:low"], out[f"{key}:high"]
        r = dict(key=key, label=label, cat=cat, range=f"{lo:g} to {hi:g}" if isinstance(lo, (int, float)) else "")
        for m in ("slip", "chk"):
            d, ci95, sig = swing(L, H, m)
            r.update({m + "_lo": L[m], m + "_hi": H[m], m + "_lo_se": L[m + "_se"], m + "_hi_se": H[m + "_se"],
                      m + "_swing": d, m + "_ci95": ci95, m + "_sig": sig})
        rows.append(r)

    for m, name in (("slip", "P_slip"), ("chk", "P_checked")):
        print(f"\nRANKING by {name} swing (high - low, pp; * = distinguishable from 0 at 95%)")
        print(f"{'#':<3}{'Parameter':<34}{'Category':<9}{'low':>8}{'high':>8}{'swing':>8}{'+-95%':>7}")
        for k, r in enumerate(sorted(rows, key=lambda r: abs(r[m + '_swing']), reverse=True), 1):
            print(f"{k:<3}{r['label']:<34}{r['cat']:<9}{r[m + '_lo']:>8.2f}{r[m + '_hi']:>8.2f}"
                  f"{r[m + '_swing']:>+8.2f}{r[m + '_ci95']:>7.2f} {'*' if r[m + '_sig'] else ''}")

    csv_rows = []
    for r in rows:
        csv_rows.append({"parameter": r["label"], "range": r["range"], "category": CAT[r["cat"]][0],
                         **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                            if k.startswith(("slip_", "chk_"))}})
    write_csv(args.outdir / f"v4a_sens_v2_{scen_tag(args)}.csv", csv_rows)
    make_figures(rows, base, scen, args.outdir / f"v4a_sens_v2_{scen_tag(args)}.png")

    if not args.skip_bulkhead:
        bulkhead_panel(args, base_p, base_cfg, ci, pi, args.policy, scen, args.workers)


if __name__ == "__main__":
    main()