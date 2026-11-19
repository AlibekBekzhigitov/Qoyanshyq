#!/usr/bin/env python3
"""
Qoyanshyq v4a -- Rigorous Sensitivity Ranking & Tornado Chart (Publication-Grade).
Addresses Monte Carlo noise floor (n=20,000 + 5-seed replication), decomposes behavioral
parameters (p_miss, react_mean, beta) separately at standardized +/-30% ranges,
evaluates P_slip, P(checked at stop), and Bulkhead Effect (Delta P_slip),
and runs across multiple regimes (single_door baseline, distinct_random, crowded mode).

Place inside MultipleEvaders/V4A_slipscale/ alongside qoyanshyq_v4a_newscale.py.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "TrialModelBus", ROOT / "media", ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import upd_montecarlo_bus_standart as q31
try:
    import qoyanshyq_v4a_newscale as v4a
except ModuleNotFoundError:
    import qoyanshyq_v4a_newscale_fixed as v4a


def eval_replicated(vc, policy, base_p, base_cfg, n_per_seed=4000, seeds=(20261009, 20261010, 20261011, 20261012, 20261013), tag=100):
    """Evaluates scenario across 5 independent seeds (total N = 20,000 trials) to quantify MC noise."""
    ci = [i for i, c in enumerate(q31.CLASSES) if c.name == vc.name][0]
    pi = 0 if policy == "single_door" else 1

    slips, chks, abds = [], [], []
    for s in seeds:
        res = v4a.run_cell(ci, pi, vc, policy, base_p, base_cfg, n=n_per_seed, seed=s, tag=tag)
        summ = v4a.summarize(res)
        slips.append(100.0 * summ["p_slip"])
        chks.append(100.0 * summ["p_checked"])
        abds.append(100.0 * summ["p_aboard"])

    return {
        "slip_mean": float(np.mean(slips)),
        "slip_se": float(np.std(slips, ddof=1) / math.sqrt(len(seeds))),
        "chk_mean": float(np.mean(chks)),
        "chk_se": float(np.std(chks, ddof=1) / math.sqrt(len(seeds))),
        "abd_mean": float(np.mean(abds)),
        "abd_se": float(np.std(abds, ddof=1) / math.sqrt(len(seeds))),
    }


def eval_bulkhead_delta(vc, policy, base_p, base_cfg, n_per_seed=4000, seeds=(20261009, 20261010, 20261011, 20261012, 20261013), tag=100):
    """Computes Delta P_slip = P_slip(Partition ON) - P_slip(Partition OFF) across seeds."""
    p_on = replace(base_p, partition_on=True)
    p_off = replace(base_p, partition_on=False)
    r_on = eval_replicated(vc, policy, p_on, base_cfg, n_per_seed=n_per_seed, seeds=seeds, tag=tag)
    r_off = eval_replicated(vc, policy, p_off, base_cfg, n_per_seed=n_per_seed, seeds=seeds, tag=tag + 50)
    return {
        "delta_mean": r_on["slip_mean"] - r_off["slip_mean"],
        "delta_se": math.hypot(r_on["slip_se"], r_off["slip_se"]),
    }


def make_custom_mix_monkeypatch(p_miss_scale=1.0, react_scale=1.0, beta_scale=1.0):
    """Returns a scaled PROFILES dictionary for standardized +/-30% behavioral decomposition."""
    scaled = {}
    for k, prof in v4a.PROFILES.items():
        # For typical/vigilant where base p_miss=0, +/-30% around baseline uses +/-0.06 absolute shift if base==0
        new_pmiss = min(1.0, max(0.0, prof.p_miss * p_miss_scale)) if prof.p_miss > 0 else (
            0.06 if p_miss_scale > 1.0 else 0.0
        )
        new_react = max(0.1, prof.react_mean * react_scale)
        new_beta = max(0.0, prof.beta * beta_scale) if prof.beta > 0 else (
            0.9 if beta_scale > 1.0 else 0.0
        )
        scaled[k] = replace(prof, p_miss=new_pmiss, react_mean=new_react, beta=new_beta)
    return scaled


def run_regime(regime_name, vc, policy, mode="baseline", n_per_seed=4000):
    base_p = q31.apply_mode(q31.Params(partition_on=True), mode)
    base_cfg = v4a.Cfg(
        n_evaders=3,
        load=0.50,
        positions="seat",
        background=True,
        service_mean=5.0,
        service_sigma=0.5,
        check_policy="nearest",
        mix=(("novice", 0.3), ("typical", 0.5), ("vigilant", 0.2)),
    )

    orig_profiles = v4a.PROFILES.copy()

    # 1. Measure baseline & Monte Carlo noise floor
    base_stat = eval_replicated(vc, policy, base_p, base_cfg, n_per_seed=n_per_seed, tag=0)
    base_delta = eval_bulkhead_delta(vc, policy, base_p, base_cfg, n_per_seed=n_per_seed, tag=10)
    noise_thresh_slip = 2.0 * math.hypot(base_stat["slip_se"], base_stat["slip_se"])
    noise_thresh_chk = 2.0 * math.hypot(base_stat["chk_se"], base_stat["chk_se"])

    print(f"\n=========================================================================================")
    print(f"REGIME: {regime_name} | {vc.name} | {policy} | mode={mode} | N={n_per_seed*5:,}")
    print(f"Baseline: P_slip = {base_stat['slip_mean']:.2f}% (±{base_stat['slip_se']:.2f}), "
          f"P_checked = {base_stat['chk_mean']:.2f}% (±{base_stat['chk_se']:.2f}), "
          f"Delta_P = {base_delta['delta_mean']:+.2f} pp (±{base_delta['delta_se']:.2f})")
    print(f"2-SE Noise Floor: P_slip = {noise_thresh_slip:.2f} pp | P_checked = {noise_thresh_chk:.2f} pp")
    print(f"=========================================================================================")

    # Standardized perturbations (+/-30% where continuous, decomposed behavioral channels)
    perturbations = [
        (
            "Inattention Rate [p_miss]\n(Low: -30% | High: +30%)",
            "Unobservable Behavioral Assumption (Sensitivity Sweep)",
            lambda p, c: (p, c, make_custom_mix_monkeypatch(p_miss_scale=0.70)),
            lambda p, c: (p, c, make_custom_mix_monkeypatch(p_miss_scale=1.30)),
        ),
        (
            "Reaction Latency [react_mean]\n(Low: -30% | High: +30%)",
            "Unobservable Behavioral Assumption (Sensitivity Sweep)",
            lambda p, c: (p, c, make_custom_mix_monkeypatch(react_scale=0.70)),
            lambda p, c: (p, c, make_custom_mix_monkeypatch(react_scale=1.30)),
        ),
        (
            "Blind-Spot Seat Preference [beta]\n(Low: -30% | High: +30%)",
            "Unobservable Behavioral Assumption (Sensitivity Sweep)",
            lambda p, c: (p, c, make_custom_mix_monkeypatch(beta_scale=0.70)),
            lambda p, c: (p, c, make_custom_mix_monkeypatch(beta_scale=1.30)),
        ),
        (
            "Free Walking Speed [agent_speed]\n(Low: -30% | High: +30%)",
            "Literature / Pedestrian Flow (Weidmann / Fruin)",
            lambda p, c: (replace(p, agent_speed=p.agent_speed * 0.70), c, orig_profiles),
            lambda p, c: (replace(p, agent_speed=p.agent_speed * 1.30), c, orig_profiles),
        ),
        (
            "Cabin Load Factor [load]\n(Low: -30% [0.35] | High: +30% [0.65])",
            "Operational (APC / AVL Counts; Low Privacy Risk)",
            lambda p, c: (p, replace(c, load=0.35), orig_profiles),
            lambda p, c: (p, replace(c, load=0.65), orig_profiles),
        ),
        (
            "Stop Dwell Window [dwell_lo, dwell_hi]\n(Low: -30% | High: +30%)",
            "Operational (APC / AVL Counts; Low Privacy Risk)",
            lambda p, c: (replace(p, dwell_lo=p.dwell_lo * 0.70, dwell_hi=p.dwell_hi * 0.70), c, orig_profiles),
            lambda p, c: (replace(p, dwell_lo=p.dwell_lo * 1.30, dwell_hi=p.dwell_hi * 1.30), c, orig_profiles),
        ),
        (
            "Inspector Walking Speed [insp_speed]\n(Low: -30% | High: +30%)",
            "Depot Timing Trial (Volunteers + Consent)",
            lambda p, c: (replace(p, insp_speed=p.insp_speed * 0.70), c, orig_profiles),
            lambda p, c: (replace(p, insp_speed=p.insp_speed * 1.30), c, orig_profiles),
        ),
        (
            "Inspector Blind-Zone Delay [t_blind]\n(Low: -30% | High: +30%)",
            "Depot Timing Trial (Volunteers + Consent)",
            lambda p, c: (replace(p, insp_blind_lo=p.insp_blind_lo * 0.70, insp_blind_hi=p.insp_blind_hi * 0.70), c, orig_profiles),
            lambda p, c: (replace(p, insp_blind_lo=p.insp_blind_lo * 1.30, insp_blind_hi=p.insp_blind_hi * 1.30), c, orig_profiles),
        ),
        (
            "Ticket Check Service Time [service_mean]\n(Low: -30% [3.5s] | High: +30% [6.5s])",
            "Depot Timing Trial (Volunteers + Consent)",
            lambda p, c: (p, replace(c, service_mean=3.5), orig_profiles),
            lambda p, c: (p, replace(c, service_mean=6.5), orig_profiles),
        ),
        (
            "Evader Group Size [n_evaders]\n(Low: E=2 | High: E=4)",
            "Unobservable Behavioral Assumption (Sensitivity Sweep)",
            lambda p, c: (p, replace(c, n_evaders=2), orig_profiles),
            lambda p, c: (p, replace(c, n_evaders=4), orig_profiles),
        ),
    ]

    rows = []
    for idx, (label, category, low_fn, high_fn) in enumerate(perturbations, start=1):
        p_lo, c_lo, prof_lo = low_fn(base_p, base_cfg)
        v4a.PROFILES = prof_lo
        res_lo = eval_replicated(vc, policy, p_lo, c_lo, n_per_seed=n_per_seed, tag=200 + idx * 2)
        del_lo = eval_bulkhead_delta(vc, policy, p_lo, c_lo, n_per_seed=n_per_seed, tag=400 + idx * 2)

        p_hi, c_hi, prof_hi = high_fn(base_p, base_cfg)
        v4a.PROFILES = prof_hi
        res_hi = eval_replicated(vc, policy, p_hi, c_hi, n_per_seed=n_per_seed, tag=200 + idx * 2 + 1)
        del_hi = eval_bulkhead_delta(vc, policy, p_hi, c_hi, n_per_seed=n_per_seed, tag=400 + idx * 2 + 1)

        v4a.PROFILES = orig_profiles

        swing_slip = abs(res_hi["slip_mean"] - res_lo["slip_mean"])
        swing_chk = abs(res_hi["chk_mean"] - res_lo["chk_mean"])
        swing_del = abs(del_hi["delta_mean"] - del_lo["delta_mean"])

        sig_slip = swing_slip > noise_thresh_slip
        sig_chk = swing_chk > noise_thresh_chk

        rows.append({
            "regime": regime_name,
            "parameter": label.replace("\n", " "),
            "short_label": label,
            "category": category,
            "slip_low_pct": round(res_lo["slip_mean"], 2),
            "slip_base_pct": round(base_stat["slip_mean"], 2),
            "slip_high_pct": round(res_hi["slip_mean"], 2),
            "slip_swing_pp": round(swing_slip, 2),
            "slip_sig": sig_slip,
            "chk_low_pct": round(res_lo["chk_mean"], 2),
            "chk_base_pct": round(base_stat["chk_mean"], 2),
            "chk_high_pct": round(res_hi["chk_mean"], 2),
            "chk_swing_pp": round(swing_chk, 2),
            "chk_sig": sig_chk,
            "delta_low_pp": round(del_lo["delta_mean"], 2),
            "delta_base_pp": round(base_delta["delta_mean"], 2),
            "delta_high_pp": round(del_hi["delta_mean"], 2),
            "delta_swing_pp": round(swing_del, 2),
        })

    rows.sort(key=lambda r: r["slip_swing_pp"], reverse=True)

    print(f"{'Rank':<5}{'Parameter':<44}{'P_slip Swing':>14}{'Sig?':>6}{'P_chk Swing':>13}{'Sig?':>6}{'DeltaP Swing':>14}")
    print("-" * 104)
    for rank, r in enumerate(rows, start=1):
        print(f"{rank:<5}{r['parameter'][:42]:<44}"
              f"{r['slip_swing_pp']:>13.2f} {'YES' if r['slip_sig'] else 'noise':>5}"
              f"{r['chk_swing_pp']:>12.2f} {'YES' if r['chk_sig'] else 'noise':>5}"
              f"{r['delta_swing_pp']:>13.2f}")

    return rows, base_stat, base_delta, noise_thresh_slip, noise_thresh_chk


def plot_three_panel_tornado(rows, base_stat, base_delta, noise_slip, noise_chk, out_png):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 6.8), dpi=300, sharey=True)
    y_pos = np.arange(len(rows))[::-1]

    cat_colors = {
        "Operational (APC / AVL Counts; Low Privacy Risk)": "#2980b9",
        "Depot Timing Trial (Volunteers + Consent)": "#27ae60",
        "Literature / Pedestrian Flow (Weidmann / Fruin)": "#8e44ad",
        "Unobservable Behavioral Assumption (Sensitivity Sweep)": "#e67e22",
    }

    # Panel A: P_slip (%)
    base_s = base_stat["slip_mean"]
    ax1.axvspan(base_s - noise_slip / 2, base_s + noise_slip / 2, color="#cccccc", alpha=0.45, zorder=0)
    for idx, r in enumerate(rows):
        y = y_pos[idx]
        col = cat_colors[r["category"]]
        alpha_lo, alpha_hi = (0.45, 0.95) if r["slip_sig"] else (0.20, 0.35)
        ax1.barh(y, r["slip_low_pct"] - base_s, left=base_s, height=0.58, color=col, alpha=alpha_lo, edgecolor=col, linewidth=1.0)
        ax1.barh(y, r["slip_high_pct"] - base_s, left=base_s, height=0.58, color=col, alpha=alpha_hi, edgecolor=col, linewidth=1.0)

        l_val = min(r["slip_low_pct"], r["slip_high_pct"])
        r_val = max(r["slip_low_pct"], r["slip_high_pct"])
        ax1.text(l_val - 0.35, y, f"{l_val:.1f}%", va="center", ha="right", fontsize=8, color="#222222")
        ax1.text(r_val + 0.35, y, f"{r_val:.1f}%", va="center", ha="left", fontsize=8, color="#222222")

    ax1.axvline(base_s, color="#111111", linestyle="--", linewidth=1.1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r["short_label"] for r in rows], fontsize=9)
    ax1.set_xlabel("Evader Slippage Probability $P_{\\mathrm{slip}}$ (%)", fontweight="bold")
    ax1.set_title("A. Slippage ($P_{\\mathrm{slip}}$ Swing)", fontweight="bold", pad=10)
    all_s = [r["slip_low_pct"] for r in rows] + [r["slip_high_pct"] for r in rows]
    ax1.set_xlim(min(all_s) - 3.5, max(all_s) + 3.5)
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel B: P(checked at stop) (%)
    base_c = base_stat["chk_mean"]
    ax2.axvspan(base_c - noise_chk / 2, base_c + noise_chk / 2, color="#cccccc", alpha=0.45, zorder=0)
    for idx, r in enumerate(rows):
        y = y_pos[idx]
        col = cat_colors[r["category"]]
        alpha_lo, alpha_hi = (0.45, 0.95) if r["chk_sig"] else (0.20, 0.35)
        ax2.barh(y, r["chk_low_pct"] - base_c, left=base_c, height=0.58, color=col, alpha=alpha_lo, edgecolor=col, linewidth=1.0)
        ax2.barh(y, r["chk_high_pct"] - base_c, left=base_c, height=0.58, color=col, alpha=alpha_hi, edgecolor=col, linewidth=1.0)

        l_val = min(r["chk_low_pct"], r["chk_high_pct"])
        r_val = max(r["chk_low_pct"], r["chk_high_pct"])
        ax2.text(l_val - 0.25, y, f"{l_val:.1f}%", va="center", ha="right", fontsize=8, color="#222222")
        ax2.text(r_val + 0.25, y, f"{r_val:.1f}%", va="center", ha="left", fontsize=8, color="#222222")

    ax2.axvline(base_c, color="#111111", linestyle="--", linewidth=1.1)
    ax2.set_xlabel("Stop Inspection Rate $P(\\mathrm{checked\\ at\\ stop})$ (%)", fontweight="bold")
    ax2.set_title("B. Stop Check Rate ($P_{\\mathrm{checked}}$ Swing)", fontweight="bold", pad=10)
    all_c = [r["chk_low_pct"] for r in rows] + [r["chk_high_pct"] for r in rows]
    ax2.set_xlim(max(0, min(all_c) - 2.0), max(all_c) + 2.0)
    ax2.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel C: Bulkhead Effect Delta P_slip (pp)
    base_d = base_delta["delta_mean"]
    for idx, r in enumerate(rows):
        y = y_pos[idx]
        col = cat_colors[r["category"]]
        ax3.barh(y, r["delta_low_pp"] - base_d, left=base_d, height=0.58, color=col, alpha=0.45, edgecolor=col, linewidth=1.0)
        ax3.barh(y, r["delta_high_pp"] - base_d, left=base_d, height=0.58, color=col, alpha=0.95, edgecolor=col, linewidth=1.0)

        l_val = min(r["delta_low_pp"], r["delta_high_pp"])
        r_val = max(r["delta_low_pp"], r["delta_high_pp"])
        ax3.text(l_val - 0.15, y, f"{l_val:+.2f}", va="center", ha="right", fontsize=8, color="#222222")
        ax3.text(r_val + 0.15, y, f"{r_val:+.2f}", va="center", ha="left", fontsize=8, color="#222222")

    ax3.axvline(base_d, color="#111111", linestyle="--", linewidth=1.1)
    ax3.set_xlabel("Bulkhead Occlusion Shift $\\Delta P_{\\mathrm{slip}}$ (pp)", fontweight="bold")
    ax3.set_title("C. Bulkhead Effect ($\\Delta P_{\\mathrm{slip}}$ Swing)", fontweight="bold", pad=10)
    all_d = [r["delta_low_pp"] for r in rows] + [r["delta_high_pp"] for r in rows]
    ax3.set_xlim(min(all_d) - 1.2, max(all_d) + 1.2)
    ax3.grid(axis="x", linestyle=":", alpha=0.5)

    legend_handles = [
        Patch(facecolor="#2980b9", edgecolor="#2980b9", label="APC / AVL Counts (Low Privacy Risk; Aggregated)"),
        Patch(facecolor="#27ae60", edgecolor="#27ae60", label="Parked Depot Trial (Volunteers + Consent)"),
        Patch(facecolor="#8e44ad", edgecolor="#8e44ad", label="Literature Parameter (Weidmann / Fruin)"),
        Patch(facecolor="#e67e22", edgecolor="#e67e22", label="Unobservable Behavioral Assumption (Sweep)"),
        Patch(facecolor="#cccccc", alpha=0.5, label="Grey Band = 2-SE MC Noise Floor (Faded Bars = Noise)"),
        Patch(facecolor="#555555", alpha=0.45, label="Light Shade = Low (-30%) | Dark Shade = High (+30%)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Qoyanshyq v4a Standardized ±30% Sensitivity Ranking (Class-IIb eCitaro 12.14m 3D, N = 20,000/point)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\n[✓] Saved 3-panel Tornado chart to: {out_png}")


def main():
    vc = [c for c in q31.CLASSES if "Class-IIb" in c.name][0]

    # Regime 1: Baseline single_door (primary figure)
    rows_sd, base_s, base_d, n_s, n_c = run_regime("R1_SingleDoor_Baseline", vc, "single_door", mode="baseline", n_per_seed=4000)
    plot_three_panel_tornado(rows_sd, base_s, base_d, n_s, n_c, HERE / "v4a_tornado_sensitivity.png")

    # Regime 2: Multi-door distinct_random
    rows_dr, _, _, _, _ = run_regime("R2_DistinctRandom_Baseline", vc, "distinct_random", mode="baseline", n_per_seed=4000)

    # Regime 3: Crowded mode single_door
    rows_cr, _, _, _, _ = run_regime("R3_SingleDoor_Crowded", vc, "single_door", mode="crowded", n_per_seed=4000)

    all_rows = rows_sd + rows_dr + rows_cr
    csv_path = HERE / "v4a_sensitivity_ranking.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in all_rows[0].keys() if k != "short_label"])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: v for k, v in r.items() if k != "short_label"})
    print(f"[✓] Saved multi-regime sensitivity CSV to: {csv_path}")


if __name__ == "__main__":
    main()