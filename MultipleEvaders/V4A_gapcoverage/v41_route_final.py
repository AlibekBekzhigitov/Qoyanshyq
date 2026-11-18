#!/usr/bin/env python3
"""
Qoyanshyq Engine v4b (Extended) -- Multi-Stop Route, Load Regimes (20%/45%/75%),
Front-View vs. Boarding-Exposure Tradeoff, and Spatial Coverage-Gap Maps.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

W_GAMMA, W_RHOMAX = 1.913, 5.4
PLATFORM_Y = -2.0


@dataclass(frozen=True)
class BusSpec:
    name: str
    short_name: str
    length: float
    door_frac: tuple
    seats: int
    stand_cap: int
    articulated: bool = False
    partition: tuple = (0.0, 2.0, 0.95, 2.55)

    @property
    def total_cap(self) -> int:
        return self.seats + self.stand_cap


FLEET = (
    BusSpec("Class-I (Citaro K 10.63m 2D)", "Class-I (10.6m 2D)", 10.63, (0.13, 0.58), seats=26, stand_cap=54, articulated=False),
    BusSpec("Class-IIa (eCitaro 12.14m 2D)", "Class-IIa (12.1m 2D)", 12.14, (0.11, 0.53), seats=32, stand_cap=56, articulated=False),
    BusSpec("Class-IIb (eCitaro 12.14m 3D)", "Class-IIb (12.1m 3D)", 12.14, (0.11, 0.53, 0.91), seats=29, stand_cap=59, articulated=False),
    BusSpec("Class-IIIa (eCitaro G 18.13m 3D)", "Class-IIIa (18.1m 3D)", 18.13, (0.07, 0.36, 0.68), seats=45, stand_cap=100, articulated=True),
    BusSpec("Class-IIIb (eCitaro G 18.13m 4D)", "Class-IIIb (18.1m 4D)", 18.13, (0.07, 0.36, 0.68, 0.91), seats=42, stand_cap=103, articulated=True),
)


@dataclass(frozen=True)
class RouteParams:
    n_stops: int = 16
    mean_trip_stops: float = 3.5
    raid_prob_q: float = 0.25
    raid_duration_stops: int = 2
    evaders_per_stop: float = 1.2
    load_factor: float = 0.45
    n_inspectors: int = 2
    width: float = 2.55
    # Kinematics & timing
    agent_speed: float = 1.3
    insp_speed: float = 0.8
    queue_mean: float = 6.0
    insp_blind_lo: float = 2.0
    insp_blind_hi: float = 5.0
    dwell_lo: float = 12.0
    dwell_hi: float = 30.0
    interstop_lo: float = 35.0
    interstop_hi: float = 65.0
    proc_time_lo: float = 15.0
    proc_time_hi: float = 30.0
    articulation_penalty: float = 1.8


def weidmann_factor(rho: float) -> float:
    rho = max(float(rho), 1e-6)
    f = 1.0 - math.exp(-W_GAMMA * (1.0 / rho - 1.0 / W_RHOMAX))
    return float(np.clip(f, 0.02, 1.0))


def segment_hits_rect(p0x, p0y, p1x, p1y, rect) -> np.ndarray:
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


def build_stylized_cabin(bus: BusSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate stylized cabin slots (seated + standing) and viewpoint score V_i.
    Includes the front-door exposure tradeoff: seats right at door 0 have great
    sightlines but high immediate capture exposure when inspectors board.
    """
    L = bus.length
    n_half = bus.seats // 2
    xs_kerb = np.linspace(0.12 * L, 0.88 * L, n_half)
    xs_drv = np.linspace(0.12 * L, 0.88 * L, bus.seats - n_half)
    seat_x = np.concatenate([xs_kerb, xs_drv])
    seat_y = np.concatenate([np.full(n_half, 0.60), np.full(bus.seats - n_half, 1.95)])

    stand_x = np.linspace(0.10 * L, 0.90 * L, bus.stand_cap)
    stand_y = np.where(np.arange(bus.stand_cap) % 2 == 0, 1.15, 1.40)

    sx = np.concatenate([seat_x, stand_x])
    sy = np.concatenate([seat_y, stand_y])
    is_seat = np.concatenate([np.ones(bus.seats, dtype=bool), np.zeros(bus.stand_cap, dtype=bool)])

    door_x = np.asarray(bus.door_frac) * L
    # Sightline score: unobstructed view of front boarding platform
    clear_view = (~segment_hits_rect(sx, sy, door_x[0], PLATFORM_Y, bus.partition)).astype(float)
    # Closeness to front door gives faster visual confirmation (sightline sharpness)
    d_front = np.hypot(sx - door_x[0], sy)
    front_sight_bonus = np.exp(-d_front / (0.35 * L))
    # Escape score: normalized closeness to nearest secondary door
    d_sec = np.min(np.abs(sx[:, None] - door_x[None, 1:]), axis=1)
    exit_score = 1.0 - np.clip(d_sec / (0.6 * L), 0.0, 1.0)

    # Evader perceives high sightline + exit proximity, which can pull high-beta agents
    # toward front kerbside windows even though inspectors board at door 0
    viewpoint_score = 0.45 * clear_view + 0.30 * front_sight_bonus + 0.25 * exit_score
    return sx, sy, is_seat, viewpoint_score


def softmax_choice(free_indices: np.ndarray, scores: np.ndarray, beta: float, rng) -> int:
    if beta <= 1e-9 or len(free_indices) == 1:
        return int(rng.choice(free_indices))
    logits = beta * scores[free_indices]
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return int(rng.choice(free_indices, p=probs))


def simulate_route_detailed(
    bus: BusSpec,
    beta: float = 2.0,
    params: RouteParams = RouteParams(),
    n_runs: int = 1500,
    seed: int = 20261118,
) -> tuple[dict, pd.DataFrame]:
    """
    Runs multi-stop route simulation and returns both aggregate summary metrics
    and per-slot coverage-gap statistics (exposures, catches, gap_escape_pct).
    """
    rng = np.random.default_rng(seed)
    sx, sy, is_seat, v_score = build_stylized_cabin(bus)
    n_slots = len(sx)
    door_x = np.asarray(bus.door_frac) * bus.length

    n_bg = min(int(round(params.load_factor * bus.total_cap)), n_slots - 15)
    rho_eff = (n_bg + 3) / (bus.length * params.width)
    f_crowd = weidmann_factor(rho_eff)

    exp_factor = 1.0 - math.exp(-0.5 * beta)
    lead_max = 3.0 * exp_factor
    miss_prob = 0.30 * (1.0 - exp_factor)

    tot_caught = 0
    tot_evaded_raid = 0
    tot_never_seen = 0
    tot_evaders = 0

    slot_caught = np.zeros(n_slots, dtype=int)
    slot_evaded = np.zeros(n_slots, dtype=int)
    slot_total_trips = np.zeros(n_slots, dtype=int)

    p_geom = 1.0 / params.mean_trip_stops

    for _ in range(n_runs):
        seat_idx = np.where(is_seat)[0]
        stand_idx = np.where(~is_seat)[0]
        rng.shuffle(seat_idx)
        rng.shuffle(stand_idx)
        # Realistic short-trip mix: ~70% of background pax sit, ~30% stand near doors
        n_bg_seats = min(int(round(0.70 * n_bg)), len(seat_idx) - 4)
        n_bg_stand = max(0, n_bg - n_bg_seats)
        bg_slots = np.concatenate([seat_idx[:n_bg_seats], stand_idx[:n_bg_stand]])
        occupied = np.zeros(n_slots, dtype=bool)
        occupied[bg_slots] = True

        active_evaders = []
        raid_stops_left = 0
        insp_x = np.full(params.n_inspectors, door_x[0])
        insp_y = np.zeros(params.n_inspectors)
        insp_busy_until = np.zeros(params.n_inspectors)

        for stop in range(params.n_stops):
            t_dwell = rng.uniform(params.dwell_lo, params.dwell_hi)

            if raid_stops_left == 0 and stop < params.n_stops - 1:
                if rng.random() < params.raid_prob_q:
                    raid_stops_left = params.raid_duration_stops
                    insp_x[:] = door_x[0]
                    insp_y[:] = 0.0
                    insp_busy_until = rng.exponential(params.queue_mean, params.n_inspectors)

            raid_active = raid_stops_left > 0

            surviving_evaders = []
            for ev in active_evaders:
                if raid_active:
                    ev["encountered"] = True

                if not raid_active:
                    if stop >= ev["s_out"]:
                        occupied[ev["slot"]] = False
                        if ev["encountered"]:
                            tot_evaded_raid += 1
                            slot_evaded[ev["slot"]] += 1
                        else:
                            tot_never_seen += 1
                    else:
                        surviving_evaders.append(ev)
                else:
                    ex, ey = sx[ev["slot"]], sy[ev["slot"]]
                    occ_p = bool(segment_hits_rect(ex, ey, door_x[0], PLATFORM_Y, bus.partition))
                    if rng.random() < miss_prob:
                        t_det = rng.exponential(params.queue_mean) + rng.uniform(1.5, 3.5)
                    else:
                        t_r = rng.lognormal(0.0, 0.35)
                        lead = rng.uniform(0.5, lead_max) if (not occ_p and lead_max > 0) else 0.0
                        t_det = max(0.0, t_r - lead)

                    v_ev = float(np.clip(rng.normal(params.agent_speed, 0.2), 0.5, 2.5)) * f_crowd
                    d_sec = float(np.min(np.hypot(ex - door_x[1:], ey)))
                    t_esc = t_det + d_sec / v_ev
                    ev["t_esc_stop"] = t_esc
                    surviving_evaders.append(ev)

            active_evaders = surviving_evaders

            if raid_active and active_evaders:
                t_interstop = rng.uniform(params.interstop_lo, params.interstop_hi) if stop < params.n_stops - 1 else 0.0
                active_evaders.sort(key=lambda e: sx[e["slot"]])

                remaining_after_sweep = []
                for ev in active_evaders:
                    ex, ey = sx[ev["slot"]], sy[ev["slot"]]
                    j = int(np.argmin(insp_busy_until))
                    dist = math.hypot(ex - insp_x[j], ey - insp_y[j])
                    v_i = float(np.clip(rng.normal(params.insp_speed, 0.15), 0.3, 1.5)) * f_crowd
                    cross_joint = bus.articulated and ((insp_x[j] < 10.2) != (ex < 10.2))
                    joint_delay = params.articulation_penalty if cross_joint else 0.0
                    occ_i = bool(segment_hits_rect(insp_x[j], insp_y[j], ex, ey, bus.partition))
                    blind_delay = rng.uniform(params.insp_blind_lo, params.insp_blind_hi) if occ_i else 0.0

                    t_reach = insp_busy_until[j] + (dist / v_i) + joint_delay + blind_delay

                    if ev["t_esc_stop"] < t_reach and ev["t_esc_stop"] <= t_dwell:
                        occupied[ev["slot"]] = False
                        tot_evaded_raid += 1
                        slot_evaded[ev["slot"]] += 1
                        insp_busy_until[j] = t_reach
                        insp_x[j], insp_y[j] = ex, ey
                    elif t_reach <= (t_dwell + t_interstop):
                        occupied[ev["slot"]] = False
                        tot_caught += 1
                        slot_caught[ev["slot"]] += 1
                        proc_t = rng.uniform(params.proc_time_lo, params.proc_time_hi)
                        insp_busy_until[j] = t_reach + proc_t
                        insp_x[j], insp_y[j] = ex, ey
                    else:
                        remaining_after_sweep.append(ev)

                active_evaders = remaining_after_sweep
                t_cycle = t_dwell + t_interstop
                insp_busy_until = np.maximum(0.0, insp_busy_until - t_cycle)

            if raid_stops_left > 0:
                raid_stops_left -= 1

            if stop < params.n_stops - 1:
                n_new = int(rng.poisson(params.evaders_per_stop))
                for _ in range(n_new):
                    free_slots = np.where(~occupied)[0]
                    if len(free_slots) == 0:
                        break
                    chosen = softmax_choice(free_slots, v_score, beta, rng)
                    occupied[chosen] = True
                    trip_len = int(rng.geometric(p_geom))
                    s_out = min(params.n_stops - 1, stop + trip_len)
                    tot_evaders += 1
                    slot_total_trips[chosen] += 1
                    active_evaders.append(dict(
                        slot=chosen,
                        s_in=stop,
                        s_out=s_out,
                        encountered=False,
                        t_esc_stop=np.inf,
                    ))

        for ev in active_evaders:
            if ev["encountered"]:
                tot_evaded_raid += 1
                slot_evaded[ev["slot"]] += 1
            else:
                tot_never_seen += 1

    summary = dict(
        bus=bus.name,
        short_name=bus.short_name,
        load_factor=params.load_factor,
        rho_eff=round(rho_eff, 3),
        weidmann_f=round(f_crowd, 3),
        beta=beta,
        total_evaders=tot_evaders,
        caught_pct=tot_caught / max(1, tot_evaders) * 100.0,
        evaded_raid_pct=tot_evaded_raid / max(1, tot_evaders) * 100.0,
        never_encountered_pct=tot_never_seen / max(1, tot_evaders) * 100.0,
        conditional_catch_pct=tot_caught / max(1, tot_caught + tot_evaded_raid) * 100.0,
    )

    slot_exposures = slot_caught + slot_evaded
    cond_gap_pct = np.where(slot_exposures > 0, slot_evaded / np.maximum(1, slot_exposures) * 100.0, np.nan)
    slot_df = pd.DataFrame({
        "bus": bus.name,
        "slot_id": np.arange(n_slots),
        "x": sx,
        "y": sy,
        "is_seat": is_seat,
        "viewpoint_score": v_score,
        "total_trips": slot_total_trips,
        "raid_exposures": slot_exposures,
        "caught": slot_caught,
        "evaded": slot_evaded,
        "coverage_gap_pct": cond_gap_pct,  # % of encountered evaders who escaped from this slot
    })
    return summary, slot_df


if __name__ == "__main__":
    import pandas as pd

    # 1. Compare 3 Load Regimes (20%, 45%, 75%) across all 5 bus classes
    print("=== EXPERIMENT 1: 3 LOAD REGIMES (20%, 45%, 75%) ACROSS ALL 5 CITARO CLASSES ===")
    load_rows = []
    for lf in (0.20, 0.45, 0.75):
        for bus in FLEET:
            summ, _ = simulate_route_detailed(bus, beta=2.0, params=RouteParams(load_factor=lf), n_runs=1200, seed=20261120)
            load_rows.append(summ)
    load_df = pd.DataFrame(load_rows)
    load_df.to_csv("v4b_load_regimes.csv", index=False)
    print(load_df[["short_name", "load_factor", "rho_eff", "weidmann_f", "caught_pct", "evaded_raid_pct", "conditional_catch_pct"]].to_string(index=False))

    # 2. Check non-monotonicity across beta with front-door exposure tradeoff
    print("\n=== EXPERIMENT 2: BETA SWEEP WITH FRONT-EXPOSURE TRADEOFF (Class-IIb & Class-IIIb) ===")
    beta_rows = []
    for bus in (FLEET[0], FLEET[2], FLEET[4]):
        for b in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0):
            summ, _ = simulate_route_detailed(bus, beta=b, params=RouteParams(load_factor=0.45), n_runs=1200, seed=20261121)
            beta_rows.append(summ)
    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv("v4b_beta_tradeoff.csv", index=False)
    print(beta_df[["short_name", "beta", "caught_pct", "evaded_raid_pct", "conditional_catch_pct"]].to_string(index=False))

    # 3. Generate High-Precision Coverage-Gap Maps (beta=1.5 so all slots get sufficient samples)
    print("\n=== EXPERIMENT 3: GENERATING SPATIAL COVERAGE-GAP MAPS ===")
    gap_dfs = []
    for bus in FLEET:
        _, sdf = simulate_route_detailed(bus, beta=1.5, params=RouteParams(load_factor=0.45), n_runs=3500, seed=20261122)
        gap_dfs.append(sdf)
    all_slots_df = pd.concat(gap_dfs, ignore_index=True)
    all_slots_df.to_csv("v4b_coverage_gap_slots.csv", index=False)

    # Plot Coverage-Gap Heatmap across 3 representative classes: Class-I (2D), Class-IIb (3D), Class-IIIb (4D)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), dpi=160)
    target_buses = [FLEET[0], FLEET[2], FLEET[4]]
    for ax, bus in zip(axes, target_buses):
        sub = all_slots_df[all_slots_df["bus"] == bus.name].copy()
        L = bus.length
        # Cabin outer shell
        ax.add_patch(patches.Rectangle((0, 0), L, 2.55, fill=True, facecolor="#f8f9fa", edgecolor="#2c3e50", lw=2))
        # Bulkhead partition B = [0, 2.0] x [0.95, 2.55]
        bx0, bx1, by0, by1 = bus.partition
        ax.add_patch(patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="#7f8c8d", edgecolor="#2c3e50", alpha=0.85, hatch="//"))
        ax.text((bx0 + bx1) / 2, (by0 + by1) / 2, "Bulkhead\n(Blind Zone)", color="white", fontsize=8, fontweight="bold", ha="center", va="center")

        # Articulation joint on 18.13m bus
        if bus.articulated:
            ax.add_patch(patches.Rectangle((9.6, 0), 1.2, 2.55, facecolor="#bdc3c7", alpha=0.45, edgecolor="#7f8c8d", linestyle="--"))
            ax.text(10.2, 1.275, "Articulation\nJoint", fontsize=8, color="#34495e", ha="center", va="center", fontweight="bold")

        # Doors on kerbside (y = 0)
        door_x = np.asarray(bus.door_frac) * L
        for di, dx in enumerate(door_x):
            color = "#e74c3c" if di == 0 else "#27ae60"
            label_d = f"D{di+1} (Insp)" if di == 0 else f"D{di+1} (Exit)"
            ax.add_patch(patches.Rectangle((dx - 0.6, -0.08), 1.2, 0.16, facecolor=color, edgecolor="black", zorder=5))
            ax.text(dx, -0.32, label_d, fontsize=8.5, fontweight="bold", color=color, ha="center", va="top")

        # Plot seats (squares) and standing aisle slots (circles) colored by Coverage Gap (% escaped during raid)
        seats = sub[sub["is_seat"]]
        stands = sub[~sub["is_seat"]]
        sc1 = ax.scatter(seats["x"], seats["y"], c=seats["coverage_gap_pct"], cmap="YlOrRd", vmin=70, vmax=98,
                         s=135, marker="s", edgecolors="#2c3e50", linewidths=0.9, zorder=4, label="Seated Slot")
        sc2 = ax.scatter(stands["x"], stands["y"], c=stands["coverage_gap_pct"], cmap="YlOrRd", vmin=70, vmax=98,
                         s=65, marker="o", edgecolors="#555555", linewidths=0.6, alpha=0.85, zorder=3, label="Standing Slot")

        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-0.55, 2.75)
        ax.set_ylabel("Width y (m)", fontsize=9)
        ax.set_title(f"{bus.name} — Spatial Coverage-Gap Map (% Escaped Given Raid Encounter, Load=45%, beta=1.5)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.4)
        if ax == axes[0]:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    axes[-1].set_xlabel("Longitudinal Cabin Position x (m)  [Front -> Rear]", fontsize=10, fontweight="bold")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Coverage Gap: Escape Probability Given Raid (%)", fontsize=10, fontweight="bold")
    plt.subplots_adjust(left=0.06, right=0.90, top=0.94, bottom=0.08, hspace=0.38)
    plt.savefig("v4b_coverage_gap_heatmap.png", dpi=180)
    print("Saved: v4b_coverage_gap_heatmap.png")