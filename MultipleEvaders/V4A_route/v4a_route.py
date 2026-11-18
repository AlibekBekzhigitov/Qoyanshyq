#!/usr/bin/env python3
"""
Qoyanshyq Engine v4b -- Multi-Stop Route & Softmax Experience (beta) Simulator.

Implements Stage v4b of the Qoyanshyq roadmap:
1. Single experience parameter beta >= 0 via Softmax viewpoint allocation:
   - P(slot i) ~ exp(beta * V_i), where V_i combines platform line-of-sight
     and exit proximity.
   - beta = 0.0 -> uniform random selection (inexperienced / v3.2 baseline).
   - High beta -> strong preference for high-visibility / tactical slots and
     faster threat recognition.
2. Multi-stop transit line (S = 16 stops) with stochastic trips:
   - Each evader boards at stop s_in and has a geometric trip length (mean 3.5 stops).
   - Inspection teams board at random stops with deployment probability q.
   - Inspectors remain aboard across consecutive stops and continue sequential
     processing while the bus travels between stops (inter-stop transit window).
3. Three-outcome decomposition separating deployment luck (q) from interior
   geometry (P_slip):
   - 'caught': verified by an inspector before alighting.
   - 'evaded_raid': encountered an inspection team on board but escaped at a stop.
   - 'never_encountered': completed trip without ever encountering inspectors.
4. Exports full sweep tables to CSV:
   - v4b_beta_sweep.csv (non-monotonicity check across beta in [0, 5])
   - v4b_route_outcomes.csv (3-outcome decomposition across all 5 Citaro classes)
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
import numpy as np

W_GAMMA, W_RHOMAX = 1.913, 5.4
PLATFORM_Y = -2.0


@dataclass(frozen=True)
class BusSpec:
    name: str
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
    BusSpec("Class-I (Citaro K 10.63m 2D)", 10.63, (0.13, 0.58), seats=26, stand_cap=54, articulated=False),
    BusSpec("Class-IIa (eCitaro 12.14m 2D)", 12.14, (0.11, 0.53), seats=32, stand_cap=56, articulated=False),
    BusSpec("Class-IIb (eCitaro 12.14m 3D)", 12.14, (0.11, 0.53, 0.91), seats=29, stand_cap=59, articulated=False),
    BusSpec("Class-IIIa (eCitaro G 18.13m 3D)", 18.13, (0.07, 0.36, 0.68), seats=45, stand_cap=100, articulated=True),
    BusSpec("Class-IIIb (eCitaro G 18.13m 4D)", 18.13, (0.07, 0.36, 0.68, 0.91), seats=42, stand_cap=103, articulated=True),
)


@dataclass(frozen=True)
class RouteParams:
    n_stops: int = 16
    mean_trip_stops: float = 3.5       # geometric trip length mean
    raid_prob_q: float = 0.25          # deployment probability q per stop
    raid_duration_stops: int = 2       # how many stops a boarded team stays on the bus
    evaders_per_stop: float = 1.2      # Poisson arrival rate of new evaders per stop
    load_factor: float = 0.45          # background cabin load fraction
    n_inspectors: int = 2
    width: float = 2.55
    # Kinematics & timing (anchored to v3.2)
    agent_speed: float = 1.3
    insp_speed: float = 0.8
    queue_mean: float = 6.0
    insp_blind_lo: float = 2.0
    insp_blind_hi: float = 5.0
    dwell_lo: float = 12.0
    dwell_hi: float = 30.0
    interstop_lo: float = 35.0         # s, travel time between consecutive stops
    interstop_hi: float = 65.0
    proc_time_lo: float = 15.0         # s, sequential processing time per caught evader
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
    Generate stylized cabin slots (seated + standing) and normalized viewpoint score V_i in [0, 1].
    V_i measures unobstructed platform sightline + proximity to secondary escape doors.
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
    # Sightline score: clear view of front boarding door platform
    clear_view = (~segment_hits_rect(sx, sy, door_x[0], PLATFORM_Y, bus.partition)).astype(float)
    # Escape score: normalized closeness to nearest secondary door
    d_sec = np.min(np.abs(sx[:, None] - door_x[None, 1:]), axis=1)
    exit_score = 1.0 - np.clip(d_sec / (0.6 * L), 0.0, 1.0)

    viewpoint_score = 0.55 * clear_view + 0.45 * exit_score
    return sx, sy, is_seat, viewpoint_score


def softmax_choice(free_indices: np.ndarray, scores: np.ndarray, beta: float, rng) -> int:
    """Select an available slot using Softmax(beta * V_i). At beta=0, uniform random."""
    if beta <= 1e-9 or len(free_indices) == 1:
        return int(rng.choice(free_indices))
    logits = beta * scores[free_indices]
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return int(rng.choice(free_indices, p=probs))


def simulate_route(
    bus: BusSpec,
    beta: float = 2.0,
    params: RouteParams = RouteParams(),
    n_runs: int = 1500,
    seed: int = 20261118,
) -> dict:
    """
    Simulate n_runs full transit trips of length n_stops.
    Tracks every evader from s_in to s_out and classifies outcome into:
      - caught
      - evaded_raid (encountered inspectors on >= 1 stop/segment, escaped at stop)
      - never_encountered (no raid active during evader's trip)
    """
    rng = np.random.default_rng(seed)
    sx, sy, is_seat, v_score = build_stylized_cabin(bus)
    n_slots = len(sx)
    door_x = np.asarray(bus.door_frac) * bus.length

    n_bg = min(int(round(params.load_factor * bus.total_cap)), n_slots - 15)
    rho_eff = (n_bg + 3) / (bus.length * params.width)
    f_crowd = weidmann_factor(rho_eff)

    # Experience-driven behavioral scaling from single parameter beta
    exp_factor = 1.0 - math.exp(-0.5 * beta)  # 0 at beta=0, approaches 1 at high beta
    lead_max = 3.0 * exp_factor
    miss_prob = 0.30 * (1.0 - exp_factor)     # inexperienced evaders miss platform cues 30% of the time

    tot_caught = 0
    tot_evaded_raid = 0
    tot_never_seen = 0
    tot_evaders = 0

    p_geom = 1.0 / params.mean_trip_stops

    for _ in range(n_runs):
        # Background passengers occupy seats first
        seat_idx = np.where(is_seat)[0]
        stand_idx = np.where(~is_seat)[0]
        rng.shuffle(seat_idx)
        rng.shuffle(stand_idx)
        bg_slots = np.concatenate([seat_idx, stand_idx])[:n_bg]
        occupied = np.zeros(n_slots, dtype=bool)
        occupied[bg_slots] = True

        # Active evaders list: dicts with id, slot, s_in, s_out, encountered_raid
        active_evaders = []
        raid_stops_left = 0
        insp_x = np.full(params.n_inspectors, door_x[0])
        insp_y = np.zeros(params.n_inspectors)
        insp_busy_until = np.zeros(params.n_inspectors)

        for stop in range(params.n_stops):
            # 1. Alighting at current stop (if raid active, evaders may also attempt early escape)
            t_dwell = rng.uniform(params.dwell_lo, params.dwell_hi)

            # Check if a new raid deploys at this stop (if no team currently aboard)
            if raid_stops_left == 0 and stop < params.n_stops - 1:
                if rng.random() < params.raid_prob_q:
                    raid_stops_left = params.raid_duration_stops
                    insp_x[:] = door_x[0]
                    insp_y[:] = 0.0
                    insp_busy_until = rng.exponential(params.queue_mean, params.n_inspectors)

            raid_active = raid_stops_left > 0

            # Process active evaders at this stop
            surviving_evaders = []
            for ev in active_evaders:
                if raid_active:
                    ev["encountered"] = True

                # Scheduled exit at s_out (if no active raid blocking or if escaping during raid)
                if not raid_active:
                    if stop >= ev["s_out"]:
                        occupied[ev["slot"]] = False
                        if ev["encountered"]:
                            tot_evaded_raid += 1
                        else:
                            tot_never_seen += 1
                    else:
                        surviving_evaders.append(ev)
                else:
                    # Raid is active at this stop: evader tries to detect and escape via secondary door
                    ex, ey = sx[ev["slot"]], sy[ev["slot"]]
                    occ_p = bool(segment_hits_rect(ex, ey, door_x[0], PLATFORM_Y, bus.partition))
                    if rng.random() < miss_prob:
                        # Missed early warning; reacts only after inspectors board
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

            # 2. If raid is active, inspectors sweep during dwell + inter-stop transit!
            if raid_active and active_evaders:
                t_interstop = rng.uniform(params.interstop_lo, params.interstop_hi) if stop < params.n_stops - 1 else 0.0
                # Sort targets nearest-first from current front position
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

                    # Case A: Evader escapes at the stop before inspector reaches them
                    if ev["t_esc_stop"] < t_reach and ev["t_esc_stop"] <= t_dwell:
                        occupied[ev["slot"]] = False
                        tot_evaded_raid += 1
                        insp_busy_until[j] = t_reach
                        insp_x[j], insp_y[j] = ex, ey
                    # Case B: Inspector reaches evader during dwell OR while bus rides to next stop!
                    elif t_reach <= (t_dwell + t_interstop):
                        occupied[ev["slot"]] = False
                        tot_caught += 1
                        proc_t = rng.uniform(params.proc_time_lo, params.proc_time_hi)
                        insp_busy_until[j] = t_reach + proc_t
                        insp_x[j], insp_y[j] = ex, ey
                    else:
                        # Inspector ran out of inter-stop time; evader survives to next stop
                        remaining_after_sweep.append(ev)

                active_evaders = remaining_after_sweep
                # Reset inspector clock relative to next stop arrival
                t_cycle = t_dwell + t_interstop
                insp_busy_until = np.maximum(0.0, insp_busy_until - t_cycle)

            if raid_stops_left > 0:
                raid_stops_left -= 1

            # 3. New evaders board at stop (if not final stop)
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
                    active_evaders.append(dict(
                        slot=chosen,
                        s_in=stop,
                        s_out=s_out,
                        encountered=False,
                        t_esc_stop=np.inf,
                    ))

        # Any evaders still aboard at final terminal alight safely
        for ev in active_evaders:
            if ev["encountered"]:
                tot_evaded_raid += 1
            else:
                tot_never_seen += 1

    return dict(
        bus=bus.name,
        beta=beta,
        raid_prob_q=params.raid_prob_q,
        total_evaders=tot_evaders,
        caught_pct=tot_caught / max(1, tot_evaders) * 100.0,
        evaded_raid_pct=tot_evaded_raid / max(1, tot_evaders) * 100.0,
        never_encountered_pct=tot_never_seen / max(1, tot_evaders) * 100.0,
        conditional_catch_pct=tot_caught / max(1, tot_caught + tot_evaded_raid) * 100.0,
    )


def save_csv(rows: list[dict], filename: str) -> None:
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    # 1. Beta sweep on Class-IIb (12.14m 3D) and Class-IIIb (18.13m 4D)
    print("=== v4b EXPERIENCE PARAMETER (BETA) SWEEP (16-stop route, q=0.25) ===")
    print(f"{'Vehicle Class':<32}{'Beta':>6}{'Caught':>10}{'Evaded Raid':>13}{'Never Seen':>12}{'Cond. Catch':>13}")
    print("-" * 86)
    beta_rows = []
    for bus in (FLEET[0], FLEET[2], FLEET[4]):
        for b in (0.0, 0.5, 1.0, 2.0, 4.0, 6.0):
            r = simulate_route(bus, beta=b, n_runs=1200, seed=20261118)
            beta_rows.append(r)
            print(f"{r['bus']:<32}{r['beta']:>6.1f}{r['caught_pct']:>9.2f}%"
                  f"{r['evaded_raid_pct']:>12.2f}%{r['never_encountered_pct']:>11.2f}%"
                  f"{r['conditional_catch_pct']:>12.2f}%")
        print()
    save_csv(beta_rows, "v4b_beta_sweep.csv")

    # 2. Full 5-class route comparison at moderate experience (beta = 2.0)
    print("=== v4b 5-CLASS ROUTE DECOMPOSITION (beta=2.0, q=0.25, 45% load) ===")
    print(f"{'Vehicle Class':<32}{'Caught':>10}{'Evaded Raid':>13}{'Never Seen':>12}{'Cond. Catch':>13}")
    print("-" * 80)
    fleet_rows = []
    for bus in FLEET:
        r = simulate_route(bus, beta=2.0, n_runs=1500, seed=20261119)
        fleet_rows.append(r)
        print(f"{r['bus']:<32}{r['caught_pct']:>9.2f}%{r['evaded_raid_pct']:>12.2f}%"
              f"{r['never_encountered_pct']:>11.2f}%{r['conditional_catch_pct']:>12.2f}%")
    save_csv(fleet_rows, "v4b_route_outcomes.csv")
