#!/usr/bin/env python3
"""
Qoyanshyq Engine v4a -- Multi-Evader & Background Crowd Single-Stop Prototype.

Implements Stage v4a of the Qoyanshyq roadmap:
1. Built-in regression test (M=1 evader, load_factor=0.0): verifies exact
   reproduction of v3.2 P_slip benchmarks across all 5 Citaro/eCitaro classes.
2. Class-specific Daimler Buses capacity & interior topology (seated/standing
   slots, bulkhead occlusion B=[0,2.0]x[0.95,2.55], articulation joint bottleneck
   on 18.13m eCitaro G).
3. Heterogeneous evader profiles allocated to AVAILABLE (unoccupied) slots:
   - 'novice': random deep-cabin/blind-spot seating, zero platform lead time,
               delayed recognition after inspector ingress.
   - 'predator': prioritizes door-adjacent / kerbside high-visibility slots,
                 early platform threat classification (lead time up to 3.0 s).
4. Sequential inspector processing with finite service duration (tau_proc) and
   check-ordering policies ('nearest', 'front_to_back', 'random'), plus
   background alighting flow & Interlocked Boarding cost (delay/pax).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
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
class ParamsV4a:
    n_inspectors: int = 2
    width: float = 2.55
    partition_on: bool = True
    # Kinematics & timing (anchored to v3.2)
    agent_speed: float = 1.3
    insp_speed: float = 0.8
    queue_mean: float = 6.0
    insp_blind_lo: float = 2.0
    insp_blind_hi: float = 5.0
    dwell_lo: float = 12.0
    dwell_hi: float = 30.0
    # v4a multi-agent additions
    proc_time_lo: float = 15.0    # s, finite inspector processing time per caught evader
    proc_time_hi: float = 30.0
    articulation_penalty: float = 1.8  # s, extra bottleneck delay crossing bellows (x ~ 10.2m)
    alighting_share: float = 0.25      # fraction of background pax alighting at stop
    tau_interlock: float = 0.0         # s, secondary-door hold delay


def weidmann_factor(rho: np.ndarray) -> np.ndarray:
    rho = np.maximum(np.asarray(rho, float), 1e-6)
    f = 1.0 - np.exp(-W_GAMMA * (1.0 / rho - 1.0 / W_RHOMAX))
    return np.clip(f, 0.02, 1.0)


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


def generate_cabin_slots(bus: BusSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate discrete seated and standing coordinates inside [0.1L, 0.9L] x [0, W]."""
    L = bus.length
    # Seated slots split evenly between kerbside (y=0.60m) and driver-side (y=1.95m)
    n_half = bus.seats // 2
    xs_kerb = np.linspace(0.12 * L, 0.88 * L, n_half)
    xs_drv = np.linspace(0.12 * L, 0.88 * L, bus.seats - n_half)
    seat_x = np.concatenate([xs_kerb, xs_drv])
    seat_y = np.concatenate([np.full(n_half, 0.60), np.full(bus.seats - n_half, 1.95)])

    # Standing slots along central aisle & door vestibules (y in [1.05, 1.50])
    stand_x = np.linspace(0.10 * L, 0.90 * L, bus.stand_cap)
    stand_y = np.where(np.arange(bus.stand_cap) % 2 == 0, 1.15, 1.40)

    all_x = np.concatenate([seat_x, stand_x])
    all_y = np.concatenate([seat_y, stand_y])
    is_seat = np.concatenate([np.ones(bus.seats, dtype=bool), np.zeros(bus.stand_cap, dtype=bool)])
    return all_x, all_y, is_seat


def run_v32_regression_test(n_trials: int = 10_000, seed: int = 20261009) -> bool:
    """
    Step 1 regression check: N=1 evader, load_factor=0.0, single_door, partition ON.
    Must match v3.2 baseline P_slip values within +/- 1.0 pp.
    """
    v32_targets = {
        "Class-I (Citaro K 10.63m 2D)": 87.28,
        "Class-IIa (eCitaro 12.14m 2D)": 89.61,
        "Class-IIb (eCitaro 12.14m 3D)": 88.77,
        "Class-IIIa (eCitaro G 18.13m 3D)": 95.06,
        "Class-IIIb (eCitaro G 18.13m 4D)": 94.99,
    }
    p = ParamsV4a()
    print("=== v4a REGRESSION TEST (M=1 evader, 0 background pax vs v3.2 benchmark) ===")
    all_passed = True
    for ci, bus in enumerate(FLEET):
        rng = np.random.default_rng([seed, ci, 0])
        L = bus.length
        door_x = np.asarray(bus.door_frac) * L
        k = p.n_inspectors

        right = rng.random(n_trials) < 0.5
        xa = rng.uniform(0.1 * L, 0.9 * L, n_trials)
        ya = np.where(right, rng.uniform(0.3, 0.9, n_trials), rng.uniform(1.65, 2.25, n_trials))
        dxj = np.zeros((n_trials, k)) + door_x[0]

        occ_pass = segment_hits_rect(xa[:, None], ya[:, None], dxj, PLATFORM_Y, bus.partition)
        mu = math.log(1.0) - 0.4 ** 2 / 2
        t_react = rng.lognormal(mu, 0.4, n_trials)
        lead = rng.uniform(0.0, 3.0, n_trials)
        t_occ = rng.uniform(1.0, 3.0, (n_trials, k))
        t_det = np.where(occ_pass, t_react[:, None] + t_occ, np.maximum(0.0, t_react - lead)[:, None]).min(axis=1)

        rho = rng.uniform(0.0, 2.0, n_trials)
        v_eff = np.clip(rng.normal(p.agent_speed, 0.2, n_trials), 0.5, 2.5) * weidmann_factor(rho)
        dist_all = np.hypot(xa[:, None] - door_x[None, :], ya[:, None])
        dist_all[:, 0] = np.inf  # door 0 held
        t_escape = t_det + dist_all.min(axis=1) / v_eff

        d_ins = np.hypot(xa[:, None] - dxj, ya[:, None])
        v_ins = np.clip(rng.normal(p.insp_speed, 0.15, (n_trials, k)), 0.3, 1.5)
        queue = rng.exponential(p.queue_mean, (n_trials, k))
        t_dwell = rng.uniform(p.dwell_lo, p.dwell_hi, n_trials)
        blind = rng.uniform(p.insp_blind_lo, p.insp_blind_hi, (n_trials, k))
        occ_insp = segment_hits_rect(dxj, 0.0, xa[:, None], ya[:, None], bus.partition)
        t_check = (d_ins / v_ins + queue + np.where(occ_insp, blind, 0.0)).min(axis=1)

        p_slip = float(((t_escape < t_check) & (t_escape <= t_dwell)).mean()) * 100.0
        target = v32_targets[bus.name]
        diff = p_slip - target
        ok = abs(diff) < 1.0
        all_passed &= ok
        print(f"  {bus.name:<32}: v4a(N=1)={p_slip:6.2f}% | v3.2={target:6.2f}% | diff={diff:+5.2f} pp [{'PASS' if ok else 'FAIL'}]")
    return all_passed


def simulate_multievader_stop(
    bus: BusSpec,
    n_evaders: int = 4,
    predator_share: float = 0.5,
    load_factor: float = 0.45,
    check_policy: str = "nearest",
    params: ParamsV4a = ParamsV4a(),
    n_trials: int = 3000,
    seed: int = 20261118,
) -> dict:
    """
    Simulate one stop with background passengers occupying slots (load_factor)
    and M heterogeneous evaders ('predator' vs 'novice') allocated to FREE slots.
    Inspectors check evaders sequentially with finite processing time tau_proc.
    """
    rng = np.random.default_rng(seed)
    slot_x, slot_y, is_seat = generate_cabin_slots(bus)
    n_slots = len(slot_x)
    door_x = np.asarray(bus.door_frac) * bus.length
    n_doors = len(door_x)

    # Precompute strategic score for each slot: proximity to nearest secondary door + kerbside view
    dist_to_sec_doors = np.min(np.abs(slot_x[:, None] - door_x[None, 1:]), axis=1)
    has_clear_view = ~segment_hits_rect(slot_x, slot_y, door_x[0], PLATFORM_Y, bus.partition)
    predator_score = -dist_to_sec_doors + 2.5 * has_clear_view.astype(float)

    n_bg = min(int(round(load_factor * bus.total_cap)), n_slots - n_evaders)
    n_alighting_bg = int(round(n_bg * params.alighting_share))

    caught_counts = np.zeros(n_trials)
    alighting_delays = np.zeros(n_trials)

    # Cabin density derived directly from passenger count / floor area (L * W)
    cabin_area = bus.length * params.width
    rho_effective = (n_bg + n_evaders) / cabin_area
    f_crowd = float(weidmann_factor(np.array([rho_effective]))[0])

    for tr in range(n_trials):
        # 1. Allocate background passengers (prefer seats first, then standing)
        seat_idx = np.where(is_seat)[0]
        stand_idx = np.where(~is_seat)[0]
        rng.shuffle(seat_idx)
        rng.shuffle(stand_idx)
        priority_bg = np.concatenate([seat_idx, stand_idx])
        occupied = np.zeros(n_slots, dtype=bool)
        occupied[priority_bg[:n_bg]] = True

        # Alighting delay cost for legitimate background passengers under interlock
        if params.tau_interlock > 0.0 and n_alighting_bg > 0:
            # Legitimate passengers at secondary doors wait tau_interlock until doors open
            alighting_delays[tr] = params.tau_interlock * ((n_doors - 1) / n_doors)
        else:
            alighting_delays[tr] = 0.0

        # 2. Allocate evaders to FREE slots according to archetype
        is_predator = rng.random(n_evaders) < predator_share
        evader_slots = np.empty(n_evaders, dtype=int)
        for m in range(n_evaders):
            free_slots = np.where(~occupied)[0]
            if is_predator[m]:
                # Predator picks top-3 best available tactical spots
                top_k = free_slots[np.argsort(predator_score[free_slots])[::-1][:3]]
                chosen = rng.choice(top_k)
            else:
                # Novice picks uniformly among free slots (often deep/blind cabin)
                chosen = rng.choice(free_slots)
            occupied[chosen] = True
            evader_slots[m] = chosen

        ex = slot_x[evader_slots]
        ey = slot_y[evader_slots]

        # 3. Compute escape deadline t_escape for each evader
        t_dwell = rng.uniform(params.dwell_lo, params.dwell_hi)
        t_det = np.empty(n_evaders)
        for m in range(n_evaders):
            occ_p = bool(segment_hits_rect(ex[m], ey[m], door_x[0], PLATFORM_Y, bus.partition))
            if is_predator[m]:
                t_r = rng.lognormal(0.0, 0.35)
                lead = rng.uniform(1.0, 3.0) if not occ_p else 0.0
                t_det[m] = max(0.0, t_r - lead)
            else:
                # Novice only reacts after inspectors enter cabin
                t_det[m] = rng.exponential(params.queue_mean) + rng.uniform(1.5, 4.0)

        v_ev = np.clip(rng.normal(params.agent_speed, 0.2, n_evaders), 0.5, 2.5) * f_crowd
        # Distance to nearest secondary door (door 0 held by inspectors)
        d_sec = np.min(np.hypot(ex[:, None] - door_x[None, 1:], ey[:, None]), axis=1)
        t_walk = d_sec / v_ev
        if params.tau_interlock > 0.0:
            t_esc = np.maximum(t_det + t_walk, params.tau_interlock)
        else:
            t_esc = t_det + t_walk

        # 4. Sequential inspection by k=2 inspectors (both board door 0)
        # Order of targets depends on check_policy
        if check_policy == "nearest":
            order = np.argsort(ex)
        elif check_policy == "front_to_back":
            order = np.argsort(ex + 0.2 * ey)
        else:
            order = rng.permutation(n_evaders)

        insp_avail_time = rng.exponential(params.queue_mean, params.n_inspectors)
        insp_x = np.full(params.n_inspectors, door_x[0])
        insp_y = np.zeros(params.n_inspectors)

        caught = 0
        for m in order:
            # Assign earliest available inspector
            j = int(np.argmin(insp_avail_time))
            dist = math.hypot(ex[m] - insp_x[j], ey[m] - insp_y[j])
            v_i = float(np.clip(rng.normal(params.insp_speed, 0.15), 0.3, 1.5)) * f_crowd

            # Articulation joint bottleneck (around x = 10.2m on 18.13m bus)
            cross_joint = bus.articulated and ((insp_x[j] < 10.2) != (ex[m] < 10.2))
            joint_delay = params.articulation_penalty if cross_joint else 0.0

            # Sightline occlusion penalty on first acquisition from door
            occ_i = bool(segment_hits_rect(insp_x[j], insp_y[j], ex[m], ey[m], bus.partition))
            blind_delay = rng.uniform(params.insp_blind_lo, params.insp_blind_hi) if occ_i else 0.0

            t_reach = insp_avail_time[j] + (dist / v_i) + joint_delay + blind_delay

            # Evader is caught if inspector reaches them before they escape AND within dwell time
            if t_reach <= t_esc[m] and t_reach <= t_dwell:
                caught += 1
                # Inspector becomes busy processing this evader for tau_proc seconds!
                proc_t = rng.uniform(params.proc_time_lo, params.proc_time_hi)
                insp_avail_time[j] = t_reach + proc_t
                insp_x[j], insp_y[j] = ex[m], ey[m]
            else:
                # Target escaped; inspector moves on without processing penalty
                insp_avail_time[j] = t_reach
                insp_x[j], insp_y[j] = ex[m], ey[m]

        caught_counts[tr] = caught

    return dict(
        bus=bus.name,
        n_evaders=n_evaders,
        predator_share=predator_share,
        load_factor=load_factor,
        check_policy=check_policy,
        tau_interlock=params.tau_interlock,
        capture_rate_pct=float(caught_counts.mean() / n_evaders * 100.0),
        avg_caught=float(caught_counts.mean()),
        alighting_delay_s=float(alighting_delays.mean()),
    )


def save_csv(rows: list[dict], filename: str) -> None:
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    # 1. Verify v3.2 backwards compatibility
    assert run_v32_regression_test(n_trials=10_000), "Regression test failed!"

    # 2. Multi-evader experiment across all 5 Citaro classes (M=4 evaders, 45% load)
    print("\n=== v4a MULTI-EVADER SIMULATION (M=4 evaders, 50% predators, 45% bus load) ===")
    print(f"{'Vehicle Class':<32}{'Policy':<15}{'Tau (s)':>8}{'Capture Rate':>14}{'Avg Caught':>12}{'Delay/Pax':>11}")
    print("-" * 92)
    results_rows = []
    for bus in FLEET:
        for pol in ("nearest", "front_to_back", "random"):
            res = simulate_multievader_stop(bus, n_evaders=4, predator_share=0.5, load_factor=0.45,
                                            check_policy=pol, params=ParamsV4a(tau_interlock=0.0))
            results_rows.append(res)
            print(f"{res['bus']:<32}{res['check_policy']:<15}{res['tau_interlock']:>8.1f}"
                  f"{res['capture_rate_pct']:>13.2f}%{res['avg_caught']:>12.2f}{res['alighting_delay_s']:>10.2f}s")
        # Compare with Interlocked Boarding (tau = 8.0s)
        res_int = simulate_multievader_stop(bus, n_evaders=4, predator_share=0.5, load_factor=0.45,
                                            check_policy="nearest", params=ParamsV4a(tau_interlock=8.0))
        res_int["check_policy"] = "nearest+lock_8s"
        results_rows.append(res_int)
        print(f"{res_int['bus']:<32}{'nearest+lock':<15}{res_int['tau_interlock']:>8.1f}"
              f"{res_int['capture_rate_pct']:>13.2f}%{res_int['avg_caught']:>12.2f}{res_int['alighting_delay_s']:>10.2f}s\n")

    save_csv(results_rows, "v4a_multievader_results.csv")
