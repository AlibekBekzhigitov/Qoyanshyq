import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BusSpec:
    code: str
    label: str
    length_m: float
    doors: int
    capacity: int
    demand_multiplier: float       # Passenger demand scaling relative to standard 12m bus
    rear_notice_delay_s: float     # Extra delay (s) before rear/articulated cabin notices squad
    inspector_sweep_penalty: float # Time overhead (s) crossing bellows / long aisle
    apc_error_rate: float          # Aggregate door APC sensor error rate


ALMATY_FLEET = [
    BusSpec(
        code="Mini_2D",
        label="Mini Bus (9m, 2D)",
        length_m=9.0, doors=2, capacity=55,
        demand_multiplier=0.55, rear_notice_delay_s=0.8,
        inspector_sweep_penalty=0.0, apc_error_rate=0.020
    ),
    BusSpec(
        code="Standard_12m_3D",
        label="Standard Yutong/GD (12m, 3D)",
        length_m=12.0, doors=3, capacity=95,
        demand_multiplier=1.00, rear_notice_delay_s=2.5,
        inspector_sweep_penalty=1.5, apc_error_rate=0.035
    ),
    BusSpec(
        code="Articulated_18m_4D",
        label="Articulated Bus (18m, 4D)",
        length_m=18.0, doors=4, capacity=160,
        demand_multiplier=1.75, rear_notice_delay_s=6.5,
        inspector_sweep_penalty=4.5, apc_error_rate=0.050
    ),
]


@dataclass
class OnayRouteConfig:
    n_stops: int = 25                  # Total stops from terminal to terminal
    dwell_range: tuple = (15.0, 30.0)  # Stop dwell window (seconds)
    inter_stop_time: float = 55.0      # Inter-stop travel time (seconds)

    # Incoming passenger flow clusters (must sum to 1.0)
    share_honest_paid: float = 0.58    # Regular paying commuters
    share_zero_tap: float = 0.18       # Beneficiaries/students tapping 0-KZT card
    share_no_tap: float = 0.09         # Preschoolers & non-tapping exempt riders (bias source)
    share_evade: float = 0.15          # True fare evaders (qoyanshyq)

    # APC sensor detection probability for preschoolers / low-height objects
    apc_child_detect_prob: float = 0.55

    # Evader payment channels upon noticing inspectors:
    # 0: QR (Kaspi/Halyk), 1: Onay mobile app, 2: SMS, 3: Physical NFC card at door
    channel_probs: tuple = (0.55, 0.25, 0.12, 0.08)
    channel_means: tuple = (4.5, 5.5, 11.0, 9.0)

    # Raid parameters (per-stop entry probability ~0.0045 -> ~10% of full trips get raided)
    raid_stop_prob: float = 0.0045
    raid_duration_stops: int = 2       # Squad stays aboard for 2 consecutive stops
    check_time_valid: float = 4.5      # Time (s) to verify a valid passenger
    fine_proc_time: float = 75.0       # Time (s) to process ID & issue fine protocol to an evader
    fare_amount: float = 120.0         # Operator fare revenue per paid ticket (KZT)
    fine_amount: float = 3700.0        # Administrative fine to state budget (KZT)


def simulate_single_trip(
    rng: np.random.Generator,
    cfg: OnayRouteConfig,
    bus: BusSpec,
    block_delay: float,
    forced_raid_stop: int = -1
):
    """
    Simulate a complete terminal-to-terminal bus trip (25 stops).
    If forced_raid_stop >= 0, triggers a controlled raid at that exact stop for paired A/B auditing.
    """
    stop_idx = np.arange(cfg.n_stops)
    base_demand = 14.0 * bus.demand_multiplier
    demand_profile = base_demand * np.exp(-0.5 * ((stop_idx - cfg.n_stops * 0.45) / (cfg.n_stops * 0.28)) ** 2)
    demand_profile[-1] = 0.0

    # Active riders in cabin: list of dicts
    # cluster: 0=honest_paid, 1=zero_tap, 2=no_tap, 3=evader
    onboard = []

    total_apc_in = 0
    total_billing_valid = 0
    total_paid_tx = 0
    total_panic_paid = 0
    total_fined = 0
    total_true_evaders = 0

    raid_timer = 0
    had_raid = False
    raid_start_stop = -1

    # Snapshots at the raid window (or reference stop) for unbiased A/B identification
    window_apc_onboard = 0
    window_valid_onboard = 0
    window_true_evaders_onboard = 0

    for k in range(cfg.n_stops):
        t_open = k * cfg.inter_stop_time
        dwell = rng.uniform(*cfg.dwell_range)

        # 1. Alighting at stop k
        if k == cfg.n_stops - 1:
            onboard = []
        else:
            onboard = [p for p in onboard if p["exit_stop"] > k]

        # 2. Boarding at stop k (capped by bus capacity)
        if k < cfg.n_stops - 1:
            free_space = max(0, bus.capacity - len(onboard))
            n_arrivals = min(free_space, rng.poisson(demand_profile[k]))
            apc_in_stop = 0

            for _ in range(n_arrivals):
                cluster = rng.choice(
                    [0, 1, 2, 3],
                    p=[cfg.share_honest_paid, cfg.share_zero_tap, cfg.share_no_tap, cfg.share_evade]
                )

                # APC sensor count with child detection rate and door sensor error
                if cluster == 2:
                    counted = rng.random() < cfg.apc_child_detect_prob
                else:
                    counted = rng.random() < (1.0 - bus.apc_error_rate)
                if counted:
                    apc_in_stop += 1

                trip_len = int(np.clip(rng.geometric(p=0.28), 1, cfg.n_stops - 1 - k))
                exit_stop = k + trip_len

                # Cabin zone: 0 = front section, 1 = rear / articulated trailer section
                in_rear_zone = rng.random() < (0.35 if bus.doors == 2 else 0.55)

                paid = False
                t_pay = np.inf
                channel = -1

                if cluster == 0:
                    # Regular payer: distributed latency kernel h(dt) after boarding
                    t_pay = t_open + rng.lognormal(mean=2.1, sigma=0.65)
                elif cluster == 1:
                    # Beneficiary with card: taps 0-KZT immediately upon entry
                    t_pay = t_open + rng.uniform(1.0, 12.0)
                elif cluster == 3:
                    # Evader: does not pay unless alarmed by an active raid
                    total_true_evaders += 1
                    channel = rng.choice([0, 1, 2, 3], p=cfg.channel_probs)

                onboard.append({
                    "cluster": cluster,
                    "paid": paid,
                    "counted_apc": counted,
                    "in_rear": in_rear_zone,
                    "channel": channel,
                    "t_pay": t_pay,
                    "exit_stop": exit_stop
                })

            total_apc_in += apc_in_stop

        # 3. Normal (non-panic) transactions completed before next stop
        t_next = t_open + cfg.inter_stop_time
        for p in onboard:
            if not p["paid"] and p["t_pay"] <= t_next:
                p["paid"] = True
                total_billing_valid += 1
                if p["cluster"] == 0:
                    total_paid_tx += 1

        # 4. Raid trigger check
        trigger_raid = False
        if forced_raid_stop >= 0:
            trigger_raid = (k == forced_raid_stop)
        elif raid_timer == 0 and k < cfg.n_stops - 2:
            trigger_raid = (rng.random() < cfg.raid_stop_prob)

        if trigger_raid:
            raid_timer = cfg.raid_duration_stops
            had_raid = True
            raid_start_stop = k

            # Last-second compliance race: evader payment time vs terminal lock delay
            for p in onboard:
                if p["cluster"] == 3 and not p["paid"]:
                    ch = p["channel"]
                    notice_lag = bus.rear_notice_delay_s if p["in_rear"] else 0.0
                    effective_lock_window = block_delay + (bus.inspector_sweep_penalty if p["in_rear"] else 0.0)
                    t_panic = notice_lag + rng.lognormal(mean=np.log(cfg.channel_means[ch]), sigma=0.35)

                    if t_panic < effective_lock_window:
                        p["paid"] = True
                        total_billing_valid += 1
                        total_paid_tx += 1
                        total_panic_paid += 1

        # Record cabin telemetry snapshot at target stop (k=12 for blind baseline, or raid_start_stop)
        if (had_raid and k == raid_start_stop) or (not had_raid and k == 12):
            window_apc_onboard = sum(1 for p in onboard if p["counted_apc"])
            window_valid_onboard = sum(1 for p in onboard if p["paid"])
            window_true_evaders_onboard = sum(1 for p in onboard if p["cluster"] == 3)

        # 5. Squad inspection work inside the cabin during active raid
        if raid_timer > 0:
            # Two inspectors share available stop + transit time minus bellows traversal overhead
            available_squad_time = max(10.0, (dwell + 35.0) * 2.0 - bus.inspector_sweep_penalty * 2.0)
            unverified = [p for p in onboard if not p["paid"]]
            rng.shuffle(unverified)

            time_spent = 0.0
            for p in unverified:
                if p["cluster"] == 3:
                    if time_spent + cfg.fine_proc_time <= available_squad_time:
                        time_spent += cfg.fine_proc_time
                        p["paid"] = True
                        total_fined += 1
                    else:
                        break
                else:
                    if time_spent + cfg.check_time_valid <= available_squad_time:
                        time_spent += cfg.check_time_valid
                    else:
                        break
            raid_timer -= 1

    operator_fare_rev = total_paid_tx * cfg.fare_amount
    state_fine_rev = total_fined * cfg.fine_amount

    return {
        "bus_code": bus.code,
        "bus_label": bus.label,
        "had_raid": had_raid,
        "apc_in": total_apc_in,
        "billing_valid": total_billing_valid,
        "paid_tx": total_paid_tx,
        "panic_paid": total_panic_paid,
        "fined": total_fined,
        "true_evaders": total_true_evaders,
        "operator_rev": operator_fare_rev,
        "state_fines": state_fine_rev,
        "win_apc": window_apc_onboard,
        "win_valid": window_valid_onboard,
        "win_evaders": window_true_evaders_onboard,
    }


def run_full_audit(n_trips: int = 4000, seed: int = 20261120, out_dir: str = "v5_onay_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = OnayRouteConfig()
    block_delays = [0.0, 3.0, 6.0, 9.0, 12.0]

    sweep_rows = []
    ab_rows = []

    print(f"=== Qoyanshyq v5: Onay Telemetry, Fleet Comparison & Bias-Free Audit (N={n_trips}/cell) ===\n")

    for bus in ALMATY_FLEET:
        print(f"--- Fleet Class: {bus.label} (Cap={bus.capacity}, Doors={bus.doors}) ---")
        print(f"{'Lock Delay (s)':<15} | {'Paid Tx':<9} | {'Panic QR':<9} | {'Fined':<7} | {'Operator Rev (KZT)':<19} | {'State Fines (KZT)':<17}")
        print("-" * 88)

        for tau in block_delays:
            rng = np.random.default_rng(seed)
            # Evaluate on raided trips to clearly contrast terminal lock policies
            trips = [simulate_single_trip(rng, cfg, bus, block_delay=tau, forced_raid_stop=12) for _ in range(n_trips)]

            avg_paid = np.mean([r["paid_tx"] for r in trips])
            avg_panic = np.mean([r["panic_paid"] for r in trips])
            avg_fined = np.mean([r["fined"] for r in trips])
            avg_op_rev = np.mean([r["operator_rev"] for r in trips])
            avg_st_fine = np.mean([r["state_fines"] for r in trips])

            print(f"{tau:<15.1f} | {avg_paid:<9.2f} | {avg_panic:<9.2f} | {avg_fined:<7.2f} | {avg_op_rev:<19.1f} | {avg_st_fine:<17.1f}")

            sweep_rows.append({
                "bus_code": bus.code,
                "bus_label": bus.label,
                "doors": bus.doors,
                "capacity": bus.capacity,
                "block_delay_s": tau,
                "avg_paid_tx": round(avg_paid, 3),
                "avg_panic_paid": round(avg_panic, 3),
                "avg_fined": round(avg_fined, 3),
                "operator_fare_rev_kzt": round(avg_op_rev, 2),
                "state_fine_rev_kzt": round(avg_st_fine, 2),
            })

        # Unbiased A/B Evasion Estimator (90% blind trips vs 10% raided trips, tau_block = 9.0s)
        rng_ab = np.random.default_rng(seed + 99)
        natural_trips = [simulate_single_trip(rng_ab, cfg, bus, block_delay=9.0, forced_raid_stop=-1) for _ in range(n_trips)]

        blind = [r for r in natural_trips if not r["had_raid"]]
        raided = [r for r in natural_trips if r["had_raid"]]

        # 1. Naive full-trip APC minus Billing gap (biased by children & non-tapping beneficiaries)
        naive_gap_pct = 100.0 * np.mean([(r["apc_in"] - r["billing_valid"]) / max(1, r["apc_in"]) for r in blind])

        # 2. Local A/B cabin conversion lift at raid stop (cancels out child/exempt bias)
        conv_blind_win = np.sum([r["win_valid"] for r in blind]) / max(1, np.sum([r["win_apc"] for r in blind]))
        conv_raid_win = np.sum([r["win_valid"] for r in raided]) / max(1, np.sum([r["win_apc"] for r in raided]))
        panic_lift_pp = 100.0 * (conv_raid_win - conv_blind_win)

        # True evader share in cabin at stop 12
        true_evader_share_pct = 100.0 * (np.sum([r["win_evaders"] for r in blind]) / max(1, np.sum([r["win_apc"] for r in blind])))

        ab_rows.append({
            "bus_code": bus.code,
            "bus_label": bus.label,
            "blind_trips": len(blind),
            "raid_trips": len(raided),
            "naive_apc_billing_gap_pct": round(naive_gap_pct, 2),
            "local_panic_lift_pp": round(panic_lift_pp, 2),
            "true_cabin_evader_pct": round(true_evader_share_pct, 2),
        })
        print()

    # Save CSV datasets
    sweep_csv_path = os.path.join(out_dir, "v5_onay_lock_delay_sweep.csv")
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    ab_csv_path = os.path.join(out_dir, "v5_onay_bias_ab_audit.csv")
    with open(ab_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ab_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ab_rows)

    print("=== Unbiased A/B Evasion Estimation Summary (Saved to CSV) ===")
    for r in ab_rows:
        print(
            f"{r['bus_label']:<28} | Blind: {r['blind_trips']:<4} | Raid: {r['raid_trips']:<3} | "
            f"Naive Gap (Biased): {r['naive_apc_billing_gap_pct']:>5.2f}% | "
            f"Raid Panic Lift: +{r['local_panic_lift_pp']:>5.2f} pp | "
            f"True Evaders: {r['true_cabin_evader_pct']:>5.2f}%"
        )

    # Generate visual figures
    plot_results(sweep_rows, ab_rows, out_dir)
    print(f"\nAll CSV files and PNG figures successfully saved to: ./{out_dir}/")


def plot_results(sweep_rows: list, ab_rows: list, out_dir: str):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # Figure 1: Last-Second Panic QR Payments vs Fined Passengers across Fleet Classes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"Mini_2D": "#1f77b4", "Standard_12m_3D": "#ff7f0e", "Articulated_18m_4D": "#2ca02c"}
    markers = {"Mini_2D": "o", "Standard_12m_3D": "s", "Articulated_18m_4D": "^"}

    for bus in ALMATY_FLEET:
        code = bus.code
        sub = [r for r in sweep_rows if r["bus_code"] == code]
        delays = [r["block_delay_s"] for r in sub]
        panic = [r["avg_panic_paid"] for r in sub]
        fined = [r["avg_fined"] for r in sub]

        axes[0].plot(delays, panic, marker=markers[code], color=colors[code], linewidth=2, label=bus.label)
        axes[1].plot(delays, fined, marker=markers[code], color=colors[code], linewidth=2, label=bus.label)

    axes[0].set_title("A. Last-Second Panic Payments (QR/App/SMS)")
    axes[0].set_xlabel("Validator Lock Delay after Door Open (s)")
    axes[0].set_ylabel("Panic-Paid Riders per Raided Trip")
    axes[0].legend()

    axes[1].set_title("B. Fined Evaders (Protocol Processing Bottleneck)")
    axes[1].set_xlabel("Validator Lock Delay after Door Open (s)")
    axes[1].set_ylabel("Fined Riders per Raided Trip")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_panic_vs_fines_by_fleet.png"), dpi=300)
    plt.close()

    # Figure 2: Operator Fare Revenue (KZT) vs State Fine Revenue (KZT)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for bus in ALMATY_FLEET:
        code = bus.code
        sub = [r for r in sweep_rows if r["bus_code"] == code]
        delays = [r["block_delay_s"] for r in sub]
        op_rev = [r["operator_fare_rev_kzt"] for r in sub]
        st_rev = [r["state_fine_rev_kzt"] for r in sub]

        axes[0].plot(delays, op_rev, marker=markers[code], color=colors[code], linewidth=2, label=bus.label)
        axes[1].plot(delays, st_rev, marker=markers[code], color=colors[code], linewidth=2, label=bus.label)

    axes[0].set_title("A. Direct Operator Fare Revenue (120 KZT/Ticket)")
    axes[0].set_xlabel("Validator Lock Delay after Door Open (s)")
    axes[0].set_ylabel("Operator Revenue per Trip (KZT)")
    axes[0].legend()

    axes[1].set_title("B. Administrative Fine Volume (3,700 KZT/Fine to State)")
    axes[1].set_xlabel("Validator Lock Delay after Door Open (s)")
    axes[1].set_ylabel("Administrative Fines per Trip (KZT)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_operator_vs_state_revenue.png"), dpi=300)
    plt.close()

    # Figure 3: Naive APC-Billing Gap (Biased) vs Local Raid Panic Lift & True Evader Share
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [r["bus_label"] for r in ab_rows]
    x = np.arange(len(labels))
    width = 0.26

    naive_vals = [r["naive_apc_billing_gap_pct"] for r in ab_rows]
    lift_vals = [r["local_panic_lift_pp"] for r in ab_rows]
    true_vals = [r["true_cabin_evader_pct"] for r in ab_rows]

    ax.bar(x - width, naive_vals, width, label="Naive APC-Billing Gap (Biased by Children/Exempt)", color="#d62728", alpha=0.85)
    ax.bar(x, true_vals, width, label="True Evader Share in Cabin (Ground Truth)", color="#2ca02c", alpha=0.85)
    ax.bar(x + width, lift_vals, width, label="Observed Raid Panic-Payment Lift (+pp)", color="#1f77b4", alpha=0.85)

    ax.set_title("Eliminating Child & Beneficiary Bias via Local Raid A/B Contrast")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage / Percentage Points (%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_bias_elimination_audit.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_full_audit()