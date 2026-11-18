#!/usr/bin/env python3
"""
QASH world simulation (stage 1.5): the September street scene, driven by the v2 model.

Design rule: the ENGINE (class World) has no drawing code and uses only simulated
time and a seeded random generator, so every run is reproducible. The VIEWER
(render / write_gif) only draws snapshots the engine recorded.

What is the same as stage 1 (qash_v2_encounter.py)
  * van timeline: slowing U(2,4) s -> parking U(1,3) s -> waiting; a captor team emerges
    tau ~ U(3,8) s after parking; benign vans pass (p = 0.35) or leave after U(6,20) s;
  * pedestrian policies ignore / naive / wait_deploy / flee_park / flee_slow;
  * flight rule (yard viability with the known earliest-emergence bound);
  * pursuer speed 5.0 m/s, chase window 15 s, contact 1 m, costs not used here.

What is new (stage-2 content, all ASSUMPTIONS)
  * a crowd of other pedestrians who are also targets (dilution), a capture quota
    (default 5), three catchers who each take at most one person per stop, escorting
    the captured person back to the van, and relocation to a later slot if the quota
    is not met (up to max_stops);
  * civilian minivans that park and then LEAVE (the September code never let them
    leave, which kept the evader in ESCAPING forever);
  * an all-clear rule: a hidden or waiting pedestrian resumes walking once no van is
    still a threat (posterior <= p_clear), so the diamond no longer stays stuck.

Units: metres and seconds, dt = 0.1 s. All parameters are illustrative; none is
calibrated to any real operation.

Commands
  python qash_world.py gif   --scenario beside --policy flee_park --out beside.gif
  python qash_world.py batch --n 200 --n-ped 0 30
  python qash_world.py check-stage1 --n 600      (needs qash_v2_encounter.py alongside)
Requires Pillow (and numpy only for check-stage1).
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace

from PIL import Image, ImageDraw, ImageFont

POLICIES = ("ignore", "naive", "wait_deploy", "flee_park", "flee_slow")
SCALE = 10                                   # pixels per metre
Y_ROAD, Y_PARKING, Y_SIDEWALK, Y_BUILDINGS = 80, 180, 250, 330
ROAD_Y, SLOT_Y, SIDE_Y, COURT_Y, CATCH_Y = 115, 215, 290, 480, 245
INF = float("inf")
Z95 = 1.959963984540054

COL = dict(bg=(220, 220, 220), building=(70, 75, 85), road=(110, 110, 110), parking=(150, 150, 150),
           line=(240, 240, 240), sidewalk=(180, 175, 165), court=(160, 190, 160), civ=(210, 170, 40),
           evader=(0, 120, 255), catcher=(235, 30, 30), car=(80, 100, 120), van=(30, 30, 40))


@dataclass(frozen=True)
class Params:
    dt: float = 0.1
    t_max: float = 90.0
    street_len: float = 120.0
    x_arch: float = 53.0                      # courtyard entrance (centre of the gap)
    x_min: float = -300.0                     # far off-screen: no wall to corner a runner
    n_slots: int = 16
    slot_x0: float = 10.25                    # centre of slot 0, metres
    slot_pitch: float = 6.5
    parked_frac: float = 0.4                  # share of slots holding ordinary parked cars
    pi: float = 0.15
    p_pass: float = 0.35
    slow: tuple = (2.0, 4.0)
    park: tuple = (1.0, 3.0)
    tau: tuple = (3.0, 8.0)
    leave: tuple = (6.0, 20.0)
    v_walk: float = 1.4
    v_run: float = 4.0
    v_catch: float = 5.0
    v_escort: float = 3.0
    t_react: float = 1.0
    contact: float = 1.0
    t_chase: float = 15.0
    t_flee_max: float = 15.0
    min_hide: float = 2.0
    n_ped: int = 30
    ped_speed: tuple = (1.1, 1.6)
    ped_spawn_rate: float = 0.15              # per second
    quota: int = 5
    n_catchers: int = 3
    r_target: float = 30.0                    # catchers only take people this close to the van (assumption)
    max_stops: int = 2
    civ_rate: float = 0.03                    # civilian minivans per second
    car_rate: float = 0.12                    # background cars per second (visual only)
    sight_ahead: float = 40.0
    sight_behind: float = 40.0
    smart_clear: bool = True
    p_clear: float = 0.02
    catcher_spread: float = 0.5
    catches_per_catcher: int = 1              # September rule: one person per catcher per stop
    stop_mode: str = "scene"                  # "scene" (run until he has passed) or "resolved"

    @property
    def t_min_dep(self) -> float:
        return self.slow[0] + self.park[0] + self.tau[0]

    def slot_x(self, i: int) -> float:
        return self.slot_x0 + i * self.slot_pitch


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def sgn(v):
    return 1.0 if v > 0 else -1.0 if v < 0 else 0.0


def _surv(a, lo, hi):
    return clamp((hi - a) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
class Van:
    def __init__(self, vid, captor, slot, x_slot, t0, p, rng):
        self.id, self.captor, self.stop = vid, captor, 1
        self.known_captor = False
        self.captured = 0                    # people taken so far (all stops)
        self.delivered = 0
        self.depart_t = self.depart_x = None
        self.exit_t = self.exit_x = self.exit_y = None
        self.reset_stop(slot, x_slot, t0, p, rng)

    def reset_stop(self, slot, x_slot, t0, p, rng):
        self.slot, self.x_slot = slot, x_slot
        self.slow_start = t0
        self.park_start = t0 + rng.uniform(*p.slow)
        self.passes = (not self.captor) and rng.random() < p.p_pass
        self.wait_start = self.park_start + rng.uniform(*p.park)
        self.deploy_time = self.wait_start + rng.uniform(*p.tau) if self.captor else INF
        self.leave_time = INF if self.captor else self.wait_start + rng.uniform(*p.leave)
        self.state = "approach"
        self.team, self.team_deployed, self.deploy_t = [], False, None

    def pos(self, t):
        """Drawing position (x in metres, y in pixels), or None if not on the scene."""
        st = self.state
        if st == "approach":
            return None
        if st == "relocating":
            f = clamp((t - self.depart_t) / max(1e-9, self.slow_start - self.depart_t), 0, 1)
            x0, x1 = self.depart_x, self.x_slot - 15
            g = clamp((t - self.depart_t) / 1.0, 0, 1)
            return x0 + f * (x1 - x0), SLOT_Y + g * (ROAD_Y - SLOT_Y)
        if st == "slowing":
            f = clamp((t - self.slow_start) / (self.park_start - self.slow_start), 0, 1)
            return self.x_slot - 15 * (1 - f) ** 2, ROAD_Y
        if st == "parking":
            g = clamp((t - self.park_start) / max(1e-9, self.wait_start - self.park_start), 0, 1)
            return self.x_slot, ROAD_Y + g * (SLOT_Y - ROAD_Y)
        if st in ("waiting", "deployed"):
            return self.x_slot, SLOT_Y
        if st == "gone" and self.exit_t is not None:
            d = t - self.exit_t
            x = self.exit_x + 5.0 * d
            return (x, self.exit_y + clamp(d, 0, 1) * (ROAD_Y - self.exit_y)) if x < 130 else None
        return None

    def cur_x(self, t):
        pp = self.pos(t) if self.state == "relocating" else None
        return pp[0] if pp else self.x_slot


class Ped:
    def __init__(self, x, y, speed):
        self.x, self.y, self.speed = x, y, speed
        self.state, self.claimed = "walk", False


class Evader(Ped):
    def __init__(self, x, policy):
        super().__init__(x, SIDE_Y, 0.0)
        self.policy = policy
        self.state = "WALKING"
        self.in_van = False
        self.caught_t = self.hidden_t = None
        self.flights = self.false_alarms = 0
        self.reset_trigger()
        self.reacted = set()                  # (van, stop, kind) events already reacted to
        self.dirf, self.has_target, self.flee_elapsed = 0.0, False, 0.0

    def reset_trigger(self):
        self.trig_t, self.trig_van = None, None

    def targetable(self):
        return self.state in ("WALKING", "WAITING", "ESCAPING") and not self.claimed


class Catcher:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.state, self.target, self.fresh, self.taken = "hunt", None, True, 0


# --------------------------------------------------------------------------- #
# World (engine)
# --------------------------------------------------------------------------- #
class World:
    def __init__(self, p, policy, seed, delta=0.0, slot=3, van_x=None, captor="yes", record=False,
                 end_on_caught=False):
        self.p, self.policy, self.record, self.end_on_caught = p, policy, record, end_on_caught
        self.rng = random.Random(seed)
        rng = self.rng
        self.t, self.k = 0.0, 0
        self.quiet_since = None
        self.events, self.snaps = [], []
        self.slot_busy = [False] * p.n_slots
        self.parked = []
        x_van = van_x if van_x is not None else p.slot_x(slot)
        if slot >= 0:
            self.slot_busy[slot] = True
        for i in range(p.n_slots):
            if i != slot and rng.random() < p.parked_frac:
                self.slot_busy[i] = True
                self.parked.append(i)
        x_e0 = x_van - delta
        self.evader = Evader(x_e0, policy)
        self.peds = [Ped(rng.uniform(max(-5, x_e0 - 20), x_e0 + 60), rng.uniform(262, 318),
                         rng.uniform(*p.ped_speed)) for _ in range(p.n_ped)]
        is_captor = (captor == "yes") or (captor == "random" and rng.random() < p.pi)
        self.vans = [Van(0, is_captor, slot, x_van, 0.0, p, rng)]
        self.cars, tt = [], 0.0
        while p.car_rate > 0 and tt < p.t_max:
            tt += rng.expovariate(p.car_rate)
            self.cars.append((tt, rng.uniform(6, 9)))
        self.log(0.0, f"van {'CAPTOR' if is_captor else 'benign'}, slot x={x_van:.1f} m, "
                      f"evader starts x={x_e0:.1f} m, yard x={p.x_arch:.1f} m")

    # ---- helpers ---------------------------------------------------------- #
    def log(self, t, msg):
        self.events.append((round(t, 1), msg))

    def release(self, v):
        if v.slot is not None and v.slot >= 0:
            self.slot_busy[v.slot] = False

    def posterior(self, v):
        p, t = self.p, self.t
        if v.known_captor or v.state == "deployed":
            return 1.0
        if v.state == "slowing":
            return p.pi
        if v.state == "parking":
            return p.pi / (p.pi + (1 - p.pi) * (1 - p.p_pass))
        if v.state == "waiting":
            a = t - v.wait_start
            s_t = _surv(a, *p.tau)
            s_b = 1.0 if a <= p.leave[0] else _surv(a, *p.leave)
            num = p.pi * s_t
            return num / (num + (1 - p.pi) * (1 - p.p_pass) * s_b) if num + s_b > 0 else 0.0
        return 0.0

    def uncleared(self, v):
        if v.state in ("approach", "slowing", "parking", "relocating", "deployed"):
            return True
        if v.state == "waiting":
            return True if not self.p.smart_clear else self.posterior(v) > self.p.p_clear
        return False

    def visible(self, v):
        if v.state not in ("slowing", "parking", "waiting", "deployed", "relocating"):
            return False
        d = v.cur_x(self.t) - self.evader.x
        return (d <= self.p.sight_ahead) and (-d <= self.p.sight_behind)

    # ---- vans ------------------------------------------------------------- #
    def update_vans(self, t):
        p = self.p
        for v in self.vans:
            st = v.state
            if st in ("approach", "relocating") and t >= v.slow_start:
                v.state = "slowing"
            elif st == "slowing" and t >= v.park_start:
                if v.passes:
                    v.state, v.exit_t, v.exit_x, v.exit_y = "gone", t, v.x_slot, ROAD_Y
                    self.release(v)
                else:
                    v.state = "parking"
            elif st == "parking" and t >= v.wait_start:
                v.state = "waiting"
            elif st == "waiting":
                if v.captor and t >= v.deploy_time:
                    self.deploy(v, t)
                elif (not v.captor) and t >= v.leave_time:
                    v.state, v.exit_t, v.exit_x, v.exit_y = "gone", t, v.x_slot, SLOT_Y
                    self.release(v)
            elif st == "deployed" and all(c.state == "inside" for c in v.team):
                self.decide_van(v, t)

    def deploy(self, v, t):
        p = self.p
        v.state, v.team_deployed, v.deploy_t, v.known_captor = "deployed", True, t, True
        v.team = [Catcher(v.x_slot + i * p.catcher_spread, CATCH_Y) for i in range(p.n_catchers)]
        self.log(t, f"van {v.id}: team of {p.n_catchers} emerges (stop {v.stop})")
        e = self.evader
        if e.targetable() and abs(e.x - v.x_slot) <= p.contact and p.r_target >= abs(e.x - v.x_slot):
            e.claimed = True
            v.team[0].target, v.team[0].fresh = e, False
            self.capture(v.team[0], v, e, t)

    def decide_van(self, v, t):
        p = self.p
        if v.captured < p.quota and v.stop < p.max_stops:
            cand = [i for i in range(p.n_slots)
                    if not self.slot_busy[i] and p.slot_x(i) >= v.x_slot + 25]
            if cand:
                i = self.rng.choice(cand)
                old_x = v.x_slot
                self.release(v)
                self.slot_busy[i] = True
                v.stop += 1
                travel = max(2.0, (p.slot_x(i) - 15 - old_x) / 6.0)
                v.depart_t, v.depart_x = t, old_x
                v.reset_stop(i, p.slot_x(i), t + travel, p, self.rng)
                v.state = "relocating"
                self.log(t, f"van {v.id}: quota {v.captured}/{p.quota} not met, relocating to x={p.slot_x(i):.1f}")
                return
        v.state, v.exit_t, v.exit_x, v.exit_y = "gone", t, v.x_slot, SLOT_Y
        self.release(v)
        self.log(t, f"van {v.id}: leaves with {v.captured}/{p.quota}")

    def spawn(self, t):
        p, rng = self.p, self.rng
        if p.civ_rate > 0 and rng.random() < p.civ_rate * p.dt:
            free = [i for i in range(p.n_slots) if not self.slot_busy[i] and 12 <= p.slot_x(i) <= 112]
            if free:
                i = rng.choice(free)
                self.slot_busy[i] = True
                self.vans.append(Van(len(self.vans), False, i, p.slot_x(i), t, p, rng))
        if p.ped_spawn_rate > 0 and rng.random() < p.ped_spawn_rate * p.dt:
            self.peds.append(Ped(-2.0, rng.uniform(262, 318), rng.uniform(*p.ped_speed)))

    # ---- evader ----------------------------------------------------------- #
    def start_flight(self, t):
        p, e = self.p, self.evader
        v = e.trig_van
        out = [c for c in v.team if c.state != "inside"]
        if v.team_deployed and out:
            ref, h = min(out, key=lambda c: abs(c.x - e.x)).x, 0.0
        else:
            ref = v.cur_x(t)
            h = max(0.0, p.t_min_dep - (t - v.slow_start))
        d_van = ref - e.x
        s_away = 1.0 if d_van < 0 else -1.0
        dy = p.x_arch - e.x
        meet = h + max(0.0, abs(d_van) - p.contact - p.v_run * h) / (p.v_run + p.v_catch)
        between = (dy * -s_away > 0) and (abs(dy) / p.v_run < meet)
        viable = (dy * s_away > 0) or between or abs(dy) < 1e-9
        e.dirf = sgn(dy) if viable else s_away
        e.has_target = viable
        e.state, e.flee_elapsed = "ESCAPING", 0.0
        e.flights += 1
        if not v.captor:
            e.false_alarms += 1
        self.log(t, f"evader flees {'to the yard' if viable else 'away from the van'}"
                    f"{' (false alarm)' if not v.captor else ''}")

    def evader_intent(self, t):
        """Decide and return (dx, crossing, s_end) without moving the evader."""
        p, e = self.p, self.evader
        none = (0.0, False, 1.0)
        if e.state in ("CAUGHT", "ARRIVED"):
            return none
        vis = [v for v in self.vans if self.visible(v)]
        if e.state in ("WALKING", "WAITING") and e.trig_t is None and e.policy != "ignore":
            for v in vis:
                if e.policy in ("naive", "wait_deploy"):
                    kind = "team"
                    hit = v.team_deployed and any(c.state != "inside" for c in v.team)
                elif e.policy == "flee_park":
                    kind = "park"
                    hit = v.state == "parking" or (v.known_captor and v.state in ("slowing", "relocating"))
                else:  # flee_slow
                    kind = "slow"
                    hit = v.state in ("slowing", "relocating")
                key = (v.id, v.stop, kind)
                if hit and key not in e.reacted:
                    e.reacted.add(key)
                    e.trig_t, e.trig_van = t, v
                    self.log(t, f"evader trigger ({e.policy}) on van {v.id}, "
                                f"posterior {self.posterior(v):.2f}")
                    break
        if e.state in ("WALKING", "WAITING") and e.trig_t is not None and t >= e.trig_t + p.t_react - 1e-9:
            self.start_flight(t)
        if e.state == "ESCAPING":
            e.flee_elapsed += p.dt
            dxe = e.dirf * p.v_run * p.dt
            if e.has_target:
                to_t = e.dirf * (p.x_arch - e.x)
                crossing = to_t <= abs(dxe) + 1e-12
                return dxe, crossing, (clamp(to_t / abs(dxe), 0.0, 1.0) if crossing else 1.0)
            # hunted: a catcher is after him, or a catcher who has not chosen yet could still choose him
            hunted = any((c.target is e) or (c.state == "hunt" and c.target is None
                                             and abs(e.x - v.x_slot) <= p.r_target)
                         for v in self.vans for c in v.team)
            open_threat = any(self.uncleared(v) and not v.team_deployed for v in self.vans)
            if not hunted and (e.flee_elapsed >= p.t_flee_max or (e.flee_elapsed >= 2.0 and not open_threat)):
                e.state = "WAITING"
                e.reset_trigger()
                self.log(t, "evader stops running (nobody is chasing him)")
                return none
            return dxe, False, 1.0
        if e.state == "HIDDEN":
            if (t - e.hidden_t >= p.min_hide) and not any(self.uncleared(v) for v in self.vans):
                e.state = "WALKING"
                e.reset_trigger()
                self.log(t, "evader: all clear, leaves the yard and walks on")
            return none
        adaptive = e.policy in ("wait_deploy", "flee_park", "flee_slow")
        hold = adaptive and any(self.uncleared(v) and v.x_slot > e.x for v in vis)
        e.state = "WAITING" if hold else "WALKING"
        return (0.0 if hold else p.v_walk * p.dt), False, 1.0

    def finalize_evader(self, t, dxe, crossing):
        p, e = self.p, self.evader
        if e.state in ("CAUGHT", "ARRIVED"):
            return
        if crossing and e.state == "ESCAPING":
            e.x, e.state, e.hidden_t = p.x_arch, "HIDDEN", t
            e.reset_trigger()
            self.log(t, "evader reaches the yard and is hidden")
        else:
            e.x = max(p.x_min, e.x + dxe)
            if e.state == "ESCAPING" and not e.has_target and e.x <= p.x_min:
                e.state = "WAITING"
                e.reset_trigger()
        want = COURT_Y if e.state == "HIDDEN" else SIDE_Y
        e.y += clamp(want - e.y, -8, 8)
        if e.state == "WALKING" and e.x >= p.street_len:
            e.state = "ARRIVED"

    # ---- catchers --------------------------------------------------------- #
    def pick_target(self, c, v):
        p, e = self.p, self.evader
        cand = [q for q in self.peds if q.state == "walk" and not q.claimed
                and abs(q.x - v.x_slot) <= p.r_target]
        if e.targetable() and abs(e.x - v.x_slot) <= p.r_target:
            cand.append(e)
        return min(cand, key=lambda q: abs(q.x - c.x)) if cand else None

    def capture(self, c, v, tgt, t):
        tgt.claimed = True
        c.state, c.target = "escort", tgt
        v.captured += 1
        if tgt is self.evader:
            tgt.state, tgt.caught_t = "CAUGHT", t
            self.log(t, f"EVADER CAUGHT (van {v.id}, {v.captured}/{self.p.quota})")
        else:
            tgt.state = "captured"
            self.log(t, f"van {v.id}: a pedestrian is taken {abs(tgt.x - v.x_slot):.1f} m from the van "
                        f"({v.captured}/{self.p.quota})")

    def update_catchers(self, t, dxe, s_end):
        p, e = self.p, self.evader
        for v in self.vans:
            if not v.team:
                continue
            window = t < v.deploy_t + p.t_chase
            for c in v.team:
                full = v.captured + sum(1 for k in v.team if k.state == "hunt" and k.target is not None) >= p.quota
                if c.state == "inside":
                    continue
                if c.fresh:
                    c.fresh = False
                    continue
                if c.state == "hunt":
                    if c.target is None:
                        if window and not full:
                            tgt = self.pick_target(c, v)
                            if tgt is not None:
                                tgt.claimed, c.target = True, tgt
                        if c.target is None:
                            c.state = "return"
                    tgt = c.target
                    if c.state == "hunt" and tgt is not None:
                        ok = (tgt.state in ("WALKING", "WAITING", "ESCAPING")) if tgt is e else tgt.state == "walk"
                        if not window or not ok:
                            tgt.claimed, c.target, c.state = False, None, "return"
                        else:
                            if tgt is e:
                                ga = e.x - c.x
                                dxc = sgn(ga) * p.v_catch * p.dt
                                gb = ga + (dxe - dxc) * s_end
                                hit = (ga * gb <= 0) or (min(abs(ga), abs(gb)) <= p.contact)
                            else:
                                ga = tgt.x - c.x
                                dxc = sgn(ga) * min(p.v_catch * p.dt, abs(ga))
                                hit = abs(tgt.x - (c.x + dxc)) <= p.contact
                            c.x += dxc
                            c.y += clamp(tgt.y - c.y, -5, 5)
                            if hit:
                                self.capture(c, v, tgt, t)
                elif c.state == "escort":
                    d = v.x_slot - c.x
                    c.x += sgn(d) * min(p.v_escort * p.dt, abs(d))
                    c.y += clamp(CATCH_Y - c.y, -5, 5)
                    c.target.x, c.target.y = c.x, c.y
                    if abs(v.x_slot - c.x) <= p.contact:
                        tgt = c.target
                        if tgt is e:
                            e.in_van = True
                        else:
                            tgt.state = "in_van"
                        tgt.claimed, c.target = False, None
                        c.taken += 1
                        c.state = "hunt" if c.taken < p.catches_per_catcher else "inside"
                        v.delivered += 1
                else:  # return
                    d = v.x_slot - c.x
                    c.x += sgn(d) * min(p.v_catch * p.dt, abs(d))
                    c.y += clamp(CATCH_Y - c.y, -5, 5)
                    if abs(v.x_slot - c.x) <= p.contact:
                        c.state = "inside"

    # ---- main loop -------------------------------------------------------- #
    def quiet(self):
        return (not any(self.uncleared(v) for v in self.vans)
                and not any(c.state != "inside" for v in self.vans for c in v.team))

    def finished(self):
        p, e, t = self.p, self.evader, self.t
        if t >= p.t_max or e.state == "ARRIVED":
            return True
        if e.state == "CAUGHT":
            return self.end_on_caught or (e.in_van and t - e.caught_t >= 4.0) or t - e.caught_t >= 25.0
        if p.stop_mode == "resolved":
            return e.state == "HIDDEN" or (self.quiet() and e.state != "ESCAPING")
        if self.quiet() and self.quiet_since is not None and t - self.quiet_since >= 8.0:
            return True
        return self.quiet() and e.state in ("WALKING", "WAITING") and e.x >= p.x_arch + 15

    def step(self):
        p, t = self.p, self.t
        self.update_vans(t)
        self.spawn(t)
        for q in self.peds:
            if q.state == "walk":
                q.x += q.speed * p.dt
        self.peds = [q for q in self.peds if q.x < p.street_len + 5 and q.state != "in_van"]
        dxe, crossing, s_end = self.evader_intent(t)
        self.update_catchers(t, dxe, s_end)
        self.finalize_evader(t, dxe, crossing)
        if self.quiet():
            if self.quiet_since is None:
                self.quiet_since = t
        else:
            self.quiet_since = None
        if self.record:
            self.snaps.append(self.snapshot(t))
        self.t += p.dt
        self.k += 1

    def run(self):
        while not self.finished():
            self.step()
        return self.result()

    def result(self):
        e = self.evader
        return dict(caught=e.caught_t is not None, flights=e.flights, false_alarms=e.false_alarms,
                    crowd=sum(1 for q in self.peds if q.state in ("captured", "in_van")) + self._gone_captured(),
                    delivered=sum(v.delivered for v in self.vans), t=self.t, state=e.state)

    def _gone_captured(self):
        return sum(v.captured for v in self.vans) - (1 if self.evader.caught_t is not None else 0) \
            - sum(1 for q in self.peds if q.state in ("captured", "in_van"))

    def snapshot(self, t):
        vans = []
        for v in self.vans:
            pp = v.pos(t)
            if pp and -8 < pp[0] < 128:
                vans.append((pp[0], pp[1], v.captor, v.state))
        e = self.evader
        post = max([self.posterior(v) for v in self.vans if self.visible(v)] or [0.0])
        return dict(t=t, peds=[(q.x, q.y) for q in self.peds if q.state in ("walk", "captured") and -5 < q.x < 125],
                    vans=vans, catchers=[(c.x, c.y) for v in self.vans for c in v.team if c.state != "inside"],
                    cars=[(-6 + sp * (t - t0), ROAD_Y + 25) for t0, sp in self.cars if 0 <= t - t0 < 25],
                    ev=(e.x, e.y, e.state, e.in_van), post=post,
                    in_van=sum(v.delivered for v in self.vans), captured=sum(v.captured for v in self.vans))


# --------------------------------------------------------------------------- #
# Viewer (draws recorded snapshots only)
# --------------------------------------------------------------------------- #
def render(snap, world):
    p = world.p
    W, H = int(p.street_len * SCALE), 800
    im = Image.new("RGB", (W, H), COL["bg"])
    d = ImageDraw.Draw(im)
    d.rectangle([0, Y_ROAD, W, Y_ROAD + 100], fill=COL["road"])
    d.rectangle([0, Y_PARKING, W, Y_PARKING + 70], fill=COL["parking"])
    for i in range(p.n_slots):
        cx = p.slot_x(i) * SCALE
        d.rectangle([cx - 22.5, 185, cx + 22.5, 245], outline=COL["line"], width=2)
    d.rectangle([0, Y_SIDEWALK, W, Y_SIDEWALK + 80], fill=COL["sidewalk"])
    d.rectangle([0, Y_BUILDINGS, W, H], fill=COL["court"])
    ax = p.x_arch * SCALE
    d.rectangle([40, 370, ax - 70, 530], fill=COL["building"])
    d.rectangle([ax + 50, 370, W - 40, 530], fill=COL["building"])
    for i in world.parked:
        cx = p.slot_x(i) * SCALE
        d.rounded_rectangle([cx - 15, SLOT_Y - 22, cx + 15, SLOT_Y + 22], radius=3, fill=COL["car"])
    for x, y in snap["cars"]:
        d.rounded_rectangle([x * SCALE - 15, y - 22, x * SCALE + 15, y + 22], radius=3, fill=COL["car"])
    for x, y, captor, st in snap["vans"]:
        d.rounded_rectangle([x * SCALE - 19, y - 25, x * SCALE + 19, y + 25], radius=3, fill=COL["van"],
                            outline=(235, 30, 30) if captor else None, width=2)
    for x, y in snap["peds"]:
        d.ellipse([x * SCALE - 4, y - 4, x * SCALE + 4, y + 4], fill=COL["civ"])
    for x, y in snap["catchers"]:
        d.ellipse([x * SCALE - 4, y - 4, x * SCALE + 4, y + 4], fill=COL["catcher"])
    ex, ey, est, in_van = snap["ev"]
    if not in_van:
        d.ellipse([ex * SCALE - 7, ey - 7, ex * SCALE + 7, ey + 7], fill=COL["evader"])
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, W, 34], fill=(235, 235, 235))
    d.text((10, 6), f"t={snap['t']:5.1f}s | policy: {world.policy} | evader: {'IN VAN' if in_van else est} | "
                    f"in van {snap['in_van']}/{p.quota} | P(captor, worst visible van)={snap['post']:.2f}",
           fill=(10, 10, 10), font=font)
    d.text((10, 38), "red outline = true captor van (shown to the viewer only)", fill=(90, 20, 20), font=font)
    return im.resize((W // 2, H // 2), Image.Resampling.LANCZOS)


def write_gif(world, path):
    frames = [render(s, world).convert("P", palette=Image.Palette.ADAPTIVE, colors=64) for s in world.snaps]
    if not frames:
        raise SystemExit("no frames recorded")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(world.p.dt * 1000),
                   loop=0, optimize=True)
    return len(frames)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
SCENARIOS = dict(behind=-12.0, beside=0.0, ahead=18.0, far=35.0)


def wilson(k, n):
    ph = k / n
    den = 1 + Z95 ** 2 / n
    c = (ph + Z95 ** 2 / (2 * n)) / den
    h = Z95 * math.sqrt(ph * (1 - ph) / n + Z95 ** 2 / (4 * n * n)) / den
    return c - h, c + h


def cmd_gif(a):
    p = replace(Params(), n_ped=a.n_ped, smart_clear=not a.no_smart_clear,
                ped_spawn_rate=0.0 if a.n_ped == 0 else Params().ped_spawn_rate)
    delta = a.delta if a.delta is not None else SCENARIOS[a.scenario]
    w = World(p, a.policy, a.seed, delta=delta, slot=a.slot, captor=a.captor, record=True)
    w.run()
    for t, msg in w.events:
        print(f"  {t:5.1f} s  {msg}")
    n = write_gif(w, a.out)
    print(f"wrote {a.out}: {n} frames, outcome {w.result()}")


def cmd_batch(a):
    print(f"delta={a.delta} m, slot={a.slot}, n={a.n} episodes per row, each catcher takes up to {a.catches} "
          f"person(s) per stop (seeded; same seeds for every policy)")
    print(f"{'n_ped':>5} {'policy':<12}{'P(caught|captor)':>18}{'95% CI':>20}{'false alarm|benign':>20}"
          f"{'crowd taken':>13}")
    for n_ped in a.n_ped:
        p = replace(Params(), n_ped=n_ped, ped_spawn_rate=0.0 if n_ped == 0 else Params().ped_spawn_rate,
                    stop_mode="resolved" if a.resolved else "scene", catches_per_catcher=a.catches)
        for pol in a.policies:
            caught = crowd = 0
            fa = 0
            for s in range(a.n):
                r = World(p, pol, 1000 + s, delta=a.delta, slot=a.slot, captor="yes", end_on_caught=True).run()
                caught += r["caught"]
                crowd += r["crowd"]
            for s in range(a.n):
                r = World(p, pol, 5000 + s, delta=a.delta, slot=a.slot, captor="no", end_on_caught=True).run()
                fa += r["flights"] > 0
            lo, hi = wilson(caught, a.n)
            print(f"{n_ped:>5} {pol:<12}{caught / a.n:>18.3f}{'[%.3f, %.3f]' % (lo, hi):>20}"
                  f"{fa / a.n:>20.3f}{crowd / a.n:>13.2f}")


def cmd_check(a):
    import numpy as np
    sys.path.insert(0, ".")
    import qash_v2_encounter as s1
    p1 = s1.Params()
    pw = replace(Params(), n_ped=0, ped_spawn_rate=0.0, civ_rate=0.0, car_rate=0.0, parked_frac=0.0,
                 quota=999, max_stops=1, sight_ahead=1e9, sight_behind=1e9, smart_clear=False,
                 catcher_spread=0.0, x_min=-1e9, street_len=1e9, stop_mode="resolved", t_max=60.0,
                 r_target=1e9)
    print(f"world engine vs stage-1 simulator, P(caught | captor), n={a.n} world episodes per row")
    print(f"{'policy':<12}{'delta':>7}{'yard':>7}{'stage 1':>10}{'world':>10}{'95% CI (world)':>20}")
    for pol in a.policies:
        for d, y in a.cells:
            ref = float(s1.simulate(p1, pol, True, np.array([float(d)]), np.array([float(y)]), 6000, [7, 0])[0][0])
            pc = replace(pw, x_arch=float(y))
            k = sum(World(pc, pol, 100 + s, delta=float(d), slot=-1, van_x=float(d), captor="yes",
                          end_on_caught=True).run()["caught"] for s in range(a.n))
            lo, hi = wilson(k, a.n)
            print(f"{pol:<12}{d:>7}{y:>7}{ref:>10.3f}{k / a.n:>10.3f}{'[%.3f, %.3f]' % (lo, hi):>20}")
    # false alarms for the early-flight policy
    nfa = max(a.n, 4000)
    for pol in ("flee_park",):
        k = sum(World(pw, pol, 900 + s, delta=0.0, slot=-1, van_x=0.0, captor="no", end_on_caught=True)
                .run()["flights"] > 0 for s in range(nfa))
        print(f"false-alarm rate {pol}: world {k / nfa:.3f} (n={nfa}) vs stage 1 0.647 (benign vans that park)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gif")
    g.add_argument("--scenario", choices=list(SCENARIOS), default="beside")
    g.add_argument("--delta", type=float, default=None, help="van position relative to the evader (m)")
    g.add_argument("--slot", type=int, default=3)
    g.add_argument("--policy", choices=POLICIES, default="wait_deploy")
    g.add_argument("--captor", choices=["yes", "no", "random"], default="yes")
    g.add_argument("--seed", type=int, default=3)
    g.add_argument("--n-ped", type=int, default=30)
    g.add_argument("--no-smart-clear", action="store_true")
    g.add_argument("--out", default="qash_world.gif")
    g.set_defaults(fn=cmd_gif)
    b = sub.add_parser("batch")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--delta", type=float, default=0.0)
    b.add_argument("--slot", type=int, default=3)
    b.add_argument("--n-ped", type=int, nargs="+", default=[0, 30])
    b.add_argument("--policies", nargs="+", choices=POLICIES, default=["naive", "wait_deploy", "flee_park"])
    b.add_argument("--resolved", action="store_true")
    b.add_argument("--catches", type=int, default=1, help="people each catcher may take per stop")
    b.set_defaults(fn=cmd_batch)
    c = sub.add_parser("check-stage1")
    c.add_argument("--n", type=int, default=600)
    c.add_argument("--policies", nargs="+", choices=POLICIES, default=["naive", "wait_deploy", "flee_park"])
    c.add_argument("--cells", type=lambda s: tuple(float(x) for x in s.split(",")), nargs="+",
                   default=[(0, 16), (8, -24), (20, 8), (30, 24), (12, 40), (-6, -40), (4, 60), (16, 40)])
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()