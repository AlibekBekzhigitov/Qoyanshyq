import sys
import os
import random
import math
import pygame
import imageio

# =====================================================================
# 1. CONSTANTS & DISPLAY SETUP (v4.1 Multi-Evader Dual-Scale Engine)
# =====================================================================
SCREEN_W = 1120
SCREEN_H = 700
FPS = 15

# Color Palette (UI Theme)
COLOR_BG = (15, 17, 23)
COLOR_PANEL = (24, 28, 38)
COLOR_BORDER = (45, 52, 65)

# Macro Map & Street Network Colors
COLOR_MAP_BG = (22, 27, 36)
COLOR_RIVER = (38, 65, 90)
COLOR_STREET_BASE = (50, 60, 75)
COLOR_ROUTE_LINE = (52, 152, 219)
COLOR_STOP_NODE = (241, 196, 15)
COLOR_BUS_MACRO = (231, 76, 60)

# Micro Bus Interior Colors (eCitaro 12.14m 3D Topology)
CELL_SIZE = 28
GRID_BUS_W = 16
GRID_BUS_H = 5

COLOR_BUS_FRAME = (50, 55, 68)
COLOR_BUS_FLOOR = (215, 220, 228)
COLOR_SEAT_BASE = (74, 105, 189)
COLOR_SEAT_BACK = (44, 62, 80)
COLOR_CABIN_WALL = (30, 39, 46)

# Agent Profile Colors (v4a / v4b Experience Profiles)
PROFILE_COLORS = {
    "Novice":   (243, 156, 18),  # Amber Gold (slow reaction, 20% miss rate)
    "Typical":  (46, 204, 113),  # Emerald Green (baseline reaction)
    "Vigilant": (26, 188, 156),  # Teal Cyan (fast reaction + blind-zone seat choice)
}
COLOR_TARGET_TAG = (255, 255, 255)
COLOR_CTRL_UNIFORM = (214, 48, 49)   # Crimson Red (Inspection Patrol)
COLOR_COMMUTER = (142, 68, 173)      # Purple (Compliant Background Flow)
COLOR_OPERATOR = (45, 52, 54)        # Dark Charcoal (Transit Driver)
COLOR_SKIN = (255, 218, 185)

# Vehicle Aperture Coordinates along perimeter (Grid X, Y)
APERTURE_FRONT = (13, 4)
APERTURE_MID = (7, 4)
APERTURE_REAR = (1, 4)
ALL_DOORS = [APERTURE_FRONT, APERTURE_MID, APERTURE_REAR]

# =====================================================================
# 2. VECTOR PATH GEOMETRY (16-STOP URBAN LINE — MATCHING v4b ROUTE)
# =====================================================================
MACRO_ROAD_POLYLINE = [
    (380, 440),  # Terminal 1 (South)
    (360, 400),
    (330, 350),
    (300, 300),
    (260, 250),
    (230, 220),
    (210, 195),
    (215, 175),  # River Crossing
    (245, 160),
    (290, 145),
    (340, 130),
    (365, 115),
    (355, 85),
    (335, 55)    # Terminal 16 (North)
]


def build_arc_length_parameterization(polyline, num_stops=16):
    segments = []
    total_length = 0.0
    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        segments.append((p1, p2, dist, total_length))
        total_length += dist

    stop_distances = [(i / (num_stops - 1)) * total_length for i in range(num_stops)]

    def sample_path(d):
        d = max(0.0, min(d, total_length))
        for p1, p2, dist, start_d in segments:
            if start_d <= d <= start_d + dist or dist == 0:
                factor = (d - start_d) / dist if dist > 0 else 0
                x = p1[0] + factor * (p2[0] - p1[0])
                y = p1[1] + factor * (p2[1] - p1[1])
                angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                return (x, y), angle
        return polyline[-1], 0.0

    stop_coords = [sample_path(sd)[0] for sd in stop_distances]
    return total_length, sample_path, stop_distances, stop_coords


TOTAL_ROUTE_LEN, SAMPLE_ROUTE_FN, STOP_DISTANCES, STOP_COORDS = build_arc_length_parameterization(
    MACRO_ROAD_POLYLINE, 16
)


# =====================================================================
# 3. PROCEDURAL SPRITES & RENDERING OPERATORS
# =====================================================================
def draw_seat(surface, rect):
    pygame.draw.rect(surface, COLOR_SEAT_BASE, rect, border_radius=4)
    backrest = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, 6)
    pygame.draw.rect(surface, COLOR_SEAT_BACK, backrest, border_radius=2)


def draw_sliding_door(surface, rect, is_open, is_held=False):
    pygame.draw.rect(surface, (20, 20, 20), rect)
    if is_open:
        col = (231, 76, 60) if is_held else (46, 204, 113)
        door_l = pygame.Rect(rect.left, rect.top, 4, rect.height)
        door_r = pygame.Rect(rect.right - 4, rect.top, 4, rect.height)
        pygame.draw.rect(surface, col, door_l)
        pygame.draw.rect(surface, col, door_r)
    else:
        pygame.draw.rect(surface, (120, 125, 135), rect)
        pygame.draw.line(surface, (60, 65, 75), (rect.centerx, rect.top), (rect.centerx, rect.bottom), 2)


def draw_humanoid(surface, rect, shirt_color, is_operator=False, is_ctrl=False, checked=False):
    cx, cy = rect.centerx, rect.centery
    torso = pygame.Rect(rect.left + 4, cy - 1, rect.width - 8, rect.height // 2)
    pygame.draw.rect(surface, shirt_color, torso, border_radius=4)
    pygame.draw.circle(surface, COLOR_SKIN, (cx, cy - 5), 4)
    if is_operator:
        pygame.draw.rect(surface, (20, 25, 35), (cx - 4, cy - 10, 8, 3))
    elif is_ctrl:
        pygame.draw.rect(surface, (150, 0, 0), (cx - 4, cy - 10, 8, 3))
    elif checked:
        # Small green validation badge indicating completed ticket check
        pygame.draw.circle(surface, (46, 204, 113), (rect.right - 5, rect.top + 5), 3)


def draw_evader_agent(surface, rect, profile_name):
    col = PROFILE_COLORS.get(profile_name, (46, 204, 113))
    cx, cy = rect.centerx, rect.centery
    body = pygame.Rect(rect.left + 4, cy - 1, rect.width - 8, rect.height // 2)
    pygame.draw.rect(surface, col, body, border_radius=4)
    pygame.draw.circle(surface, col, (cx, cy - 5), 4)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, (cx - 4, cy - 11, 3, 6), border_radius=1)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, (cx + 1, cy - 11, 3, 6), border_radius=1)


# =====================================================================
# 4. MULTI-EVADER DUAL-SCALE ENGINE (v4.1)
# =====================================================================
class MultiEvaderTransitEngine:
    """
    Coupled continuous-discrete simulation engine (v4.1) featuring:
      - E = 3 heterogeneous fare evaders (Novice, Typical, Vigilant)
      - Sequential ticket verification queue (nearest-neighbor patrol sweep)
      - Driver cabin partition occlusion & unguarded aperture evasion
    """

    def __init__(self, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Qoyanshyq Engine v4.1 // Multi-Evader & Patrol Queue Audit")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo, Consolas, monospace", 12)
        self.bold_font = pygame.font.SysFont("Menlo, Consolas, monospace", 14, bold=True)
        self.title_font = pygame.font.SysFont("Menlo, Consolas, monospace", 16, bold=True)

        self.frames_buffer = []
        self.reset()

    def reset(self):
        random.seed(20261009)
        self.current_distance = 0.0
        self.target_stop_idx = 1
        self.move_speed = 4.5

        self.current_stop_num = 1
        self.is_at_stop = False
        self.dwell_ticks = 0

        self.operator_pos = (14, 1)
        self.commuters = set()
        self.checked_commuters = set()

        # Active evaders list: dicts with {id, profile, pos, target_exit, board_stop}
        self.evaders = []
        self.patrol_units = []  # list of dicts: {pos, target, service_ticks}
        self.held_doors = set()
        self.doors_open = False

        # Telemetry counters (v4a / v4b outcomes)
        self.stats = {
            "total_evaders": 0,
            "caught": 0,
            "slipped": 0,
            "completed_uninspected": 0,
            "tickets_checked": 0,
        }
        self.active_sightlines = []  # (p1, p2, occluded_bool)
        self.status_msg = "TERMINAL STOP 1: eCitaro 12.14m 3D departed depot. Moving to Stop 2."

    def is_occluded_by_bulkhead(self, p1, p2):
        """Checks if line-of-sight between grid cells p1 and p2 crosses driver bulkhead x=13.5, y in [1,2]."""
        x1, y1 = p1
        x2, y2 = p2
        if (x1 > 13.5 and x2 <= 13.5) or (x2 > 13.5 and x1 <= 13.5):
            t = (13.5 - x1) / (x2 - x1)
            y_cross = y1 + t * (y2 - y1)
            if 0.8 <= y_cross <= 2.4:
                return True
        return False

    def get_vacant_seats(self, prefer_blind_spot=False):
        occupied = {self.operator_pos} | self.commuters | {e["pos"] for e in self.evaders}
        seats = []
        for x in range(1, 14):
            for y in [1, 3]:
                if (x, y) not in occupied:
                    seats.append((x, y))
        if not seats:
            return []
        if prefer_blind_spot:
            # Vigilant evaders favor seats behind bulkhead / far from front door
            seats.sort(key=lambda s: (s[0] >= 11 and s[1] == 1, -abs(s[0] - 7)), reverse=True)
            return seats[:max(1, len(seats) // 3)]
        return seats

    def spawn_evader(self, profile_name, stop_num):
        seats = self.get_vacant_seats(prefer_blind_spot=(profile_name == "Vigilant"))
        if not seats:
            return
        pos = random.choice(seats)
        # Assign planned exit stop 5..9 stops later
        exit_stop = min(16, stop_num + random.randint(5, 8))
        self.stats["total_evaders"] += 1
        self.evaders.append({
            "id": self.stats["total_evaders"],
            "profile": profile_name,
            "pos": pos,
            "exit_stop": exit_stop,
            "alerted": False,
        })

    def initiate_raid(self, stop):
        """Deploys k=2 inspectors at Front Door (13,4) with sequential nearest-neighbor check queue."""
        self.held_doors = {APERTURE_FRONT}
        self.patrol_units = [
            {"pos": (13, 3), "service_ticks": 0},
            {"pos": (12, 3), "service_ticks": 1},
        ]
        self.active_sightlines.clear()

        # Check line-of-sight from front door to each evader
        for ev in self.evaders:
            occ = self.is_occluded_by_bulkhead((13, 4), ev["pos"])
            self.active_sightlines.append(((13, 4), ev["pos"], occ))
            if ev["profile"] == "Novice" and random.random() < 0.35:
                ev["alerted"] = False  # Missed patrol arrival
            elif occ and ev["profile"] != "Vigilant":
                ev["alerted"] = False  # Occluded behind bulkhead
            else:
                ev["alerted"] = True

    def step_patrol_and_evaders(self):
        """Advances one micro-step of inspector sequential checking and evader escape."""
        if not self.patrol_units:
            return

        # 1. Alerted evaders flee toward unguarded Mid (7,4) or Rear (1,4) doors
        free_doors = [d for d in ALL_DOORS if d not in self.held_doors]
        remaining_evaders = []
        for ev in self.evaders:
            if ev["alerted"] and self.doors_open and free_doors:
                gx, gy = ev["pos"]
                # Pick nearest unguarded door
                target_door = min(free_doors, key=lambda d: math.hypot(d[0] - gx, d[1] - gy))
                dx, dy = target_door
                # Move 1-2 grid cells per tick toward free door
                step_x = gx + (1 if dx > gx else (-1 if dx < gx else 0))
                step_y = gy + (1 if dy > gy else (-1 if dy < gy else 0))
                ev["pos"] = (step_x, step_y)
                if math.hypot(dx - step_x, dy - step_y) <= 1.1:
                    self.stats["slipped"] += 1
                    continue
            remaining_evaders.append(ev)
        self.evaders = remaining_evaders

        # 2. Inspectors perform sequential nearest-neighbor verification (with service delay)
        for insp in self.patrol_units:
            if insp["service_ticks"] > 0:
                insp["service_ticks"] -= 1
                continue

            ix, iy = insp["pos"]
            # Candidate targets: unchecked commuters + remaining evaders
            unchecked_c = list(self.commuters - self.checked_commuters)
            ev_positions = [e["pos"] for e in self.evaders]
            all_targets = unchecked_c + ev_positions

            if not all_targets:
                continue

            tx, ty = min(all_targets, key=lambda p: math.hypot(p[0] - ix, p[1] - iy))
            nx = ix + (1 if tx > ix else (-1 if tx < ix else 0))
            ny = iy + (1 if ty > iy else (-1 if ty < iy else 0))
            insp["pos"] = (nx, ny)

            # Reached target -> spend 2 ticks checking ticket
            if (nx, ny) == (tx, ty):
                insp["service_ticks"] = 2
                self.stats["tickets_checked"] += 1
                if (tx, ty) in self.commuters:
                    self.checked_commuters.add((tx, ty))
                # Check if an evader was caught at this cell
                caught_here = [e for e in self.evaders if e["pos"] == (tx, ty)]
                for e in caught_here:
                    self.stats["caught"] += 1
                    self.evaders.remove(e)

    def handle_stop_arrival(self):
        stop = self.current_stop_num
        self.doors_open = True
        self.held_doors.clear()
        self.patrol_units.clear()
        self.active_sightlines.clear()
        self.checked_commuters.clear()

        if stop == 16:
            for _ in self.evaders:
                self.stats["completed_uninspected"] += 1
            self.evaders.clear()
            self.commuters.clear()
            self.status_msg = "TERMINAL STOP 16: Route completed. Full audit telemetry recorded."
            return

        # Evaders reaching their intended stop disembark cleanly
        still_riding = []
        for ev in self.evaders:
            if stop >= ev["exit_stop"]:
                self.stats["completed_uninspected"] += 1
            else:
                still_riding.append(ev)
        self.evaders = still_riding

        # Background commuter turnover (~45-55% load regime)
        if self.commuters:
            leaving = random.randint(1, min(len(self.commuters), 4))
            for _ in range(leaving):
                self.commuters.pop()

        vacant = self.get_vacant_seats(prefer_blind_spot=False)
        if vacant:
            entering = random.randint(2, min(len(vacant), 5))
            for p in random.sample(vacant, entering):
                self.commuters.add(p)

        # Spawn multi-evader cohort across early/mid route stops
        if stop == 2:
            self.spawn_evader("Novice", stop)
            self.spawn_evader("Typical", stop)
            self.spawn_evader("Vigilant", stop)
            self.status_msg = "STOP 2: E=3 evaders boarded (Novice, Typical, Vigilant). Vigilant took bulkhead shadow."
            return
        elif stop in (5, 9, 12) and len(self.evaders) < 3:
            prof = random.choice(["Novice", "Typical", "Vigilant"])
            self.spawn_evader(prof, stop)

        # Trigger patrol inspection raid on key stops (Stops 4, 8, 11, 14)
        if stop in (4, 8, 11, 14) and self.evaders:
            self.initiate_raid(stop)
            self.status_msg = (
                f"STOP {stop}: PATROL RAID (Front Door held)! Sequential ticket check vs unguarded door evasion."
            )
        else:
            occ_pct = int(100 * (len(self.commuters) + len(self.evaders)) / 25)
            self.status_msg = (
                f"STOP {stop}: Nominal exchange (Load ~{occ_pct}%). Active evaders aboard: {len(self.evaders)}."
            )

    def update(self):
        if self.is_at_stop:
            self.dwell_ticks += 1
            if self.dwell_ticks == 2:
                self.handle_stop_arrival()
            elif 3 <= self.dwell_ticks <= 8:
                self.step_patrol_and_evaders()
            elif self.dwell_ticks >= 10:
                self.is_at_stop = False
                self.doors_open = False
                self.held_doors.clear()
                self.patrol_units.clear()
                self.active_sightlines.clear()
                self.dwell_ticks = 0
                self.target_stop_idx += 1
            return True

        if self.target_stop_idx < len(STOP_DISTANCES):
            target_d = STOP_DISTANCES[self.target_stop_idx]
            if self.current_distance < target_d:
                self.current_distance += self.move_speed
                if self.current_distance >= target_d:
                    self.current_distance = target_d
                    self.is_at_stop = True
                    self.current_stop_num = self.target_stop_idx + 1
            return True
        else:
            self.status_msg = "v4.1 Simulation Complete. Saving macro_micro_v4.gif..."
            return False

    def draw(self):
        self.screen.fill(COLOR_BG)

        # =============================================================
        # 1. MACRO STREET MAP (LEFT PANEL)
        # =============================================================
        map_rect = pygame.Rect(20, 20, 480, 470)
        pygame.draw.rect(self.screen, COLOR_MAP_BG, map_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, map_rect, width=2, border_radius=8)

        river_pts = [(160, 230), (200, 210), (250, 190), (320, 185), (450, 200)]
        pygame.draw.lines(self.screen, COLOR_RIVER, False, river_pts, 18)
        pygame.draw.lines(self.screen, COLOR_STREET_BASE, False, MACRO_ROAD_POLYLINE, 10)
        pygame.draw.lines(self.screen, COLOR_ROUTE_LINE, False, MACRO_ROAD_POLYLINE, 4)

        for idx, (sx, sy) in enumerate(STOP_COORDS):
            stop_n = idx + 1
            is_current = (stop_n == self.current_stop_num)
            is_raid_stop = stop_n in (4, 8, 11, 14)
            color = COLOR_STOP_NODE if is_current else ((231, 76, 60) if is_raid_stop else (240, 240, 240))
            radius = 7 if is_current else 5
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(sx), int(sy)), radius, width=1)
            lbl = self.font.render(str(stop_n), True, (200, 205, 215))
            self.screen.blit(lbl, (sx + 7, sy - 7))

        (bus_x, bus_y), bus_angle = SAMPLE_ROUTE_FN(self.current_distance)
        bus_surface = pygame.Surface((18, 10), pygame.SRCALPHA)
        bus_surface.fill(COLOR_BUS_MACRO)
        pygame.draw.rect(bus_surface, (255, 255, 255), (12, 2, 4, 6))
        rotated_bus = pygame.transform.rotate(bus_surface, -math.degrees(bus_angle))
        self.screen.blit(rotated_bus, rotated_bus.get_rect(center=(int(bus_x), int(bus_y))).topleft)

        self.screen.blit(
            self.bold_font.render("MACRO ROUTE // 16-STOP CORRIDOR (RED = RAID NODES)", True, (220, 220, 220)),
            (32, 30)
        )

        # =============================================================
        # 2. MICRO BUS INTERIOR (RIGHT PANEL)
        # =============================================================
        micro_rect = pygame.Rect(520, 20, 580, 470)
        pygame.draw.rect(self.screen, COLOR_PANEL, micro_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, micro_rect, width=2, border_radius=8)

        self.screen.blit(
            self.bold_font.render("MICRO CABIN // MULTI-EVADER & SEQUENTIAL CHECK QUEUE", True, (220, 220, 220)),
            (540, 30)
        )

        # Live Telemetry Scorecard inside Micro Panel
        score_txt = (
            f"EVADERS: {self.stats['total_evaders']}  |  "
            f"CAUGHT: {self.stats['caught']}  |  "
            f"SLIPPED: {self.stats['slipped']}  |  "
            f"UNINSPECTED EXIT: {self.stats['completed_uninspected']}  |  "
            f"CHECKS: {self.stats['tickets_checked']}"
        )
        self.screen.blit(self.font.render(score_txt, True, (130, 210, 255)), (540, 62))

        bus_ox, bus_oy = 560, 165
        hull_rect = pygame.Rect(bus_ox - 8, bus_oy - 8, GRID_BUS_W * CELL_SIZE + 16, GRID_BUS_H * CELL_SIZE + 16)
        pygame.draw.rect(self.screen, COLOR_BUS_FRAME, hull_rect, border_radius=10)
        pygame.draw.rect(self.screen, (20, 25, 35), hull_rect, width=3, border_radius=10)

        glass_rect = pygame.Rect(bus_ox + GRID_BUS_W * CELL_SIZE - 2, bus_oy + 4, 8, GRID_BUS_H * CELL_SIZE - 8)
        pygame.draw.rect(self.screen, (90, 180, 240), glass_rect, border_radius=3)

        for x in range(GRID_BUS_W):
            for y in range(GRID_BUS_H):
                cell_rect = pygame.Rect(bus_ox + x * CELL_SIZE, bus_oy + y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
                if (y == 1 or y == 3) and 1 <= x <= 13:
                    draw_seat(self.screen, cell_rect)
                elif y == 0 or y == 4:
                    pygame.draw.rect(self.screen, (180, 185, 195), cell_rect)
                else:
                    pygame.draw.rect(self.screen, COLOR_BUS_FLOOR, cell_rect)

        # Driver Cabin Partition (Bulkhead B)
        cab_wall = pygame.Rect(bus_ox + 13 * CELL_SIZE + 24, bus_oy + 1 * CELL_SIZE, 5, CELL_SIZE * 2)
        pygame.draw.rect(self.screen, COLOR_CABIN_WALL, cab_wall)

        # Sliding Doors (Front, Mid, Rear)
        for door in ALL_DOORS:
            dx_d, dy_d = door
            door_rect = pygame.Rect(bus_ox + dx_d * CELL_SIZE, bus_oy + dy_d * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_sliding_door(self.screen, door_rect, self.doors_open, is_held=(door in self.held_doors))

        # Sightline Rays during raid
        for (p1, p2, occ) in self.active_sightlines:
            c1 = (bus_ox + p1[0] * CELL_SIZE + CELL_SIZE // 2, bus_oy + p1[1] * CELL_SIZE + CELL_SIZE // 2)
            c2 = (bus_ox + p2[0] * CELL_SIZE + CELL_SIZE // 2, bus_oy + p2[1] * CELL_SIZE + CELL_SIZE // 2)
            ray_col = (231, 76, 60) if occ else (46, 204, 113)
            pygame.draw.line(self.screen, ray_col, c1, c2, 2)

        # Operator
        vx, vy = self.operator_pos
        draw_humanoid(
            self.screen,
            pygame.Rect(bus_ox + vx * CELL_SIZE, bus_oy + vy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
            COLOR_OPERATOR,
            is_operator=True,
        )

        # Commuters (with green check badge once inspected)
        for (px, py) in self.commuters:
            draw_humanoid(
                self.screen,
                pygame.Rect(bus_ox + px * CELL_SIZE, bus_oy + py * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                COLOR_COMMUTER,
                checked=((px, py) in self.checked_commuters),
            )

        # Evaders (color-coded by profile)
        for ev in self.evaders:
            ex, ey = ev["pos"]
            draw_evader_agent(
                self.screen,
                pygame.Rect(bus_ox + ex * CELL_SIZE, bus_oy + ey * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                ev["profile"],
            )

        # Patrol Units
        for insp in self.patrol_units:
            cx, cy = insp["pos"]
            draw_humanoid(
                self.screen,
                pygame.Rect(bus_ox + cx * CELL_SIZE, bus_oy + cy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1),
                COLOR_CTRL_UNIFORM,
                is_ctrl=True,
            )

        # Door labels under bus
        for lbl_txt, dx_d in [("REAR", 1), ("MID", 7), ("FRONT", 13)]:
            self.screen.blit(
                self.font.render(lbl_txt, True, (160, 170, 185)),
                (bus_ox + dx_d * CELL_SIZE - 4, bus_oy + GRID_BUS_H * CELL_SIZE + 12),
            )

        # =============================================================
        # 3. HUD TELEMETRY & LEGEND (BOTTOM PANEL)
        # =============================================================
        hud_rect = pygame.Rect(20, 505, 1080, 175)
        pygame.draw.rect(self.screen, COLOR_PANEL, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, hud_rect, width=2, border_radius=8)

        self.screen.blit(
            self.title_font.render(f"QOYANSHYQ v4.1 AUDIT MONITOR // STOP {self.current_stop_num} OF 16", True, (255, 255, 255)),
            (40, 520),
        )
        self.screen.blit(self.bold_font.render(self.status_msg, True, COLOR_STOP_NODE), (40, 550))

        leg_y = 595
        draw_evader_agent(self.screen, pygame.Rect(40, leg_y, 16, 16), "Novice")
        self.screen.blit(self.font.render("Novice Evader", True, (243, 156, 18)), (64, leg_y))

        draw_evader_agent(self.screen, pygame.Rect(185, leg_y, 16, 16), "Typical")
        self.screen.blit(self.font.render("Typical Evader", True, (46, 204, 113)), (209, leg_y))

        draw_evader_agent(self.screen, pygame.Rect(335, leg_y, 16, 16), "Vigilant")
        self.screen.blit(self.font.render("Vigilant Evader", True, (26, 188, 156)), (359, leg_y))

        draw_humanoid(self.screen, pygame.Rect(490, leg_y, 16, 16), COLOR_CTRL_UNIFORM, is_ctrl=True)
        self.screen.blit(self.font.render("Inspector (k=2)", True, (240, 100, 100)), (514, leg_y))

        draw_humanoid(self.screen, pygame.Rect(645, leg_y, 16, 16), COLOR_COMMUTER, checked=True)
        self.screen.blit(self.font.render("Commuter (Checked)", True, (190, 195, 205)), (669, leg_y))

        door_txt = "APERTURES: OPEN" if self.doors_open else "APERTURES: LOCKED"
        door_col = (46, 204, 113) if self.doors_open else (150, 155, 165)
        self.screen.blit(self.bold_font.render(door_txt, True, door_col), (840, leg_y))

        self.screen.blit(
            self.font.render("Sightlines: GREEN = Clear LOS | RED = Occluded by Bulkhead B   |   [R] Reset  [SPACE] Pause", True, (130, 140, 160)),
            (40, leg_y + 35),
        )

        pygame.display.flip()

    def record_frame(self):
        scaled = pygame.transform.smoothscale(self.screen, (560, 350))
        view = pygame.surfarray.array3d(scaled).transpose([1, 0, 2])
        self.frames_buffer.append(view)

    def save_gif(self, filename="macro_micro_v4.gif", target_fps=12):
        if not self.frames_buffer:
            return
        file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(os.path.dirname(file_path)) if ".venv" in file_path else os.path.dirname(file_path)
        output_path = os.path.join(base_dir, filename)
        imageio.mimsave(output_path, self.frames_buffer[::2], fps=target_fps, loop=0)
        print(f"[✓] Saved v4.1 multi-evader animation to: {output_path}")

    def run(self):
        running = True
        paused = False
        try:
            while running:
                self.clock.tick(FPS)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r or event.scancode in [15, 19]:
                            self.reset()
                            paused = False
                        elif event.key == pygame.K_SPACE or event.key == pygame.K_TAB:
                            paused = not paused

                if not paused:
                    if self.update() is False:
                        running = False

                self.draw()
                if len(self.frames_buffer) < 550:
                    self.record_frame()
        finally:
            self.save_gif("macro_micro_v4.gif", target_fps=12)
            pygame.quit()


if __name__ == "__main__":
    engine = MultiEvaderTransitEngine(headless=False)
    engine.run()