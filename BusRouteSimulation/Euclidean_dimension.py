import os
import sys
import random
import math
import pygame
import imageio

# =====================================================================
# 1. CONSTANTS & DISPLAY SETUP
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

# Micro Bus Interior Colors
CELL_SIZE = 28
GRID_BUS_W = 16
GRID_BUS_H = 5

COLOR_BUS_FRAME = (50, 55, 68)
COLOR_BUS_FLOOR = (215, 220, 228)
COLOR_SEAT_BASE = (74, 105, 189)
COLOR_SEAT_BACK = (44, 62, 80)
COLOR_CABIN_WALL = (30, 39, 46)

# Agent Classification Colors
COLOR_TARGET_AGENT = (46, 204, 113)  # Emerald Green (Adversarial Benchmark Agent)
COLOR_TARGET_TAG = (255, 255, 255)
COLOR_CTRL_UNIFORM = (214, 48, 49)   # Crimson Red (Inspection Patrol)
COLOR_COMMUTER = (142, 68, 173)      # Purple (Background Commuter Flow)
COLOR_OPERATOR = (45, 52, 54)        # Dark Charcoal (Transit Driver)
COLOR_SKIN = (255, 218, 185)

# Vehicle Aperture Coordinates along perimeter (Grid X, Y)
APERTURE_FRONT = (13, 4)
APERTURE_MID = (7, 4)
APERTURE_REAR = (1, 4)

# =====================================================================
# 2. VECTOR PATH GEOMETRY (STREET ROUTE: TERMINAL A TO TERMINAL B)
# =====================================================================
# Polyline waypoint coordinates representing urban street layout
MACRO_ROAD_POLYLINE = [
    (380, 440),  # Terminal 1 (South)
    (360, 400),
    (330, 350),
    (300, 300),  # Straight Avenue
    (260, 250),
    (230, 220),  # Bridge Bend
    (210, 195),
    (215, 175),  # River Crossing
    (245, 160),
    (290, 145),  # North-East Bend
    (340, 130),
    (365, 115),  # North Turn
    (355, 85),
    (335, 55)    # Terminal 20 (North)
]


def build_arc_length_parameterization(polyline, num_stops=20):
    """
    Computes Euclidean arc-length parameterization for a 2D polyline path
    and uniformly distributes stop nodes along the continuous trajectory.
    """
    segments = []
    total_length = 0.0
    for i in range(len(polyline) - 1):
        p1 = polyline[i]
        p2 = polyline[i + 1]
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
    MACRO_ROAD_POLYLINE, 20
)


# =====================================================================
# 3. PROCEDURAL SPRITES & RENDERING OPERATORS
# =====================================================================
def draw_seat(surface, rect):
    """Renders passenger seating topology."""
    pygame.draw.rect(surface, COLOR_SEAT_BASE, rect, border_radius=4)
    backrest = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, 6)
    pygame.draw.rect(surface, COLOR_SEAT_BACK, backrest, border_radius=2)


def draw_sliding_door(surface, rect, is_open):
    """Renders dual-leaf sliding aperture mechanics."""
    pygame.draw.rect(surface, (20, 20, 20), rect)
    if is_open:
        door_l = pygame.Rect(rect.left, rect.top, 4, rect.height)
        door_r = pygame.Rect(rect.right - 4, rect.top, 4, rect.height)
        pygame.draw.rect(surface, (46, 204, 113), door_l)
        pygame.draw.rect(surface, (46, 204, 113), door_r)
    else:
        pygame.draw.rect(surface, (120, 125, 135), rect)
        pygame.draw.line(surface, (60, 65, 75), (rect.centerx, rect.top), (rect.centerx, rect.bottom), 2)


def draw_humanoid(surface, rect, shirt_color, is_operator=False, is_ctrl=False):
    """Renders standard agent primitives (commuters, operators, patrol)."""
    cx, cy = rect.centerx, rect.centery
    torso = pygame.Rect(rect.left + 4, cy - 1, rect.width - 8, rect.height // 2)
    pygame.draw.rect(surface, shirt_color, torso, border_radius=4)
    pygame.draw.circle(surface, COLOR_SKIN, (cx, cy - 5), 4)
    if is_operator:
        pygame.draw.rect(surface, (20, 25, 35), (cx - 4, cy - 10, 8, 3))
    elif is_ctrl:
        pygame.draw.rect(surface, (150, 0, 0), (cx - 4, cy - 10, 8, 3))


def draw_audit_agent(surface, rect):
    """Renders the adversarial benchmark agent evaluating spatial occlusion."""
    cx, cy = rect.centerx, rect.centery
    body = pygame.Rect(rect.left + 4, cy - 1, rect.width - 8, rect.height // 2)
    pygame.draw.rect(surface, COLOR_TARGET_AGENT, body, border_radius=4)
    pygame.draw.circle(surface, COLOR_TARGET_AGENT, (cx, cy - 5), 4)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, (cx - 4, cy - 11, 3, 6), border_radius=1)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, (cx + 1, cy - 11, 3, 6), border_radius=1)


# =====================================================================
# 4. INTEGRATED DUAL-SCALE (MACRO-MICRO) ENGINE
# =====================================================================
class DualStateTransitEngine:
    """Coupled continuous-discrete simulation engine for macro-micro transit dynamics."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Qoyanshyq Engine // Macro Street Dynamics & Micro Audit Framework")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo, Consolas, monospace", 12)
        self.bold_font = pygame.font.SysFont("Menlo, Consolas, monospace", 14, bold=True)
        self.title_font = pygame.font.SysFont("Menlo, Consolas, monospace", 17, bold=True)

        self.frames_buffer = []
        self.reset()

    def reset(self):
        """Resets engine state variables to initial terminal boundary conditions."""
        self.current_distance = 0.0
        self.target_stop_idx = 1  # Initiate transit movement toward Stop 2
        self.move_speed = 4.0     # Macro translation velocity along arc length

        self.current_stop_num = 1
        self.is_at_stop = False
        self.dwell_ticks = 0

        # Micro vehicle interior state configuration
        self.operator_pos = (14, 1)      # Fixed driver cabin coordinate
        self.audit_target_seat = (6, 3)  # Benchmark node located in blind spot
        self.target_pos = None
        self.target_onboard = False
        self.target_uninspected_exit = False
        self.coverage_gap_detected = False

        self.commuters = set()
        self.patrol_units = set()
        self.doors_open = False

        self.status_msg = "TERMINAL STOP 1: Bus departed depot. Moving to Stop 2."

    def get_vacant_seats(self):
        """Returns list of unassigned seating coordinates within vehicle grid."""
        seats = []
        for x in range(1, 14):
            for y in [1, 3]:
                pos = (x, y)
                if pos != self.operator_pos and pos != self.audit_target_seat:
                    if pos not in self.commuters and pos not in self.patrol_units:
                        seats.append(pos)
        return seats

    def handle_stop_cycle(self):
        """Executes dynamic agent inflow, outflow, and patrol interception events."""
        stop = self.current_stop_num
        self.doors_open = True
        self.patrol_units.clear()

        # Stop 2: Ingress of adversarial benchmark agent
        if stop == 2 and not self.target_onboard and not self.coverage_gap_detected:
            self.target_onboard = True
            self.target_pos = self.audit_target_seat
            self.status_msg = f"STOP {stop}: Audit agent boarded. Seated at mid-right evaluation node."
            return

        # Stop 19: Planned terminal vulnerability assessment point
        if stop == 19 and self.target_onboard:
            self.target_onboard = False
            self.target_pos = None
            self.target_uninspected_exit = True
            self.status_msg = f"STOP {stop}: Complete transit uninspected. Vulnerability confirmed."
            return

        # Stop 20: Final terminal clearance
        if stop == 20:
            self.commuters.clear()
            self.status_msg = "TERMINAL STOP 20: Route completed. Interior cleared."
            return

        # Intermediate stops: Stochastic passenger turnover
        if len(self.commuters) > 0:
            leaving = random.randint(1, min(len(self.commuters), 4))
            for _ in range(leaving):
                self.commuters.pop()

        empty_seats = self.get_vacant_seats()
        if empty_seats:
            entering = random.randint(1, min(len(empty_seats), 5))
            for p in random.sample(empty_seats, entering):
                self.commuters.add(p)

        # Inspection Patrol Deployment (28% probability sweep trigger)
        if 3 <= stop <= 18 and self.target_onboard and random.random() < 0.28:
            dx, dy = APERTURE_FRONT
            self.patrol_units.add((dx, dy - 1))
            self.patrol_units.add((dx - 1, dy - 1))
            self.target_onboard = False
            self.coverage_gap_detected = True
            self.target_pos = None
            self.status_msg = f"STOP {stop}: PATROL INSPECTION: Audit agent executed evasion exit."
        else:
            if not self.coverage_gap_detected and not self.target_uninspected_exit:
                total_density = len(self.commuters) + int(self.target_onboard)
                self.status_msg = f"STOP {stop}: Flow exchange nominal. Interior density: {total_density} units."

    def update(self):
        """Updates continuous kinematics and discrete state transitions."""
        if self.is_at_stop:
            self.dwell_ticks += 1
            if self.dwell_ticks == 2:
                self.handle_stop_cycle()
            elif self.dwell_ticks >= 6:
                self.is_at_stop = False
                self.doors_open = False
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
            self.status_msg = "Macro-Micro Simulation complete. Saving GIF..."
            return False

    def draw(self):
        """Renders macro street topology, micro vehicle interior, and telemetry HUD."""
        self.screen.fill(COLOR_BG)

        # =============================================================
        # 1. MACRO STREET MAP (LEFT PANEL)
        # =============================================================
        map_rect = pygame.Rect(20, 20, 480, 470)
        pygame.draw.rect(self.screen, COLOR_MAP_BG, map_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, map_rect, width=2, border_radius=8)

        # River geometry
        river_pts = [(160, 230), (200, 210), (250, 190), (320, 185), (450, 200)]
        pygame.draw.lines(self.screen, COLOR_RIVER, False, river_pts, 18)

        # Polyline road infrastructure
        pygame.draw.lines(self.screen, COLOR_STREET_BASE, False, MACRO_ROAD_POLYLINE, 10)
        pygame.draw.lines(self.screen, COLOR_ROUTE_LINE, False, MACRO_ROAD_POLYLINE, 4)

        # Discrete stop nodes along polyline
        for idx, (sx, sy) in enumerate(STOP_COORDS):
            is_current = (idx + 1 == self.current_stop_num)
            color = COLOR_STOP_NODE if is_current else (240, 240, 240)
            radius = 7 if is_current else 5
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(sx), int(sy)), radius, width=1)

            lbl = self.font.render(str(idx + 1), True, (200, 205, 215))
            self.screen.blit(lbl, (sx + 7, sy - 7))

        # Macro vehicle translation and heading angle evaluation
        (bus_x, bus_y), bus_angle = SAMPLE_ROUTE_FN(self.current_distance)

        bus_surface = pygame.Surface((18, 10), pygame.SRCALPHA)
        bus_surface.fill(COLOR_BUS_MACRO)
        pygame.draw.rect(bus_surface, (255, 255, 255), (12, 2, 4, 6))
        rotated_bus = pygame.transform.rotate(bus_surface, -math.degrees(bus_angle))
        rot_rect = rotated_bus.get_rect(center=(int(bus_x), int(bus_y)))
        self.screen.blit(rotated_bus, rot_rect.topleft)

        map_title = self.bold_font.render("MACRO ROUTE TOPOLOGY // METROPOLITAN LINE 1", True, (220, 220, 220))
        self.screen.blit(map_title, (35, 30))

        # =============================================================
        # 2. MICRO BUS INTERIOR (RIGHT PANEL)
        # =============================================================
        micro_rect = pygame.Rect(520, 20, 580, 470)
        pygame.draw.rect(self.screen, COLOR_PANEL, micro_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, micro_rect, width=2, border_radius=8)

        micro_title = self.bold_font.render("MICRO STATE // CABIN OCCLUSION & APERTURE AUDIT", True, (220, 220, 220))
        self.screen.blit(micro_title, (540, 30))

        bus_ox, bus_oy = 560, 160

        hull_rect = pygame.Rect(bus_ox - 8, bus_oy - 8, GRID_BUS_W * CELL_SIZE + 16, GRID_BUS_H * CELL_SIZE + 16)
        pygame.draw.rect(self.screen, COLOR_BUS_FRAME, hull_rect, border_radius=10)
        pygame.draw.rect(self.screen, (20, 25, 35), hull_rect, width=3, border_radius=10)

        glass_rect = pygame.Rect(bus_ox + GRID_BUS_W * CELL_SIZE - 2, bus_oy + 4, 8, GRID_BUS_H * CELL_SIZE - 8)
        pygame.draw.rect(self.screen, (90, 180, 240), glass_rect, border_radius=3)

        # Floor grid and seat distribution
        for x in range(GRID_BUS_W):
            for y in range(GRID_BUS_H):
                cell_rect = pygame.Rect(bus_ox + x * CELL_SIZE, bus_oy + y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
                if (y == 1 or y == 3) and 1 <= x <= 13:
                    draw_seat(self.screen, cell_rect)
                elif y == 0 or y == 4:
                    pygame.draw.rect(self.screen, (180, 185, 195), cell_rect)
                else:
                    pygame.draw.rect(self.screen, COLOR_BUS_FLOOR, cell_rect)

        # Opaque driver cabin partition (Occlusion Generator)
        cab_wall = pygame.Rect(bus_ox + 13 * CELL_SIZE + 24, bus_oy + 1 * CELL_SIZE, 4, CELL_SIZE * 2)
        pygame.draw.rect(self.screen, COLOR_CABIN_WALL, cab_wall)

        # Sliding doors (Apertures)
        doors = [APERTURE_FRONT, APERTURE_MID, APERTURE_REAR]
        for (dx_d, dy_d) in doors:
            door_rect = pygame.Rect(bus_ox + dx_d * CELL_SIZE, bus_oy + dy_d * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_sliding_door(self.screen, door_rect, self.doors_open)

        # Render agents
        vx, vy = self.operator_pos
        d_rect = pygame.Rect(bus_ox + vx * CELL_SIZE, bus_oy + vy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
        draw_humanoid(self.screen, d_rect, COLOR_OPERATOR, is_operator=True)

        for (px, py) in self.commuters:
            p_rect = pygame.Rect(bus_ox + px * CELL_SIZE, bus_oy + py * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, p_rect, COLOR_COMMUTER)

        for (cx, cy) in self.patrol_units:
            c_rect = pygame.Rect(bus_ox + cx * CELL_SIZE, bus_oy + cy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, c_rect, COLOR_CTRL_UNIFORM, is_ctrl=True)

        if self.target_onboard and self.target_pos is not None:
            hx, hy = self.target_pos
            h_rect = pygame.Rect(bus_ox + hx * CELL_SIZE, bus_oy + hy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_audit_agent(self.screen, h_rect)

        # =============================================================
        # 3. HUD TELEMETRY & SYSTEM LOGS (BOTTOM PANEL)
        # =============================================================
        hud_rect = pygame.Rect(20, 505, 1080, 175)
        pygame.draw.rect(self.screen, COLOR_PANEL, hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, hud_rect, width=2, border_radius=8)

        msg_color = (240, 240, 240)
        if self.coverage_gap_detected:
            msg_color = COLOR_STOP_NODE
        elif self.target_uninspected_exit:
            msg_color = COLOR_TARGET_AGENT

        self.screen.blit(
            self.title_font.render(f"SYSTEM AUDIT MONITOR // NODE {self.current_stop_num} OF 20", True,
                                   (255, 255, 255)),
            (40, 520))
        self.screen.blit(self.bold_font.render(self.status_msg, True, msg_color), (40, 550))

        leg_y = 600
        draw_seat(self.screen, pygame.Rect(40, leg_y, 16, 16))
        self.screen.blit(self.font.render("Passenger Seat", True, (190, 195, 205)), (65, leg_y))

        draw_audit_agent(self.screen, pygame.Rect(190, leg_y, 16, 16))
        self.screen.blit(self.font.render("Audit Agent (Node)", True, (190, 195, 205)), (215, leg_y))

        draw_humanoid(self.screen, pygame.Rect(370, leg_y, 16, 16), COLOR_CTRL_UNIFORM, is_ctrl=True)
        self.screen.blit(self.font.render("Inspection Patrol", True, (240, 100, 100)), (395, leg_y))

        draw_humanoid(self.screen, pygame.Rect(550, leg_y, 16, 16), COLOR_COMMUTER)
        self.screen.blit(self.font.render("Commuter Flow", True, (190, 195, 205)), (575, leg_y))

        door_txt = "APERTURES: OPEN" if self.doors_open else "APERTURES: LOCKED"
        door_col = (46, 204, 113) if self.doors_open else (150, 155, 165)
        self.screen.blit(self.bold_font.render(door_txt, True, door_col), (780, leg_y))

        self.screen.blit(
            self.font.render("Commands: [R] Reset Benchmark | [SPACE] Pause Step Execution", True, (120, 130, 150)),
            (40, leg_y + 35))

        pygame.display.flip()

    def record_frame(self):
        """Scales current screen state to 560x350 and appends to GIF buffer."""
        scaled = pygame.transform.smoothscale(self.screen, (560, 350))
        view = pygame.surfarray.array3d(scaled)
        view = view.transpose([1, 0, 2])
        self.frames_buffer.append(view)

    def save_gif(self, filename="macro_micro_simulation.gif", target_fps=12):
        """Compiles recorded frame sequence into a lightweight GIF file."""
        if not self.frames_buffer:
            print("[-] Frames buffer is empty.")
            return

        file_path = os.path.abspath(__file__)
        base_dir = os.path.dirname(os.path.dirname(file_path)) if ".venv" in file_path else os.path.dirname(file_path)
        output_path = os.path.join(base_dir, filename)

        print(f"\n[+] Compiling dual-view GIF from {len(self.frames_buffer)} frames...")
        sampled_frames = self.frames_buffer[::2]
        imageio.mimsave(output_path, sampled_frames, fps=target_fps, loop=0)
        print(f"[✓] Successfully saved dual-scale recording to: {output_path}")

    def run(self):
        """Main execution loop for real-time visualization and animation capture."""
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
                    is_active = self.update()
                    if is_active is False:
                        running = False

                self.draw()

                if len(self.frames_buffer) < 400:
                    self.record_frame()
        finally:
            self.save_gif("macro_micro_simulation.gif", target_fps=12)
            pygame.quit()


if __name__ == "__main__":
    engine = DualStateTransitEngine()
    engine.run()