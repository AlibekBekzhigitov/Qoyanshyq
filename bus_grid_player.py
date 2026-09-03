import os
import sys
import random
import pygame
import imageio

# =====================================================================
# 1. CONSTANTS & DISPLAY SETUP
# =====================================================================
SCREEN_W = 1060
SCREEN_H = 700

CELL_SIZE = 30
GRID_BUS_W = 16
GRID_BUS_H = 5
FPS = 6

# Color Palette
COLOR_BG = (15, 17, 23)
COLOR_PANEL = (24, 28, 38)
COLOR_ROAD = (32, 38, 50)
COLOR_STOP = (241, 196, 15)

COLOR_BUS_EXT_BODY = (41, 128, 185)
COLOR_BUS_EXT_HEAD = (52, 152, 219)

COLOR_BUS_FRAME = (50, 55, 68)
COLOR_BUS_FLOOR = (215, 220, 228)
COLOR_SEAT_BASE = (74, 105, 189)
COLOR_SEAT_BACK = (44, 62, 80)
COLOR_CABIN_WALL = (30, 39, 46)

# Agent colors
COLOR_TARGET_AGENT = (46, 204, 113)  # Audit benchmark agent (emerald green)
COLOR_TARGET_TAG = (255, 255, 255)
COLOR_CTRL_UNIFORM = (214, 48, 49)   # Inspection patrol (crimson red)
COLOR_COMMUTER = (142, 68, 173)      # Commuter background flow (purple)
COLOR_OPERATOR = (45, 52, 54)        # Driver / Transit operator
COLOR_SKIN = (255, 218, 185)

# Transit vehicle apertures along bottom perimeter (Y = 4)
APERTURE_FRONT = (13, 4)
APERTURE_MID = (7, 4)
APERTURE_REAR = (1, 4)

# =====================================================================
# 2. METROPOLITAN ROUTE GENERATION (20 STOPS)
# =====================================================================
CITY_COLS = 36
CITY_ROWS = 10

route_cells = []
for row in [2, 4, 6, 8]:
    col_range = range(2, 34) if (row // 2) % 2 == 1 else range(33, 1, -1)
    for col in col_range:
        route_cells.append((col, row))

step_stride = len(route_cells) // 20
STOPS = [route_cells[i * step_stride] for i in range(20)]


# =====================================================================
# 3. PROCEDURAL SPRITE ASSETS
# =====================================================================
def draw_seat(surface, rect):
    pygame.draw.rect(surface, COLOR_SEAT_BASE, rect, border_radius=4)
    backrest = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, 7)
    pygame.draw.rect(surface, COLOR_SEAT_BACK, backrest, border_radius=2)
    pygame.draw.rect(surface, (30, 45, 75), rect, width=1, border_radius=4)


def draw_sliding_door(surface, rect, is_open):
    pygame.draw.rect(surface, (20, 20, 20), rect)
    if is_open:
        door_l = pygame.Rect(rect.left, rect.top, 5, rect.height)
        door_r = pygame.Rect(rect.right - 5, rect.top, 5, rect.height)
        pygame.draw.rect(surface, (46, 204, 113), door_l)
        pygame.draw.rect(surface, (46, 204, 113), door_r)
        pygame.draw.line(surface, (46, 204, 113), (rect.left + 5, rect.bottom - 2), (rect.right - 5, rect.bottom - 2), 2)
    else:
        pygame.draw.rect(surface, (120, 125, 135), rect)
        pygame.draw.line(surface, (60, 65, 75), (rect.centerx, rect.top), (rect.centerx, rect.bottom), 2)


def draw_humanoid(surface, rect, shirt_color, is_operator=False, is_ctrl=False):
    cx, cy = rect.centerx, rect.centery
    torso = pygame.Rect(rect.left + 5, cy - 1, rect.width - 10, rect.height // 2)
    pygame.draw.rect(surface, shirt_color, torso, border_radius=4)

    head_pos = (cx, cy - 6)
    pygame.draw.circle(surface, COLOR_SKIN, head_pos, 5)

    if is_operator:
        cap = pygame.Rect(head_pos[0] - 5, head_pos[1] - 6, 10, 3)
        pygame.draw.rect(surface, (20, 25, 35), cap)
    elif is_ctrl:
        cap = pygame.Rect(head_pos[0] - 5, head_pos[1] - 6, 10, 3)
        pygame.draw.rect(surface, (150, 0, 0), cap)


def draw_audit_agent(surface, rect):
    """Renders the adversarial benchmark agent evaluating blind spots."""
    cx, cy = rect.centerx, rect.centery
    body = pygame.Rect(rect.left + 5, cy - 1, rect.width - 10, rect.height // 2)
    pygame.draw.rect(surface, COLOR_TARGET_AGENT, body, border_radius=4)

    head_pos = (cx, cy - 5)
    pygame.draw.circle(surface, COLOR_TARGET_AGENT, head_pos, 5)

    # Benchmark sensor/tag distinction
    tag_w, tag_h = 3, 8
    tag_l = pygame.Rect(head_pos[0] - 4, head_pos[1] - 12, tag_w, tag_h)
    tag_r = pygame.Rect(head_pos[0] + 1, head_pos[1] - 12, tag_w, tag_h)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, tag_l, border_radius=2)
    pygame.draw.rect(surface, COLOR_TARGET_TAG, tag_r, border_radius=2)


# =====================================================================
# 4. SIMULATION & AUDIT ENGINE
# =====================================================================
class BusGridPlayer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Transit Security Audit: Spatial Bottleneck & Coverage Engine")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo, Consolas, monospace", 13)
        self.bold_font = pygame.font.SysFont("Menlo, Consolas, monospace", 15, bold=True)
        self.title_font = pygame.font.SysFont("Menlo, Consolas, monospace", 18, bold=True)

        self.frames_buffer = []
        self.reset()

    def reset(self):
        self.bus_route_idx = 0
        self.bus_segments = [route_cells[2], route_cells[1], route_cells[0]]

        self.current_stop_num = 1
        self.is_at_stop = False
        self.dwell_ticks = 0

        # Interior spatial allocations
        self.operator_pos = (14, 1)      # Fixed driver position
        self.audit_target_seat = (6, 3)  # Benchmark node seated mid-right aisle
        self.target_pos = None
        self.target_onboard = False
        self.target_uninspected_exit = False
        self.coverage_gap_detected = False

        self.commuters = set()
        self.patrol_units = set()
        self.doors_open = False

        self.status_msg = "TERMINAL STOP 1: Bus departed depot with operator. Interior empty."

    def get_vacant_seats(self):
        seats = []
        for x in range(1, 14):
            for y in [1, 3]:
                pos = (x, y)
                if pos != self.operator_pos and pos != self.audit_target_seat:
                    if pos not in self.commuters and pos not in self.patrol_units:
                        seats.append(pos)
        return seats

    def handle_stop_cycle(self):
        stop = self.current_stop_num
        self.doors_open = True
        self.patrol_units.clear()

        # Stop 2: Ingress of benchmark audit agent
        if stop == 2 and not self.target_onboard and not self.coverage_gap_detected:
            self.target_onboard = True
            self.target_pos = self.audit_target_seat
            self.status_msg = "STOP 2: Audit benchmark agent boarded. Seated at mid-right evaluation node."
            return

        # Stop 19: Planned terminal evaluation point
        if stop == 19 and self.target_onboard:
            self.target_onboard = False
            self.target_pos = None
            self.target_uninspected_exit = True
            self.status_msg = "STOP 19: Route completed without inspection. Coverage vulnerability confirmed."
            return

        # Stop 20: Final terminal stop
        if stop == 20:
            self.commuters.clear()
            self.status_msg = "TERMINAL STOP 20: Route ended. All remaining commuter nodes cleared."
            return

        # Intermediate stops: Stochastic passenger turnover
        if len(self.commuters) > 0:
            leaving = random.randint(1, min(len(self.commuters), 4))
            for _ in range(leaving):
                self.commuters.pop()

        empty_seats = self.get_vacant_seats()
        entering = random.randint(1, min(len(empty_seats), 5))
        for p in random.sample(empty_seats, entering):
            self.commuters.add(p)

        # Inspection Sweep Deployment (28% probability between stops 3-18)
        if 3 <= stop <= 18 and self.target_onboard and random.random() < 0.28:
            num_ctrls = random.randint(2, 3)
            dx, dy = APERTURE_FRONT
            self.patrol_units.add((dx, dy - 1))
            self.patrol_units.add((dx - 1, dy - 1))
            if num_ctrls == 3:
                self.patrol_units.add((dx + 1, dy - 1))

            # Spatial Egress Dynamic: Benchmark node utilizes alternate aperture before patrol seals corridor
            self.target_onboard = False
            self.coverage_gap_detected = True
            self.target_pos = None  # Node exits vehicle topology completely
            self.status_msg = f"STOP {stop}: AUDIT EVENT: Single-door ingress latency permitted egress slip."
        else:
            if not self.coverage_gap_detected and not self.target_uninspected_exit:
                total_density = len(self.commuters) + int(self.target_onboard)
                self.status_msg = f"STOP {stop}: Flow exchange nominal. Interior vehicle density: {total_density} units."

    def update(self):
        if self.is_at_stop:
            self.dwell_ticks += 1
            if self.dwell_ticks == 2:
                self.handle_stop_cycle()
            elif self.dwell_ticks >= 7:
                self.is_at_stop = False
                self.doors_open = False
                self.dwell_ticks = 0
                self.bus_route_idx += 1
            return True

        if self.bus_route_idx + 1 < len(route_cells):
            self.bus_route_idx += 1
            head = route_cells[self.bus_route_idx]
            self.bus_segments.insert(0, head)
            self.bus_segments.pop()

            if head in STOPS:
                self.is_at_stop = True
                self.current_stop_num = STOPS.index(head) + 1
            return True
        else:
            self.status_msg = "Route complete. Saving recording..."
            return False

    def draw(self):
        self.screen.fill(COLOR_BG)

        # -------------------------------------------------------------
        # 1. TOP OVERVIEW: CITY TOPOLOGY & NETWORK NODES
        # -------------------------------------------------------------
        city_ox, city_oy = 40, 45
        c_size = 18

        for (cx, cy) in route_cells:
            r = pygame.Rect(city_ox + cx * c_size, city_oy + cy * c_size, c_size - 2, c_size - 2)
            pygame.draw.rect(self.screen, COLOR_ROAD, r)

        for idx, (sx, sy) in enumerate(STOPS):
            r = pygame.Rect(city_ox + sx * c_size, city_oy + sy * c_size, c_size - 2, c_size - 2)
            pygame.draw.rect(self.screen, COLOR_STOP, r)
            lbl = self.font.render(str(idx + 1), True, (0, 0, 0))
            self.screen.blit(lbl, (r.x + 2, r.y + 1))

        for i, (bx, by) in enumerate(self.bus_segments):
            r = pygame.Rect(city_ox + bx * c_size, city_oy + by * c_size, c_size - 2, c_size - 2)
            color = COLOR_BUS_EXT_HEAD if i == 0 else COLOR_BUS_EXT_BODY
            pygame.draw.rect(self.screen, color, r, border_radius=4)

        # -------------------------------------------------------------
        # 2. BOTTOM OVERVIEW: CABIN GEOMETRY & APERTURES
        # -------------------------------------------------------------
        bus_ox, bus_oy = 150, 310

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
                    pygame.draw.line(self.screen, (195, 200, 210), (cell_rect.left, cell_rect.bottom),
                                     (cell_rect.right, cell_rect.bottom))

        cab_wall = pygame.Rect(bus_ox + 13 * CELL_SIZE + 26, bus_oy + 1 * CELL_SIZE, 3, CELL_SIZE * 2)
        pygame.draw.rect(self.screen, COLOR_CABIN_WALL, cab_wall)

        doors = [(APERTURE_FRONT, "FRONT"), (APERTURE_MID, "MIDDLE"), (APERTURE_REAR, "REAR")]
        for (dx, dy), name in doors:
            door_rect = pygame.Rect(bus_ox + dx * CELL_SIZE, bus_oy + dy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_sliding_door(self.screen, door_rect, self.doors_open)

        # 1. Driver / Transit Operator
        vx, vy = self.operator_pos
        d_rect = pygame.Rect(bus_ox + vx * CELL_SIZE, bus_oy + vy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
        draw_humanoid(self.screen, d_rect, COLOR_OPERATOR, is_operator=True)

        # 2. Commuter Flow
        for (px, py) in self.commuters:
            p_rect = pygame.Rect(bus_ox + px * CELL_SIZE, bus_oy + py * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, p_rect, COLOR_COMMUTER)

        # 3. Patrol Sweep Units
        for (cx, cy) in self.patrol_units:
            c_rect = pygame.Rect(bus_ox + cx * CELL_SIZE, bus_oy + cy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, c_rect, COLOR_CTRL_UNIFORM, is_ctrl=True)

        # 4. Audit Benchmark Agent (Only displayed when physically onboard)
        if self.target_onboard and self.target_pos is not None:
            hx, hy = self.target_pos
            h_rect = pygame.Rect(bus_ox + hx * CELL_SIZE, bus_oy + hy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_audit_agent(self.screen, h_rect)

        # -------------------------------------------------------------
        # 3. TELEMETRY & HUD METRICS
        # -------------------------------------------------------------
        hud_y = 510
        msg_color = (240, 240, 240)
        if self.coverage_gap_detected:
            msg_color = COLOR_STOP
        elif self.target_uninspected_exit:
            msg_color = (46, 204, 113)

        self.screen.blit(
            self.title_font.render(f"TOPOLOGICAL AUDIT MONITOR // NODE {self.current_stop_num} OF 20", True, (255, 255, 255)),
            (40, hud_y))
        self.screen.blit(self.bold_font.render(self.status_msg, True, msg_color), (40, hud_y + 30))

        leg_y = hud_y + 70

        # Legend descriptors
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
        self.screen.blit(self.bold_font.render(door_txt, True, door_col), (750, leg_y))

        self.screen.blit(self.font.render("Commands: [R] Reset Benchmark | [SPACE] Pause Step Execution", True, (120, 130, 150)),
                         (40, leg_y + 35))

        pygame.display.flip()

    def record_frame(self):
        view = pygame.surfarray.array3d(self.screen)
        view = view.transpose([1, 0, 2])
        self.frames_buffer.append(view)

    def save_gif(self, filename="simulation.gif", target_fps=8):
        if not self.frames_buffer:
            print("[-] Frames buffer is empty, nothing to save.")
            return

        # Гарантированное сохранение в корень проекта
        file_path = os.path.abspath(__file__)
        if ".venv" in file_path:
            base_dir = os.path.dirname(os.path.dirname(file_path))
        else:
            base_dir = os.path.dirname(file_path)

        output_path = os.path.join(base_dir, filename)

        print(f"\n[+] Compiling GIF from {len(self.frames_buffer)} frames...")
        sampled_frames = self.frames_buffer[::2]
        imageio.mimsave(output_path, sampled_frames, fps=target_fps, loop=0)
        print(f"[✓] Successfully saved to: {output_path}")

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
                    is_active = self.update()
                    if is_active is False:
                        running = False

                self.draw()

                if len(self.frames_buffer) < 400:
                    self.record_frame()
        finally:
            self.save_gif("simulation.gif", target_fps=8)
            pygame.quit()


if __name__ == "__main__":
    game = BusGridPlayer()
    game.run()