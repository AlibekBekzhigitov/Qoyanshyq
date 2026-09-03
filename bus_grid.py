import pygame
import random
import sys

# =====================================================================
# 1. GRID & DISPLAY CONFIGURATION
# =====================================================================
CELL_SIZE = 30
GRID_BUS_W, GRID_BUS_H = 16, 5

SCREEN_W, SCREEN_H = 1060, 700
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
COLOR_HARE = (46, 204, 113)  # Distinct green
COLOR_HARE_EARS = (255, 255, 255)
COLOR_CTRL_UNIFORM = (214, 48, 49)  # Crimson Red
COLOR_PASSENGER = (142, 68, 173)  # Commuter Purple
COLOR_DRIVER = (45, 52, 54)
COLOR_SKIN = (255, 218, 185)

# Doors at Y = 4
DOOR_FRONT = (13, 4)
DOOR_MID = (7, 4)
DOOR_REAR = (1, 4)

# =====================================================================
# 2. CITY ROUTE (20 STOPS)
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
# 3. SPRITE DRAWING
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
        pygame.draw.line(surface, (46, 204, 113), (rect.left + 5, rect.bottom - 2), (rect.right - 5, rect.bottom - 2),
                         2)
    else:
        pygame.draw.rect(surface, (120, 125, 135), rect)
        pygame.draw.line(surface, (60, 65, 75), (rect.centerx, rect.top), (rect.centerx, rect.bottom), 2)


def draw_humanoid(surface, rect, shirt_color, is_driver=False, is_ctrl=False):
    cx, cy = rect.centerx, rect.centery
    torso = pygame.Rect(rect.left + 5, cy - 1, rect.width - 10, rect.height // 2)
    pygame.draw.rect(surface, shirt_color, torso, border_radius=4)

    head_pos = (cx, cy - 6)
    pygame.draw.circle(surface, COLOR_SKIN, head_pos, 5)

    if is_driver:
        cap = pygame.Rect(head_pos[0] - 5, head_pos[1] - 6, 10, 3)
        pygame.draw.rect(surface, (20, 25, 35), cap)
    elif is_ctrl:
        cap = pygame.Rect(head_pos[0] - 5, head_pos[1] - 6, 10, 3)
        pygame.draw.rect(surface, (150, 0, 0), cap)


def draw_hare_agent(surface, rect):
    cx, cy = rect.centerx, rect.centery
    body = pygame.Rect(rect.left + 5, cy - 1, rect.width - 10, rect.height // 2)
    pygame.draw.rect(surface, COLOR_HARE, body, border_radius=4)

    head_pos = (cx, cy - 5)
    pygame.draw.circle(surface, COLOR_HARE, head_pos, 5)

    # White bunny ears
    ear_w, ear_h = 3, 8
    ear_l = pygame.Rect(head_pos[0] - 4, head_pos[1] - 12, ear_w, ear_h)
    ear_r = pygame.Rect(head_pos[0] + 1, head_pos[1] - 12, ear_w, ear_h)
    pygame.draw.rect(surface, COLOR_HARE_EARS, ear_l, border_radius=2)
    pygame.draw.rect(surface, COLOR_HARE_EARS, ear_r, border_radius=2)


# =====================================================================
# 4. GAME & STATE MANAGEMENT
# =====================================================================
class BusTransitGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Transit Evasion Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo, Consolas, monospace", 13)
        self.bold_font = pygame.font.SysFont("Menlo, Consolas, monospace", 15, bold=True)
        self.title_font = pygame.font.SysFont("Menlo, Consolas, monospace", 18, bold=True)
        self.reset()

    def reset(self):
        self.bus_route_idx = 0
        self.bus_segments = [route_cells[2], route_cells[1], route_cells[0]]

        self.current_stop_num = 1
        self.is_at_stop = False
        self.dwell_ticks = 0

        # Interior salon state
        self.driver_pos = (14, 1)  # Fixed driver position
        self.hare_seat = (6, 3)  # Optimal seat: right perimeter, mid-salon
        self.hare_pos = None
        self.hare_in_bus = False
        self.hare_escaped = False
        self.hare_reached_dest = False

        self.passengers = set()
        self.controllers = set()
        self.doors_open = False

        self.status_msg = "TERMINAL STOP 1: Bus departed empty with driver."

    def get_vacant_seats(self):
        seats = []
        for x in range(1, 14):
            for y in [1, 3]:
                pos = (x, y)
                if pos != self.driver_pos and pos != self.hare_seat:
                    if pos not in self.passengers and pos not in self.controllers:
                        seats.append(pos)
        return seats

    def handle_stop_cycle(self):
        stop = self.current_stop_num
        self.doors_open = True
        self.controllers.clear()

        # Stop 2: Hare boards the bus
        if stop == 2 and not self.hare_in_bus and not self.hare_escaped:
            self.hare_in_bus = True
            self.hare_pos = self.hare_seat
            self.status_msg = "STOP 2: Target (Hare) boarded and seated on right side near middle."
            return

        # Stop 19: Planned safe exit if no raid happened
        if stop == 19 and self.hare_in_bus:
            self.hare_in_bus = False
            self.hare_pos = None
            self.hare_reached_dest = True
            self.status_msg = "STOP 19: Hare disembarked safely at destination! Evasion success."
            return

        # Stop 20: Terminal stop
        if stop == 20:
            self.passengers.clear()
            self.status_msg = "TERMINAL STOP 20: Route completed. All remaining passengers exited."
            return

        # Intermediate stops: Passenger turnover
        if len(self.passengers) > 0:
            leaving = random.randint(1, min(len(self.passengers), 4))
            for _ in range(leaving):
                self.passengers.pop()

        empty_seats = self.get_vacant_seats()
        entering = random.randint(1, min(len(empty_seats), 5))
        for p in random.sample(empty_seats, entering):
            self.passengers.add(p)

        # Inspection raid: occurs between stops 3-18 with 28% chance
        if 3 <= stop <= 18 and self.hare_in_bus and random.random() < 0.28:
            # Inspectors ALWAYS enter via Front (Driver) Door
            num_ctrls = random.randint(2, 3)
            dx, dy = DOOR_FRONT
            self.controllers.add((dx, dy - 1))
            self.controllers.add((dx - 1, dy - 1))
            if num_ctrls == 3:
                self.controllers.add((dx + 1, dy - 1))

            # REACTION: Hare spots danger, IMMEDIATELY exits bus and completely vanishes
            self.hare_in_bus = False
            self.hare_escaped = True
            self.hare_pos = None  # Removed from bus grid completely!
            self.status_msg = f"STOP {stop}: RAID! {num_ctrls} Patrols entered front door. Hare exited and left!"
        else:
            if not self.hare_escaped and not self.hare_reached_dest:
                total_load = len(self.passengers) + int(self.hare_in_bus)
                self.status_msg = f"STOP {stop}: Normal stop turnover. Bus load: {total_load} commuters."

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
            return

        if self.bus_route_idx + 1 < len(route_cells):
            self.bus_route_idx += 1
            head = route_cells[self.bus_route_idx]
            self.bus_segments.insert(0, head)
            self.bus_segments.pop()

            if head in STOPS:
                self.is_at_stop = True
                self.current_stop_num = STOPS.index(head) + 1
        else:
            self.status_msg = "Route complete. Press [R] to reload simulation."

    def draw(self):
        self.screen.fill(COLOR_BG)

        # -------------------------------------------------------------
        # 1. TOP OVERVIEW: CITY MAP
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
        # 2. BOTTOM OVERVIEW: INTERIOR BUS BLUEPRINT
        # -------------------------------------------------------------
        bus_ox, bus_oy = 150, 310

        # Outer Hull
        hull_rect = pygame.Rect(bus_ox - 8, bus_oy - 8, GRID_BUS_W * CELL_SIZE + 16, GRID_BUS_H * CELL_SIZE + 16)
        pygame.draw.rect(self.screen, COLOR_BUS_FRAME, hull_rect, border_radius=10)
        pygame.draw.rect(self.screen, (20, 25, 35), hull_rect, width=3, border_radius=10)

        # Front windshield
        glass_rect = pygame.Rect(bus_ox + GRID_BUS_W * CELL_SIZE - 2, bus_oy + 4, 8, GRID_BUS_H * CELL_SIZE - 8)
        pygame.draw.rect(self.screen, (90, 180, 240), glass_rect, border_radius=3)

        # Tiles and seats
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

        # Driver Partition
        cab_wall = pygame.Rect(bus_ox + 13 * CELL_SIZE + 26, bus_oy + 1 * CELL_SIZE, 3, CELL_SIZE * 2)
        pygame.draw.rect(self.screen, COLOR_CABIN_WALL, cab_wall)

        # Doors along bottom
        doors = [(DOOR_FRONT, "FRONT"), (DOOR_MID, "MIDDLE"), (DOOR_REAR, "REAR")]
        for (dx, dy), name in doors:
            door_rect = pygame.Rect(bus_ox + dx * CELL_SIZE, bus_oy + dy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_sliding_door(self.screen, door_rect, self.doors_open)

        # 1. Driver
        vx, vy = self.driver_pos
        d_rect = pygame.Rect(bus_ox + vx * CELL_SIZE, bus_oy + vy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
        draw_humanoid(self.screen, d_rect, COLOR_DRIVER, is_driver=True)

        # 2. Commuters
        for (px, py) in self.passengers:
            p_rect = pygame.Rect(bus_ox + px * CELL_SIZE, bus_oy + py * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, p_rect, COLOR_PASSENGER)

        # 3. Patrol units
        for (cx, cy) in self.controllers:
            c_rect = pygame.Rect(bus_ox + cx * CELL_SIZE, bus_oy + cy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_humanoid(self.screen, c_rect, COLOR_CTRL_UNIFORM, is_ctrl=True)

        # 4. Hare Agent: ТОЛЬКО если заяц физически находится внутри автобуса
        if self.hare_in_bus and self.hare_pos is not None:
            hx, hy = self.hare_pos
            h_rect = pygame.Rect(bus_ox + hx * CELL_SIZE, bus_oy + hy * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            draw_hare_agent(self.screen, h_rect)

        # -------------------------------------------------------------
        # 3. HUD & STATUS
        # -------------------------------------------------------------
        hud_y = 510
        msg_color = (240, 240, 240)
        if self.hare_escaped:
            msg_color = COLOR_STOP
        elif self.hare_reached_dest:
            msg_color = (46, 204, 113)

        self.screen.blit(
            self.title_font.render(f"ROUTE MONITOR // STOP {self.current_stop_num} OF 20", True, (255, 255, 255)),
            (40, hud_y))
        self.screen.blit(self.bold_font.render(self.status_msg, True, msg_color), (40, hud_y + 30))

        leg_y = hud_y + 70

        # Legend items
        draw_seat(self.screen, pygame.Rect(40, leg_y, 16, 16))
        self.screen.blit(self.font.render("Passenger Seat", True, (190, 195, 205)), (65, leg_y))

        draw_hare_agent(self.screen, pygame.Rect(190, leg_y, 16, 16))
        self.screen.blit(self.font.render("Hare (Target)", True, (190, 195, 205)), (215, leg_y))

        draw_humanoid(self.screen, pygame.Rect(350, leg_y, 16, 16), COLOR_CTRL_UNIFORM, is_ctrl=True)
        self.screen.blit(self.font.render("Patrol (Front Door)", True, (240, 100, 100)), (375, leg_y))

        draw_humanoid(self.screen, pygame.Rect(540, leg_y, 16, 16), COLOR_PASSENGER)
        self.screen.blit(self.font.render("Commuter", True, (190, 195, 205)), (565, leg_y))

        door_txt = "DOORS: OPEN" if self.doors_open else "DOORS: CLOSED"
        door_col = (46, 204, 113) if self.doors_open else (150, 155, 165)
        self.screen.blit(self.bold_font.render(door_txt, True, door_col), (750, leg_y))

        self.screen.blit(self.font.render("Commands: [R] Reset | [SPACE] Pause", True, (120, 130, 150)),
                         (40, leg_y + 35))

        pygame.display.flip()

    def run(self):
        running = True
        paused = False
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                self.update()
            self.draw()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = BusTransitGame()
    game.run()

