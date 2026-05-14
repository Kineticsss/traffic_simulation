"""
=============================================================
  CS 324: Modeling and Simulation
  Final Project: Traffic Light System Simulation
  Batangas State University — CICS
  Tools: Python, SimPy, Pygame, Matplotlib, Pandas
=============================================================
"""

import simpy
import pygame
import random
import math
import sys
import threading
import time
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 780

# Traffic light timing (seconds)
SCENARIOS = {
    "Normal Traffic": {
        "green":  30,
        "yellow":  5,
        "red":    35,
        "arrival_rate": 4.0,   # avg seconds between vehicles per lane
    },
    "Rush Hour": {
        "green":  45,
        "yellow":  5,
        "red":    50,
        "arrival_rate": 1.5,
    },
    "Low Traffic": {
        "green":  20,
        "yellow":  3,
        "red":    23,
        "arrival_rate": 8.0,
    },
}

# Colors — professional dark theme
C = {
    "bg":           (15,  17,  23),
    "road":         (30,  32,  40),
    "road_line":    (60,  63,  75),
    "sidewalk":     (40,  43,  52),
    "panel":        (22,  25,  35),
    "panel2":       (28,  32,  45),
    "border":       (50,  55,  75),
    "text":         (220, 225, 240),
    "text_dim":     (120, 128, 150),
    "accent":       (80, 160, 255),
    "accent2":      (255, 180,  50),
    "green_light":  ( 50, 220,  90),
    "yellow_light": (255, 210,  50),
    "red_light":    (255,  60,  60),
    "car_colors": [
        (100, 180, 255), (255, 120,  80), (120, 220, 140),
        (255, 210,  80), (180, 120, 255), (255, 160, 100),
        ( 80, 220, 220), (200, 200, 200),
    ],
}

ROAD_W = 380   # total intersection road width
LANE_W =  52   # single lane width

# ─────────────────────────────────────────────
#  SIMULATION ENGINE  (SimPy)
# ─────────────────────────────────────────────
class SimData:
    """Shared state between SimPy thread and Pygame thread."""
    def __init__(self):
        self.lock = threading.Lock()
        self.vehicles: list[dict] = []          # live vehicle list
        self.completed: list[dict] = []         # finished vehicles
        self.light_ns = "green"                 # N-S light state
        self.light_ew = "red"                   # E-W light state
        self.light_timer = 0                    # countdown seconds
        self.sim_time = 0.0
        self.total_vehicles = 0
        self.scenario = "Normal Traffic"
        self.running = True
        self.speed_factor = 0.5                 # simulation speed multiplier (0.5 = comfortable real-time feel)
        self.wait_times: list[float] = []
        self.throughput_log: list[tuple] = []   # (sim_time, count)

SD = SimData()

# Directions: 0=North, 1=East, 2=South, 3=West
DIR_NAMES = ["North", "East", "South", "West"]

def run_simulation(sd: SimData):
    """SimPy simulation loop — runs in a background thread."""
    env = simpy.Environment()

    def traffic_light_controller(env):
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            green_t  = cfg["green"]
            yellow_t = cfg["yellow"]
            red_t    = cfg["red"]
            cycle    = green_t + yellow_t + red_t + yellow_t  # full cycle

            # Phase 1 — N-S green
            with sd.lock:
                sd.light_ns = "green"
                sd.light_ew = "red"
                sd.light_timer = green_t
            for t in range(green_t):
                if not sd.running: return
                yield env.timeout(1 / sd.speed_factor)
                with sd.lock:
                    sd.light_timer = green_t - t - 1
                    sd.sim_time = env.now

            # Phase 2 — N-S yellow
            with sd.lock:
                sd.light_ns = "yellow"
                sd.light_timer = yellow_t
            for t in range(yellow_t):
                if not sd.running: return
                yield env.timeout(1 / sd.speed_factor)
                with sd.lock:
                    sd.light_timer = yellow_t - t - 1

            # Phase 3 — E-W green
            with sd.lock:
                sd.light_ns = "red"
                sd.light_ew = "green"
                sd.light_timer = red_t
            for t in range(red_t):
                if not sd.running: return
                yield env.timeout(1 / sd.speed_factor)
                with sd.lock:
                    sd.light_timer = red_t - t - 1

            # Phase 4 — E-W yellow
            with sd.lock:
                sd.light_ew = "yellow"
                sd.light_timer = yellow_t
            for t in range(yellow_t):
                if not sd.running: return
                yield env.timeout(1 / sd.speed_factor)
                with sd.lock:
                    sd.light_timer = yellow_t - t - 1

    def vehicle_generator(env, direction):
        """Spawn vehicles from one direction."""
        vid = 0
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            interarrival = random.expovariate(1.0 / cfg["arrival_rate"])
            yield env.timeout(interarrival / sd.speed_factor)

            vid += 1
            veh_id = f"{DIR_NAMES[direction][0]}{vid}"
            arrive_t = env.now

            with sd.lock:
                sd.total_vehicles += 1
                color = random.choice(C["car_colors"])
                sd.vehicles.append({
                    "id":       veh_id,
                    "dir":      direction,
                    "state":    "queued",   # queued / moving / done
                    "arrive":   arrive_t,
                    "depart":   None,
                    "wait":     0.0,
                    "progress": 0.0,        # 0..1 through intersection
                    "color":    color,
                    "lane":     random.choice([-1, 1]),  # left/right lane
                })

    def vehicle_mover(env):
        """Advance vehicle positions every tick."""
        while sd.running:
            yield env.timeout(0.05 / sd.speed_factor)
            with sd.lock:
                light_ns = sd.light_ns
                light_ew = sd.light_ew
                now = env.now
                remove = []

                # group queued vehicles by direction, sort by arrival
                queues: dict[int, list] = {0:[], 1:[], 2:[], 3:[]}
                for v in sd.vehicles:
                    if v["state"] in ("queued", "moving"):
                        queues[v["dir"]].append(v)
                for q in queues.values():
                    q.sort(key=lambda x: x["arrive"])

                # Assign queue_slot so renderer stacks them at the stop line
                for q in queues.values():
                    slot = 0
                    for vv in q:
                        if vv["state"] == "queued":
                            vv["queue_slot"] = slot
                            slot += 1

                # Build progress map per direction for leader lookups
                # progress_by_dir[dir] = sorted list of progress values (moving vehicles)
                progress_by_dir: dict[int, list[float]] = {0:[], 1:[], 2:[], 3:[]}
                for v in sd.vehicles:
                    if v["state"] == "moving":
                        progress_by_dir[v["dir"]].append(v["progress"])
                for lst in progress_by_dir.values():
                    lst.sort()

                MIN_GAP = 0.07   # minimum progress gap between vehicles (~safe following distance)

                for v in sd.vehicles:
                    direction = v["dir"]
                    is_ns  = direction in (0, 2)
                    light  = light_ns if is_ns else light_ew
                    queue  = queues[direction]
                    pos_in_q = queue.index(v) if v in queue else 99

                    if v["state"] == "queued":
                        if light == "green" and pos_in_q == 0:
                            # Only release if road ahead is clear
                            ahead = progress_by_dir[direction]
                            if not ahead or min(ahead) > MIN_GAP:
                                v["state"] = "moving"
                                v["wait"]  = now - v["arrive"]
                                v["progress"] = 0.0
                        elif light == "green" and pos_in_q > 0:
                            # Wait for the vehicle directly ahead to create enough gap
                            leader = queue[pos_in_q - 1]
                            leader_prog = leader.get("progress", 0.0)
                            if leader["state"] == "moving" and leader_prog > MIN_GAP * (pos_in_q + 1):
                                v["state"] = "moving"
                                v["wait"]  = now - v["arrive"]
                                v["progress"] = 0.0
                        else:
                            v["wait"] = now - v["arrive"]

                    elif v["state"] == "moving":
                        base_speed = 0.004
                        # Find the nearest vehicle ahead in same direction
                        ahead_progs = [p for p in progress_by_dir[direction]
                                       if p > v["progress"] + 0.001]
                        if ahead_progs:
                            gap = min(ahead_progs) - v["progress"]
                            if gap < MIN_GAP:
                                # Slow down proportionally to close gap — stop if too close
                                speed = base_speed * max(0.0, (gap / MIN_GAP) - 0.1)
                            else:
                                speed = base_speed
                        else:
                            speed = base_speed

                        v["progress"] += speed
                        # Update our entry in progress_by_dir so followers see new position
                        try:
                            old_idx = progress_by_dir[direction].index(v["progress"] - speed)
                            progress_by_dir[direction][old_idx] = v["progress"]
                        except ValueError:
                            pass
                        progress_by_dir[direction].sort()

                        if v["progress"] >= 1.0:
                            v["state"]  = "done"
                            v["depart"] = now
                            sd.wait_times.append(v["wait"])
                            sd.throughput_log.append((now, len(sd.completed)))
                            remove.append(v)

                for v in remove:
                    sd.vehicles.remove(v)
                    sd.completed.append(v)

    env.process(traffic_light_controller(env))
    env.process(vehicle_mover(env))
    for d in range(4):
        env.process(vehicle_generator(env, d))

    while sd.running:
        env.step()
        time.sleep(0.01 / sd.speed_factor)

# ─────────────────────────────────────────────
#  CHART GENERATION  (Matplotlib → Pygame surface)
# ─────────────────────────────────────────────
_chart_cache: pygame.Surface | None = None
_chart_last_len = 0

def build_chart(sd: SimData, w: int, h: int) -> pygame.Surface:
    global _chart_cache, _chart_last_len
    with sd.lock:
        waits = list(sd.wait_times)
        completed = len(sd.completed)

    if completed == _chart_last_len and _chart_cache is not None:
        return _chart_cache
    _chart_last_len = completed

    fig, axes = plt.subplots(1, 2, figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#161820")

    # — Wait time histogram
    ax1 = axes[0]
    ax1.set_facecolor("#1e2130")
    if waits:
        ax1.hist(waits, bins=max(8, min(20, len(waits)//3 + 1)),
                 color="#50a0ff", edgecolor="#0a0e1a", linewidth=0.5)
    ax1.set_title("Vehicle Wait Times", color="#dce1f0", fontsize=9, pad=6)
    ax1.set_xlabel("Wait (s)", color="#787e96", fontsize=7)
    ax1.set_ylabel("Count", color="#787e96", fontsize=7)
    ax1.tick_params(colors="#787e96", labelsize=6)
    for sp in ax1.spines.values(): sp.set_color("#32364a")

    # — Throughput over time
    ax2 = axes[1]
    ax2.set_facecolor("#1e2130")
    with sd.lock:
        log = list(sd.throughput_log)
    if log:
        xs = [l[0] for l in log]
        ys = list(range(1, len(log)+1))
        ax2.plot(xs, ys, color="#50ffa0", linewidth=1.2)
        ax2.fill_between(xs, ys, alpha=0.15, color="#50ffa0")
    ax2.set_title("Cumulative Throughput", color="#dce1f0", fontsize=9, pad=6)
    ax2.set_xlabel("Sim Time (s)", color="#787e96", fontsize=7)
    ax2.set_ylabel("Vehicles", color="#787e96", fontsize=7)
    ax2.tick_params(colors="#787e96", labelsize=6)
    for sp in ax2.spines.values(): sp.set_color("#32364a")

    fig.tight_layout(pad=1.5)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    surf = pygame.image.frombuffer(buf, canvas.get_width_height(), "RGBA")
    plt.close(fig)
    _chart_cache = surf.copy()
    return _chart_cache

# ─────────────────────────────────────────────
#  PYGAME RENDERER
# ─────────────────────────────────────────────
def lerp(a, b, t): return a + (b - a) * t

def draw_rounded_rect(surf, color, rect, r=8, alpha=None):
    if alpha is not None:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)

def draw_intersection(surf: pygame.Surface, sd: SimData, cx: int, cy: int):
    """Draw the road, markings, lights, and vehicles."""
    rw = ROAD_W
    lw = LANE_W
    half = rw // 2

    # ── Sidewalk / grass border ──
    pygame.draw.rect(surf, C["sidewalk"],
                     (cx - half - 30, cy - half - 30, rw + 60, rw + 60),
                     border_radius=6)

    # ── Road surfaces ──
    # Vertical road (N-S)
    pygame.draw.rect(surf, C["road"], (cx - half, 0, rw, HEIGHT))
    # Horizontal road (E-W)
    pygame.draw.rect(surf, C["road"], (0, cy - half, WIDTH, rw))  # full width

    # ── Center box ──
    pygame.draw.rect(surf, (25, 28, 36), (cx - half, cy - half, rw, rw))

    # ── Lane markings ──
    dash_len, gap = 22, 14
    # Vertical dashes (outside intersection)
    for y in range(0, cy - half, dash_len + gap):
        pygame.draw.rect(surf, C["road_line"], (cx - 3, y, 5, dash_len))
        pygame.draw.rect(surf, C["road_line"], (cx + 3 + lw - 5, y, 5, dash_len))
    for y in range(cy + half, HEIGHT, dash_len + gap):
        pygame.draw.rect(surf, C["road_line"], (cx - 3, y, 5, dash_len))
        pygame.draw.rect(surf, C["road_line"], (cx + 3 + lw - 5, y, 5, dash_len))
    # Horizontal dashes
    for x in range(0, cx - half, dash_len + gap):
        pygame.draw.rect(surf, C["road_line"], (x, cy - 3, dash_len, 5))
        pygame.draw.rect(surf, C["road_line"], (x, cy + lw - 3, dash_len, 5))
    for x in range(cx + half, WIDTH // 2 + 200, dash_len + gap):
        pygame.draw.rect(surf, C["road_line"], (x, cy - 3, dash_len, 5))
        pygame.draw.rect(surf, C["road_line"], (x, cy + lw - 3, dash_len, 5))

    # ── Stop lines ──
    stop_gap = 8
    pygame.draw.rect(surf, (200, 200, 200),
                     (cx - half, cy - half - stop_gap - 4, rw, 4))
    pygame.draw.rect(surf, (200, 200, 200),
                     (cx - half, cy + half + stop_gap, rw, 4))
    pygame.draw.rect(surf, (200, 200, 200),
                     (cx - half - stop_gap - 4, cy - half, 4, rw))
    pygame.draw.rect(surf, (200, 200, 200),
                     (cx + half + stop_gap, cy - half, 4, rw))

    # ── Crosswalk stripes ──
    stripe_h, stripe_gap, n_stripes = 6, 4, 5
    total_stripe = n_stripes * (stripe_h + stripe_gap)
    for i in range(n_stripes):
        yy = cy - half - stop_gap - total_stripe - 4 + i * (stripe_h + stripe_gap)
        pygame.draw.rect(surf, (200, 200, 210, 120),
                         (cx - half, yy, rw, stripe_h))
        pygame.draw.rect(surf, (200, 200, 210, 120),
                         (cx - half, cy + half + stop_gap + 4 + i * (stripe_h + stripe_gap),
                          rw, stripe_h))
    for i in range(n_stripes):
        xx = cx - half - stop_gap - total_stripe - 4 + i * (stripe_h + stripe_gap)
        pygame.draw.rect(surf, (200, 200, 210, 120),
                         (xx, cy - half, stripe_h, rw))
        pygame.draw.rect(surf, (200, 200, 210, 120),
                         (cx + half + stop_gap + 4 + i * (stripe_h + stripe_gap),
                          cy - half, stripe_h, rw))

    # ── Traffic lights ──
    with sd.lock:
        l_ns = sd.light_ns
        l_ew = sd.light_ew
        timer = sd.light_timer

    def draw_light_pole(x, y, state, horizontal=False):
        # Pole
        pygame.draw.rect(surf, (60, 65, 80), (x - 3, y - 3, 6, 30), border_radius=3)
        # Housing
        hw, hh = 18, 50
        hx, hy = x - hw//2, y - hh - 3
        draw_rounded_rect(surf, (25, 28, 36), (hx, hy, hw, hh), r=5)
        pygame.draw.rect(surf, (50, 55, 70), (hx, hy, hw, hh), 1, border_radius=5)
        # Lights
        light_states = {
            "red":    (C["red_light"],    (40, 40, 40),   (40, 40, 40)),
            "yellow": ((40, 40, 40),      C["yellow_light"], (40, 40, 40)),
            "green":  ((40, 40, 40),      (40, 40, 40),   C["green_light"]),
        }
        cols = light_states.get(state, ((40,40,40),(40,40,40),(40,40,40)))
        for i, col in enumerate(cols):
            lx = hx + hw//2
            ly = hy + 10 + i * 14
            # glow
            if col != (40, 40, 40):
                glow = pygame.Surface((24, 24), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*col, 60), (12, 12), 12)
                surf.blit(glow, (lx - 12, ly - 12))
            pygame.draw.circle(surf, col, (lx, ly), 6)

    # Place lights at corners of intersection
    draw_light_pole(cx - half - 22, cy - half - 5, l_ns)
    draw_light_pole(cx + half + 10, cy + half - 45, l_ns)
    draw_light_pole(cx - half - 22, cy + half - 45, l_ew)
    draw_light_pole(cx + half + 10, cy - half - 5,  l_ew)

    # ── Timer badge ──
    font_sm = pygame.font.SysFont("monospace", 14, bold=True)
    timer_surf = font_sm.render(f"{int(timer):02d}s", True, C["text"])
    surf.blit(timer_surf, (cx - 14, cy - 12))

    # ── Vehicles ──
    with sd.lock:
        vehicles = list(sd.vehicles)

    for v in vehicles:
        _draw_vehicle(surf, v, cx, cy, half, lw)

def _draw_vehicle(surf, v, cx, cy, half, lw):
    """Draw a single vehicle moving through the scene."""
    p         = v["progress"]   # 0 → 1
    d         = v["dir"]        # 0=N,1=E,2=S,3=W
    col       = v["color"]
    state     = v["state"]
    lane_side = v["lane"]       # -1 or +1
    lane_off  = lane_side * (lw // 2 + 4)
    slot      = v.get("queue_slot", 0)

    car_w, car_h = 18, 30      # relative to travel direction
    stop_margin  = 18
    car_spacing  = 36          # pixels between queued cars

    is_queued = (state == "queued")

    if d == 0:   # North → coming from top
        x     = cx + lane_off + lw // 2
        y     = int(lerp(-30, HEIGHT + 30, p))
        angle = 0
        if is_queued:
            y = cy - half - stop_margin - slot * car_spacing
    elif d == 2:  # South → coming from bottom
        x     = cx - lane_off + lw // 2
        y     = int(lerp(HEIGHT + 30, -30, p))
        angle = 180
        if is_queued:
            y = cy + half + stop_margin + slot * car_spacing
    elif d == 1:  # East → coming from right
        y     = cy + lane_off + lw // 2
        x     = int(lerp(WIDTH + 30, -30, p))
        angle = 90
        if is_queued:
            x = cx + half + stop_margin + slot * car_spacing
    else:         # West → coming from left
        y     = cy - lane_off + lw // 2
        x     = int(lerp(-30, WIDTH + 30, p))
        angle = 270
        if is_queued:
            x = cx - half - stop_margin - slot * car_spacing

    # ── Car body surface ──
    body = pygame.Surface((car_w, car_h), pygame.SRCALPHA)

    # Main body
    pygame.draw.rect(body, col, (0, 4, car_w, car_h - 8), border_radius=4)

    # Roof
    roof_col = tuple(min(255, c + 30) for c in col)
    pygame.draw.rect(body, roof_col, (3, 0, car_w - 6, car_h - 14), border_radius=3)

    # Headlights (front = bottom of car surface before rotation)
    hl_col = (255, 240, 180)
    pygame.draw.circle(body, hl_col, (4, car_h - 5), 3)
    pygame.draw.circle(body, hl_col, (car_w - 4, car_h - 5), 3)

    # Brake lights — bright red at rear when queued/stopped
    if is_queued:
        brake_col = (255, 50, 50)
        pygame.draw.circle(body, brake_col, (3, 5), 3)
        pygame.draw.circle(body, brake_col, (car_w - 3, 5), 3)
        # Glow behind brake lights
        glow = pygame.Surface((car_w, 10), pygame.SRCALPHA)
        pygame.draw.rect(glow, (255, 50, 50, 60), (0, 0, car_w, 10))
        body.blit(glow, (0, 0))

    # Windows
    win_col = (120, 180, 230, 180)
    pygame.draw.rect(body, win_col, (4, 2, car_w - 8, car_h - 18), border_radius=2)

    rotated = pygame.transform.rotate(body, -angle)
    rr      = rotated.get_rect(center=(x, y))
    surf.blit(rotated, rr.topleft)

    # ── Wait time badge above queued vehicles ──
    if is_queued and v["wait"] > 1.0:
        wait_s  = int(v["wait"])
        fnt     = pygame.font.SysFont("monospace", 9, bold=True)
        badge_w = 28
        badge_h = 13
        bx      = x - badge_w // 2
        # Position badge toward the rear of car (depends on direction)
        if d == 0:   by = y + 20
        elif d == 2: by = y - 32
        elif d == 1: by = y - 10
        else:        by = y - 10

        # Badge background
        badge = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
        pygame.draw.rect(badge, (200, 40, 40, 210), (0, 0, badge_w, badge_h), border_radius=4)
        txt = fnt.render(f"{wait_s}s", True, (255, 255, 255))
        badge.blit(txt, (badge_w // 2 - txt.get_width() // 2,
                         badge_h // 2 - txt.get_height() // 2))
        surf.blit(badge, (bx, by))


def draw_panel(surf, sd, font, font_sm, font_xs, panel_x, panel_y, panel_w, panel_h):
    """Right-side stats panel."""
    draw_rounded_rect(surf, C["panel"], (panel_x, panel_y, panel_w, panel_h), r=10)
    pygame.draw.rect(surf, C["border"], (panel_x, panel_y, panel_w, panel_h), 1, border_radius=10)

    y = panel_y + 16
    # Title
    title = font.render("TRAFFIC SIM", True, C["accent"])
    surf.blit(title, (panel_x + panel_w // 2 - title.get_width() // 2, y))
    y += 28
    sub = font_xs.render("CS 324 • BatStateU CICS", True, C["text_dim"])
    surf.blit(sub, (panel_x + panel_w // 2 - sub.get_width() // 2, y))
    y += 22

    # Divider
    pygame.draw.line(surf, C["border"], (panel_x + 12, y), (panel_x + panel_w - 12, y))
    y += 12

    # Scenario
    sc_label = font_xs.render("SCENARIO", True, C["text_dim"])
    surf.blit(sc_label, (panel_x + 14, y))
    y += 16
    sc_val = font_sm.render(sd.scenario, True, C["accent2"])
    surf.blit(sc_val, (panel_x + 14, y))
    y += 26

    # Light states
    with sd.lock:
        l_ns    = sd.light_ns
        l_ew    = sd.light_ew
        timer   = sd.light_timer
        sim_t   = sd.sim_time
        total_v = sd.total_vehicles
        waiting = sum(1 for v in sd.vehicles if v["state"] == "queued")
        moving  = sum(1 for v in sd.vehicles if v["state"] == "moving")
        done    = len(sd.completed)
        waits   = list(sd.wait_times)

    pygame.draw.line(surf, C["border"], (panel_x + 12, y), (panel_x + panel_w - 12, y))
    y += 12

    # Light indicator boxes
    def light_box(lbl, state, bx, by):
        col_map = {"green": C["green_light"], "yellow": C["yellow_light"], "red": C["red_light"]}
        col = col_map.get(state, C["text_dim"])
        draw_rounded_rect(surf, (30, 34, 48), (bx, by, 80, 44), r=6)
        pygame.draw.rect(surf, col, (bx, by, 80, 44), 2, border_radius=6)
        l_surf = font_xs.render(lbl, True, C["text_dim"])
        surf.blit(l_surf, (bx + 40 - l_surf.get_width() // 2, by + 4))
        dot = pygame.Surface((14, 14), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*col, 220), (7, 7), 7)
        surf.blit(dot, (bx + 33, by + 22))
        s_surf = font_xs.render(state.upper(), True, col)
        surf.blit(s_surf, (bx + 40 - s_surf.get_width() // 2, by + 26))

    light_box("N–S", l_ns, panel_x + 10, y)
    light_box("E–W", l_ew, panel_x + 100, y)
    y += 56

    # Timer
    t_lbl = font_xs.render("PHASE TIMER", True, C["text_dim"])
    surf.blit(t_lbl, (panel_x + 14, y))
    t_val = font_sm.render(f"{int(timer):02d} s", True, C["text"])
    surf.blit(t_val, (panel_x + panel_w - 14 - t_val.get_width(), y))
    y += 20

    pygame.draw.line(surf, C["border"], (panel_x + 12, y), (panel_x + panel_w - 12, y))
    y += 12

    # Stats
    def stat_row(label, value, vy, col=None):
        lbl = font_xs.render(label, True, C["text_dim"])
        surf.blit(lbl, (panel_x + 14, vy))
        val = font_xs.render(str(value), True, col or C["text"])
        surf.blit(val, (panel_x + panel_w - 14 - val.get_width(), vy))
        return vy + 18

    y = stat_row("Sim Time",      f"{sim_t:.1f}s",         y)
    y = stat_row("Total Spawned", total_v,                  y)
    y = stat_row("Passed Through",done,                     y, C["green_light"])
    y = stat_row("Waiting",       waiting,                  y, C["red_light"])
    y = stat_row("Moving",        moving,                   y, C["yellow_light"])

    pygame.draw.line(surf, C["border"], (panel_x + 12, y), (panel_x + panel_w - 12, y))
    y += 12

    avg_wait = sum(waits) / len(waits) if waits else 0
    max_wait = max(waits) if waits else 0
    y = stat_row("Avg Wait",      f"{avg_wait:.1f}s",       y)
    y = stat_row("Max Wait",      f"{max_wait:.1f}s",       y)

    # Speed
    pygame.draw.line(surf, C["border"], (panel_x + 12, y + 4), (panel_x + panel_w - 12, y + 4))
    y += 16
    sp_lbl = font_xs.render(f"SIM SPEED  ×{sd.speed_factor:.1f}", True, C["accent"])
    surf.blit(sp_lbl, (panel_x + panel_w // 2 - sp_lbl.get_width() // 2, y))
    y += 20

    # Controls hint
    pygame.draw.line(surf, C["border"], (panel_x + 12, y), (panel_x + panel_w - 12, y))
    y += 10
    hints = [
        ("[1] Normal  [2] Rush  [3] Low", C["text_dim"]),
        ("[↑/↓] Speed  [Q] Quit",         C["text_dim"]),
        ("[C] Charts  [R] Reset",          C["text_dim"]),
    ]
    for hint, hcol in hints:
        h = font_xs.render(hint, True, hcol)
        surf.blit(h, (panel_x + panel_w // 2 - h.get_width() // 2, y))
        y += 16

    return y


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CS 324 — Traffic Light Simulation | BatStateU CICS")
    clock = pygame.time.Clock()

    try:
        font    = pygame.font.SysFont("monospace", 16, bold=True)
        font_sm = pygame.font.SysFont("monospace", 13, bold=True)
        font_xs = pygame.font.SysFont("monospace", 11)
    except:
        font = font_sm = font_xs = pygame.font.Font(None, 14)

    # Start SimPy thread
    sim_thread = threading.Thread(target=run_simulation, args=(SD,), daemon=True)
    sim_thread.start()

    # Layout
    panel_w  = 200
    panel_x  = WIDTH - panel_w - 10
    panel_y  = 10
    panel_h  = HEIGHT - 20
    view_w   = panel_x - 10           # road canvas width
    cx       = view_w // 2            # intersection center x
    cy       = HEIGHT // 2            # intersection center y

    show_charts = False
    chart_surf  = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_1:
                    with SD.lock: SD.scenario = "Normal Traffic"
                elif event.key == pygame.K_2:
                    with SD.lock: SD.scenario = "Rush Hour"
                elif event.key == pygame.K_3:
                    with SD.lock: SD.scenario = "Low Traffic"
                elif event.key == pygame.K_UP:
                    SD.speed_factor = min(10.0, SD.speed_factor + 0.5)
                elif event.key == pygame.K_DOWN:
                    SD.speed_factor = max(0.5, SD.speed_factor - 0.5)
                elif event.key == pygame.K_c:
                    show_charts = not show_charts
                    chart_surf  = None
                elif event.key == pygame.K_r:
                    with SD.lock:
                        SD.vehicles.clear()
                        SD.completed.clear()
                        SD.wait_times.clear()
                        SD.throughput_log.clear()
                        SD.total_vehicles = 0

        # Background
        screen.fill(C["bg"])

        # Road background behind panel
        pygame.draw.rect(screen, C["bg"], (0, 0, view_w, HEIGHT))

        if show_charts:
            # Render charts
            if chart_surf is None or len(SD.completed) != _chart_last_len:
                chart_surf = build_chart(SD, view_w, HEIGHT)
            screen.blit(chart_surf, (0, 0))
            lbl = font_sm.render("Press [C] to return to simulation", True, C["text_dim"])
            screen.blit(lbl, (view_w // 2 - lbl.get_width() // 2, HEIGHT - 30))
        else:
            # Road + vehicles
            draw_intersection(screen, SD, cx, cy)

        # Stats panel
        draw_panel(screen, SD, font, font_sm, font_xs,
                   panel_x, panel_y, panel_w, panel_h)

        # FPS
        fps_surf = font_xs.render(f"FPS {int(clock.get_fps())}", True, C["text_dim"])
        screen.blit(fps_surf, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    SD.running = False
    pygame.quit()

    # Export results CSV
    if SD.completed:
        df = pd.DataFrame(SD.completed)
        df.to_csv("simulation_results.csv", index=False)
        print(f"\n✅ Results saved to simulation_results.csv ({len(SD.completed)} vehicles)")

    print("Simulation ended.")
    sys.exit(0)


if __name__ == "__main__":
    main()