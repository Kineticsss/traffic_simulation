"""
CS 324 — Modeling and Simulation
Traffic Light System Simulation | Batangas State University — CICS

KEY DESIGN DECISIONS
====================
1. Vehicles spawn off-screen and travel to the stop line, joining the queue.
2. Queue is pixel-space: each car sits exactly SLOT px behind the one in front.
3. On green, slot-0 releases. Every other car follows the one ahead — no gaps.
4. Moving cars use path-distance tracking; they NEVER pass each other.
5. No U-turns. Straight → opposite outbound lane. Turn → adjacent outbound arm.
6. One NS or EW phase green at a time. All-red clearance gap between phases.
7. Lane count toggle: 2 / 3 / 4 / 6 lanes per direction. Road rebuilds on change.
"""

import simpy, pygame, random, math, sys, threading, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd

# ═══════════════════════════════════════════════════════
#  WINDOW CONFIGURATION
# ═══════════════════════════════════════════════════════
WIDTH, HEIGHT = 1280, 780
PANEL_W       = 230
VIEW_W        = WIDTH - PANEL_W - 8
CX            = VIEW_W // 2
CY            = HEIGHT  // 2

# ═══════════════════════════════════════════════════════
#  ROAD GEOMETRY
# ═══════════════════════════════════════════════════════
LANE_OPTIONS = [2, 3, 4, 6]
N_LANES      = 3          # default lane count (toggled at runtime)

# Physical lane width shrinks as lane count grows so road fits on screen
def lane_w_for(n):
    return {2: 36, 3: 28, 4: 24, 6: 18}[n]

DIV_W    = 6
DIV_HALF = DIV_W // 2

def compute_geometry(n):
    lw   = lane_w_for(n)
    offs = [DIV_HALF + lw // 2 + i * lw for i in range(n)]
    hr   = DIV_HALF + n * lw
    return lw, offs, hr

LANE_W, LANE_OFFS, HR = compute_geometry(N_LANES)

# ═══════════════════════════════════════════════════════
#  VEHICLE / QUEUE
# ═══════════════════════════════════════════════════════
CAR_LEN   = 20
CAR_W     = 12
CAR_GAP   = 5
SLOT      = CAR_LEN + CAR_GAP     # px per queue slot
STOP_DIST = 12                     # px from box edge to stop line

# Speed in px per sim-frame (at 60 fps, 60 px/s feels natural)
CAR_SPEED = 1.0

# ═══════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════
SIM_FPS = 60.0
FRAME_T = 1.0 / SIM_FPS   # sim-seconds per mover tick

# ═══════════════════════════════════════════════════════
#  TRAFFIC CONGESTION SCENARIOS
# ═══════════════════════════════════════════════════════
SCENARIOS = {
    "Normal": {"green": 30, "yellow": 4, "clear": 3, "red": 30, "arrival": 3.5},
    "Rush":   {"green": 40, "yellow": 4, "clear": 3, "red": 40, "arrival": 1.5},
    "Low":    {"green": 20, "yellow": 3, "clear": 3, "red": 20, "arrival": 8.0},
}

DIR_NAMES  = ["North","East","South","West"]
TURNS      = ["right","straight","left"]
TURN_PROBS = [0.25,   0.50,     0.25]

# ═══════════════════════════════════════════════════════
#  COLOURS
# ═══════════════════════════════════════════════════════
C = {
    "bg":           (12,  14,  20),
    "grass":        (22,  38,  22),
    "road":         (38,  40,  50),
    "road_box":     (30,  32,  42),
    "divider":      (210, 180,  30),
    "lane_dash":    (85,  89,  70),
    "stop_line":    (222, 222, 222),
    "kerb":         (54,  57,  68),
    "panel":        (15,  18,  28),
    "border":       (42,  47,  68),
    "text":         (214, 221, 238),
    "text_dim":     (90, 100, 128),
    "accent":       (60,  140, 255),
    "accent2":      (255, 178,  36),
    "green_light":  (36,  205,  75),
    "yellow_light": (255, 198,  36),
    "red_light":    (255,  46,  46),
    "car_colors": [
        (80,162,255),(255,102,65),(100,208,125),(255,198,60),
        (168,100,255),(255,145,80),(60,208,208),(192,192,202),
        (255,120,162),(120,255,192),(255,170,95),(95,195,255),
    ],
}

# ═══════════════════════════════════════════════════════
#  THEMES
# ═══════════════════════════════════════════════════════
THEMES = {

    "classic": {
        "bg": (12,14,20),
        "grass": (24,40,24),
        "road": (36,38,47),
        "road_box": (28,30,39),
        "divider": (200,175,30),
        "lane_dash": (85,89,70),
        "kerb": (50,53,63),
        "stop_line": (220,220,220),
        "panel": (15,18,28),
        "border": (42,47,68),
        "text": (214,221,238),
        "text_dim": (90,100,128),
        "accent": (60,140,255),
        "accent2": (255,178,36),
        "green_light": (36,205,75),
        "yellow_light": (255,198,36),
        "red_light": (255,46,46),

        "car_colors": [
            (80,162,255),
            (255,102,65),
            (100,208,125),
            (255,198,60),
        ]
    },

    "cyber": {
        "bg": (5,8,20),
        "grass": (10,10,18),
        "road": (20,20,35),
        "road_box": (12,14,28),
        "divider": (0,255,255),
        "lane_dash": (120,120,255),
        "kerb": (70,70,120),
        "stop_line": (240,240,255),
        "panel": (10,12,25),
        "border": (0,180,255),
        "text": (220,240,255),
        "text_dim": (120,160,220),
        "accent": (0,255,255),
        "accent2": (255,0,180),
        "green_light": (0,255,120),
        "yellow_light": (255,220,0),
        "red_light": (255,50,80),

        "car_colors": [
            (0,255,255),
            (255,0,180),
            (120,255,0),
            (255,140,0),
        ]
    }
}

CURRENT_THEME = "classic"
C = THEMES[CURRENT_THEME]

# ═══════════════════════════════════════════════════════
#  COORDINATE HELPERS
#
#  from_dir meaning:
#    0 = coming FROM North  → car travels South → IB on east side  (CX + off)
#    1 = coming FROM East   → car travels West  → IB on south side (CY + off)
#    2 = coming FROM South  → car travels North → IB on west side  (CX - off)
#    3 = coming FROM West   → car travels East  → IB on north side (CY - off)
#
#  Exit direction (no U-turn; delta never = 2 relative to from_dir):
#    right    = (from_dir + 3) % 4
#    straight = (from_dir + 2) % 4   → opposite arm outbound
#    left     = (from_dir + 1) % 4
#
#  Outbound coord for exit_dir:
#    exit 0 (north) → OB on west side  (CX - off)
#    exit 2 (south) → OB on east side  (CX + off)
#    exit 1 (east)  → OB on north side (CY - off)
#    exit 3 (west)  → OB on south side (CY + off)
# ═══════════════════════════════════════════════════════

def ib_coord(from_dir, lane):
    lane = max(0, min(lane, len(LANE_OFFS)-1))   # clamp — prevents IndexError on lane toggle
    off = LANE_OFFS[lane]
    if from_dir == 0: return CX + off
    if from_dir == 2: return CX - off
    if from_dir == 1: return CY + off
    return             CY - off

def ob_coord(exit_dir, lane):
    lane = max(0, min(lane, len(LANE_OFFS)-1))   # clamp
    off = LANE_OFFS[lane]
    if exit_dir == 0: return CX - off
    if exit_dir == 2: return CX + off
    if exit_dir == 1: return CY - off
    return             CY + off

def stop_px(from_dir):
    d = HR + STOP_DIST
    if from_dir == 0: return CY - d
    if from_dir == 2: return CY + d
    if from_dir == 1: return CX + d
    return             CX - d

# ═══════════════════════════════════════════════════════
#  PATH BUILDER
#  Waypoints: [spawn, stop_line, ...box_curve..., depart]
#  spawn    = off-screen entry point
#  stop_line = where car stops at red (path[1])
#  depart   = off-screen exit point
# ═══════════════════════════════════════════════════════
def _bezier(p0, p1, p2, n=16):
    return [
        ((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t**2*p2[0],
         (1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t**2*p2[1])
        for t in (i/n for i in range(n+1))
    ]

def build_path(from_dir, turn, lane):
    ic  = ib_coord(from_dir, lane)
    stp = stop_px(from_dir)

    # Approach: spawn well off-screen → stop line → box edge
    if from_dir == 0:
        spawn   = (ic, -80)
        stop_pt = (ic, stp)
        box_in  = (ic, CY - HR)
    elif from_dir == 2:
        spawn   = (ic, HEIGHT + 80)
        stop_pt = (ic, stp)
        box_in  = (ic, CY + HR)
    elif from_dir == 1:
        spawn   = (VIEW_W + 80, ic)
        stop_pt = (stp, ic)
        box_in  = (CX + HR, ic)
    else:
        spawn   = (-80, ic)
        stop_pt = (stp, ic)
        box_in  = (CX - HR, ic)

    # Exit direction — delta never 0 mod 4 (no U-turn)
    exit_dir = {"right":    (from_dir+3)%4,
                "straight": (from_dir+2)%4,
                "left":     (from_dir+1)%4}[turn]

    # Exit lane:
    #   straight → same lane index on the opposite arm
    #   turn     → innermost lane (0) of the new arm
    ex_lane = min(lane, N_LANES - 1)
    ec = ob_coord(exit_dir, ex_lane)

    # Box exit point and off-screen depart
    if exit_dir == 0:
        box_out = (ec, CY - HR)
        depart  = (ec, -80)
    elif exit_dir == 2:
        box_out = (ec, CY + HR)
        depart  = (ec, HEIGHT + 80)
    elif exit_dir == 1:
        box_out = (CX + HR, ec)
        depart  = (VIEW_W + 80, ec)
    else:
        box_out = (CX - HR, ec)
        depart  = (-80, ec)

    if turn == "straight":
        # Keep fixed axis through box so car never drifts
        if from_dir in (0, 2):
            box_out = (ic, box_out[1])
            depart  = (ic, depart[1])
        else:
            box_out = (box_out[0], ic)
            depart  = (depart[0], ic)
        return [spawn, stop_pt, box_in, box_out, depart]

    # Turn: quadratic Bézier through the intersection corner
    bix, biy = box_in
    bx2, by2 = box_out
    cp = (bix, by2) if from_dir in (0, 2) else (bx2, biy)
    n  = 10 if turn == "right" else 18
    return [spawn, stop_pt] + _bezier(box_in, cp, box_out, n=n) + [depart]

# ═══════════════════════════════════════════════════════
#  PATH UTILITIES
# ═══════════════════════════════════════════════════════
def path_length(path):
    return sum(math.hypot(path[i+1][0]-path[i][0],
                          path[i+1][1]-path[i][1])
               for i in range(len(path)-1))

def path_pos_at_dist(path, dist):
    """Return (x, y, heading_deg) at `dist` pixels along the path.
    Correctly handles every segment without skipping."""
    dist = max(0.0, dist)
    acc  = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        sl = math.hypot(dx, dy)
        if sl < 1e-9:
            continue          # skip zero-length segment
        if dist <= acc + sl:  # target is inside this segment
            t = (dist - acc) / sl
            return (path[i][0] + t*dx,
                    path[i][1] + t*dy,
                    math.degrees(math.atan2(dy, dx)))
        acc += sl
    # Past the end — return final point with heading of last segment
    dx = path[-1][0] - path[-2][0]
    dy = path[-1][1] - path[-2][1]
    return path[-1][0], path[-1][1], math.degrees(math.atan2(dy, dx))

# ═══════════════════════════════════════════════════════
#  QUEUE PIXEL POSITION
#  Slot 0 = immediately behind stop line
#  Slot N = N slots further back
#  Completely independent of path "dist" — purely geometric
# ═══════════════════════════════════════════════════════
def queue_pixel(v):
    d    = v["from_dir"]
    ic   = ib_coord(d, v["lane"])
    stp  = stop_px(d)
    ofs  = (v.get("queue_slot", 0) + 1) * SLOT
    if d == 0: return float(ic), float(stp - ofs), 90.0
    if d == 2: return float(ic), float(stp + ofs), 270.0
    if d == 1: return float(stp + ofs), float(ic), 180.0
    return      float(stp - ofs), float(ic), 0.0

# ═══════════════════════════════════════════════════════
#  SIMULATION STATE
# ═══════════════════════════════════════════════════════
class SimData:
    def __init__(self):
        self.lock            = threading.Lock()
        self.vehicles        : list[dict] = []
        self.completed       : list[dict] = []
        self.lights          = ["green","red","red","red"]  # per direction
        self.timers          = [30,0,0,0]
        self.active_dir      = 0
        self.sim_time        = 0.0
        self.total_vehicles  = 0
        self.scenario        = "Normal"
        self.running         = True
        self.speed_factor    = 1.0
        self.wait_times      : list[float] = []
        self.throughput_log  : list[tuple] = []
        self.queue_log : list[tuple] = []
        self.wait_times      : list[float] = []
        self.throughput_log  : list[tuple] = []
        self.queue_log       : list[tuple] = []
        self.max_queue_seen = 0
        self.reset_time = 0.0

        self.reset_flag      = False

SD = SimData()

# ═══════════════════════════════════════════════════════
#  SIMPY ENGINE
# ═══════════════════════════════════════════════════════
def run_simulation(sd: SimData):
    env = simpy.Environment()

    # ── 4-direction rotating light controller ──
    # Only ONE direction is green at a time.
    # Sequence: dir0 green→yellow → all-red → dir1 green→yellow → all-red → ...
    def light_ctrl(env):
        cur = 0
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            G, Y, CLR = cfg["green"], cfg["yellow"], cfg["clear"]

            # Green
            with sd.lock:
                sd.lights     = ["red"]*4
                sd.lights[cur]= "green"
                sd.timers     = [0]*4
                sd.timers[cur]= G
                sd.active_dir = cur
            for t in range(G):
                if not sd.running: return
                yield env.timeout(1.0)
                with sd.lock:
                    sd.timers[cur] = max(0, G-t-1)
                    sd.sim_time    = env.now

            # Yellow
            with sd.lock:
                sd.lights[cur] = "yellow"
                sd.timers[cur] = Y
            for t in range(Y):
                if not sd.running: return
                yield env.timeout(1.0)
                with sd.lock:
                    sd.timers[cur] = max(0, Y-t-1)
                    sd.sim_time    = env.now

            # All-red clearance
            with sd.lock:
                sd.lights = ["red"]*4
                sd.timers = [CLR]*4
            for t in range(CLR):
                if not sd.running: return
                yield env.timeout(1.0)
                with sd.lock:
                    for i in range(4): sd.timers[i] = max(0, CLR-t-1)
                    sd.sim_time = env.now

            cur = (cur + 1) % 4

    # ── Spawner ──
    def gen_vehicles(env, from_dir):
        vid = 0
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            yield env.timeout(random.expovariate(1.0 / cfg["arrival"]))

            # Skip spawning while reset is occurring
            if sd.reset_flag:
                yield env.timeout(0.1)
                continue
            vid += 1
            turn = random.choices(TURNS, TURN_PROBS)[0]
            # Lane selection based on movement
            # 0 = innermost/leftmost
            # last = outermost/rightmost

            # Corrected lane usage for YOUR geometry
            # lane 0 = outermost lane
            # lane N-1 = innermost lane

            if turn == "left":
                # Left turns use inner lane
                lane = N_LANES - 1

            elif turn == "right":
                # Right turns use outer lane
                lane = 0

            else:
                # Straight uses middle lanes
                if N_LANES <= 2:
                    lane = random.randint(0, N_LANES - 1)
                else:
                    mids = list(range(1, N_LANES - 1))
                    lane = random.choice(mids)
            path = build_path(from_dir, turn, lane)
            plen = path_length(path)
            stop_d = math.hypot(path[1][0]-path[0][0], path[1][1]-path[0][1])
            with sd.lock:
                sd.total_vehicles += 1
                sd.vehicles.append({
                    "id":        f"{DIR_NAMES[from_dir][0]}{vid}",
                    "from_dir":  from_dir,
                    "lane":      lane,
                    "turn":      turn,
                    "path":      path,
                    "plen":      plen,
                    "stop_d":    stop_d,
                    "dist":      0.0,
                    "state":     "approach",
                    "arrive":    env.now,
                    "depart":    None,
                    "wait":      0.0,
                    "queue_slot":0,
                    "color":     random.choice(C["car_colors"]),
                })

    # ── Mover ──
    def mover(env):
        while sd.running:
            yield env.timeout(FRAME_T)
            with sd.lock:
                if sd.reset_flag:
                    sd.vehicles.clear()
                    sd.completed.clear()

                    sd.wait_times.clear()
                    sd.throughput_log.clear()
                    sd.queue_log.clear()

                    sd.sim_time = 0.0

                    sd.total_vehicles = 0
                    sd.reset_flag = False

                    sd.reset_time = env.now

                lights = list(sd.lights)
                now    = env.now

                queued_count = sum(
                    1 for v in sd.vehicles
                    if v["state"] == "queued"
                )

                sd.max_queue_seen = max(
                sd.max_queue_seen,
                queued_count
                )

                sd.queue_log.append((now - sd.reset_time, queued_count))

                # Per-(dir,lane) queue lists with slots
                ql: dict[tuple,list] = {}
                for v in sd.vehicles:
                    if v["state"] == "queued":
                        key = (v["from_dir"], v["lane"])
                        ql.setdefault(key,[]).append(v)
                for q in ql.values():
                    q.sort(key=lambda v: v["arrive"])
                    for slot,v in enumerate(q): v["queue_slot"] = slot

                # Moving vehicles tracked per (direction,lane)
                mov_lane = {}

                for v in sd.vehicles:
                    if v["state"] == "moving":
                        key = (v["from_dir"], v["lane"])
                        mov_lane.setdefault(key, []).append(v["dist"])

                for lst in mov_lane.values():
                    lst.sort()

                # Box occupancy: which from_dirs have a car inside the box?
                # A car is "in the box" when its dist is between stop_d and
                # stop_d + HR*4 (generous diagonal crossing distance).
                BOX_CROSS = HR * 4
                in_box: set[int] = set()
                for v in sd.vehicles:
                    if (v["state"] == "moving"
                            and v["stop_d"] <= v["dist"] <= v["stop_d"] + BOX_CROSS):
                        in_box.add(v["from_dir"])

                remove = []
                for v in sd.vehicles:
                    d    = v["from_dir"]
                    lane = v["lane"]
                    key  = (d, lane)
                    grn  = lights[d] == "green"

                    # ── APPROACH ──
                    if v["state"] == "approach":
                        q        = ql.get(key, [])
                        n_q      = len(q)
                        tail_d   = v["stop_d"] - (n_q + 1) * SLOT

                        # Gap-follow moving cars ahead in same direction
                        ahead_mv = [dd for dd in mov_lane.get((d, lane), []) if dd > v["dist"]]
                        if ahead_mv:
                            gap = min(ahead_mv) - v["dist"]
                            safe = CAR_LEN + CAR_GAP
                            spd = CAR_SPEED * max(0.0,(gap/safe)-0.05) if gap < safe else CAR_SPEED
                        else:
                            spd = CAR_SPEED

                        new_d = min(v["dist"] + spd, max(v["dist"], tail_d))
                        v["dist"] = new_d

                        if v["dist"] >= tail_d - 0.5:
                            v["dist"] = tail_d
                            v["state"] = "queued"
                            v["queue_slot"] = n_q
                            ql.setdefault(key,[]).append(v)
                            ql[key].sort(key=lambda v: v["arrive"])
                            for i,qv in enumerate(ql[key]): qv["queue_slot"] = i

                    # ── QUEUED ──
                    elif v["state"] == "queued":
                        v["wait"] = now - v["arrive"]
                        slot = v.get("queue_slot", 0)
                        q    = ql.get(key, [])

                        if grn and slot == 0:

                            # Only check spacing in SAME lane
                            lane_clear = True

                            for ov in sd.vehicles:
                                if (
                                    ov is not v
                                    and ov["state"] == "moving"
                                    and ov["from_dir"] == d
                                    and ov["lane"] == lane
                                ):
                                    if abs(ov["dist"] - v["stop_d"]) < SLOT * 1.2:
                                        lane_clear = False
                                        break

                            if lane_clear:
                                v["state"] = "moving"
                                v["dist"]  = v["stop_d"]

                                mov_lane.get((d, lane), []).append(v["dist"])
                                mov_lane.get((d, lane), []).sort()

                                q.remove(v)

                                for i, rv in enumerate(q):
                                    rv["queue_slot"] = i

                        elif grn and slot > 0:
                            leader = q[slot-1]
                            if (leader["state"] == "moving"
                                    and leader["dist"] >= v["stop_d"] + SLOT):
                                v["state"] = "moving"
                                v["dist"]  = v["stop_d"]
                                mov_lane.get((d, lane), []).append(v["dist"]); mov_lane.get((d, lane), []).sort()
                                in_box.add(d)
                                q.remove(v)
                                for i,rv in enumerate(q): rv["queue_slot"] = i

                    # ── MOVING ──
                    elif v["state"] == "moving":
                        ahead_mv = [dd for dd in mov_lane.get((d, lane), []) if dd > v["dist"] + 0.5]
                        if ahead_mv:
                            gap  = min(ahead_mv) - v["dist"]
                            safe = CAR_LEN + CAR_GAP
                            spd  = CAR_SPEED * max(0.0,(gap/safe)-0.05) if gap < safe else CAR_SPEED
                        else:
                            spd = CAR_SPEED

                        old_d     = v["dist"]
                        v["dist"] = min(v["plen"], old_d + spd)
                        lane_list = mov_lane.get((d, lane), [])
                        if old_d in lane_list:
                            lane_list.remove(old_d)
                        mov_lane.get((d, lane), []).append(v["dist"]); mov_lane.get((d, lane), []).sort()

                        if v["dist"] >= v["plen"] - 0.5:
                            v["state"] = "done"; v["depart"] = now
                            sd.wait_times.append(v["wait"])
                            sd.throughput_log.append((now,len(sd.completed)+1))
                            remove.append(v)

                for v in remove:
                    sd.vehicles.remove(v); sd.completed.append(v)

    env.process(light_ctrl(env))
    env.process(mover(env))
    for d in range(4): env.process(gen_vehicles(env, d))

    while sd.running:
        target = env.now + FRAME_T
        while env.peek() <= target and sd.running: env.step()
        time.sleep(FRAME_T / sd.speed_factor)


# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def rrect(surf, color, rect, r=8, alpha=None):
    if alpha is not None:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0,0,rect[2],rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)

# ═══════════════════════════════════════════════════════
#  STATIC ROAD SURFACE  (rebuilt whenever N_LANES changes)
# ═══════════════════════════════════════════════════════
def build_road_surface():
    surf = pygame.Surface((VIEW_W, HEIGHT))
    cx, cy = CX, CY
    surf.fill(C["grass"])

    # Road base rects
    pygame.draw.rect(surf, C["road"],     (cx-HR, 0,     HR*2,   HEIGHT))
    pygame.draw.rect(surf, C["road"],     (0,     cy-HR, VIEW_W, HR*2))
    pygame.draw.rect(surf, C["road_box"], (cx-HR, cy-HR, HR*2,   HR*2))

    dh = DIV_HALF

    # Yellow centre divider (outside box only)
    pygame.draw.rect(surf, C["divider"], (cx-dh, 0,      DIV_W, cy-HR))
    pygame.draw.rect(surf, C["divider"], (cx-dh, cy+HR,  DIV_W, HEIGHT))
    pygame.draw.rect(surf, C["divider"], (0,      cy-dh, cx-HR, DIV_W))
    pygame.draw.rect(surf, C["divider"], (cx+HR,  cy-dh, VIEW_W, DIV_W))

    dk = C["lane_dash"]
    dl, dg = 18, 12

    # Dashed lane separators between adjacent lanes (both IB and OB sides)
    for i in range(1, N_LANES):
        # N-S road: IB east side, OB west side
        lx_ib = cx + dh + i * LANE_W
        lx_ob = cx - dh - i * LANE_W
        for y0, y1 in [(0, cy-HR), (cy+HR, HEIGHT)]:
            for y in range(y0, y1, dl+dg):
                w = min(dl, y1-y)
                pygame.draw.rect(surf, dk, (lx_ib-1, y, 2, w))
                pygame.draw.rect(surf, dk, (lx_ob-1, y, 2, w))
        # E-W road: IB south side, OB north side
        ly_ib = cy + dh + i * LANE_W
        ly_ob = cy - dh - i * LANE_W
        for x0, x1 in [(0, cx-HR), (cx+HR, VIEW_W)]:
            for x in range(x0, x1, dl+dg):
                w = min(dl, x1-x)
                pygame.draw.rect(surf, dk, (x, ly_ib-1, w, 2))
                pygame.draw.rect(surf, dk, (x, ly_ob-1, w, 2))

    # Kerb edges
    kc = C["kerb"]
    pygame.draw.line(surf, kc, (cx-HR, 0),    (cx-HR, HEIGHT), 2)
    pygame.draw.line(surf, kc, (cx+HR, 0),    (cx+HR, HEIGHT), 2)
    pygame.draw.line(surf, kc, (0, cy-HR),    (VIEW_W, cy-HR), 2)
    pygame.draw.line(surf, kc, (0, cy+HR),    (VIEW_W, cy+HR), 2)

    # Stop lines — span IB half only
    wl, st = C["stop_line"], 3
    ib_span = N_LANES * LANE_W
    soff    = HR + STOP_DIST
    pygame.draw.rect(surf, wl, (cx+dh,      cy-soff-st, ib_span, st))
    pygame.draw.rect(surf, wl, (cx-HR,      cy+soff,    ib_span, st))
    pygame.draw.rect(surf, wl, (cx+soff,    cy+dh,      st, ib_span))
    pygame.draw.rect(surf, wl, (cx-soff-st, cy-HR,      st, ib_span))

    _draw_arrows(surf, cx, cy)
    _draw_crosswalks(surf, cx, cy)
    return surf

def _draw_arrows(surf, cx, cy):
    col, sz, dist = (75, 80, 96), 8, 55
    def arrow(x, y, deg):
        r   = math.radians(deg)
        tip = (x+sz*math.cos(r),        y+sz*math.sin(r))
        l   = (x+sz*.5*math.cos(r+2.3), y+sz*.5*math.sin(r+2.3))
        ri  = (x+sz*.5*math.cos(r-2.3), y+sz*.5*math.sin(r-2.3))
        pygame.draw.polygon(surf, col, [tip, l, ri])
    for lane in range(N_LANES):
        off = LANE_OFFS[lane]
        arrow(cx+off,      cy-HR-dist, 90)
        arrow(cx-off,      cy+HR+dist, 270)
        arrow(cx+HR+dist,  cy+off,     180)
        arrow(cx-HR-dist,  cy-off,     0)

def _draw_crosswalks(surf, cx, cy):
    n, sh, sg = 5, 5, 4
    total, ofs = n*(sh+sg), 5
    for i in range(n):
        s = pygame.Surface((HR*2, sh), pygame.SRCALPHA)
        pygame.draw.rect(s,(218,220,228,75),(0,0,HR*2,sh))
        surf.blit(s,(cx-HR, cy-HR-ofs-total+i*(sh+sg)))
        surf.blit(s,(cx-HR, cy+HR+ofs+i*(sh+sg)))
        s2 = pygame.Surface((sh, HR*2), pygame.SRCALPHA)
        pygame.draw.rect(s2,(218,220,228,75),(0,0,sh,HR*2))
        surf.blit(s2,(cx+HR+ofs+i*(sh+sg),         cy-HR))
        surf.blit(s2,(cx-HR-ofs-total+i*(sh+sg),   cy-HR))

# ═══════════════════════════════════════════════════════
#  TRAFFIC LIGHTS
# ═══════════════════════════════════════════════════════
def draw_lights(surf, sd, fxs):
    cx, cy = CX, CY
    with sd.lock:
        lights = list(sd.lights)   # [dir0, dir1, dir2, dir3]
        timers = list(sd.timers)
        active = sd.active_dir

    def pole(px, py, state):
        pygame.draw.rect(surf,(46,50,65),(px-2,py,4,24),border_radius=2)
        hw,hh=18,50; hx,hy=px-hw//2,py-hh
        rrect(surf,(16,19,29),(hx,hy,hw,hh),r=5)
        pygame.draw.rect(surf,(38,43,60),(hx,hy,hw,hh),1,border_radius=5)
        lm={"red":  [C["red_light"],(26,26,26),(26,26,26)],
            "yellow":[(26,26,26),C["yellow_light"],(26,26,26)],
            "green": [(26,26,26),(26,26,26),C["green_light"]]}
        for i,col in enumerate(lm.get(state,[(26,26,26)]*3)):
            lcy=hy+10+i*14
            if col!=(26,26,26):
                g=pygame.Surface((20,20),pygame.SRCALPHA)
                pygame.draw.circle(g,(*col,55),(10,10),10)
                surf.blit(g,(px-10,lcy-10))
            pygame.draw.circle(surf,col,(px,lcy),6)

    # NW pole = from-North (dir 0), NE pole = from-East (dir 1)
    # SE pole = from-South (dir 2), SW pole = from-West (dir 3)
    pole(cx-HR-22, cy-HR-54, lights[0])   # NW → dir0 (from North)
    pole(cx+HR+8,  cy-HR-54, lights[1])   # NE → dir1 (from East)
    pole(cx+HR+8,  cy+HR+4,  lights[2])   # SE → dir2 (from South)
    pole(cx-HR-22, cy+HR+4,  lights[3])   # SW → dir3 (from West)

    # Active direction timer badge
    tb=pygame.Surface((52,18),pygame.SRCALPHA)
    pygame.draw.rect(tb,(0,0,0,165),(0,0,52,18),border_radius=4)
    surf.blit(tb,(cx-26,cy-9))
    label = f"D{active}:{int(timers[active]):02d}s"
    ts=fxs.render(label,True,C["text"])
    surf.blit(ts,(cx-ts.get_width()//2,cy-8))

# ═══════════════════════════════════════════════════════
#  VEHICLE RENDERER
# ═══════════════════════════════════════════════════════
def draw_vehicles(surf, sd, fxs):
    with sd.lock:
        vehs = list(sd.vehicles)
    for v in vehs:
        if v["state"] in ("queued","approach"): _draw_veh(surf,v,fxs)
    for v in vehs:
        if v["state"] == "moving": _draw_veh(surf,v,fxs)

def _draw_veh(surf, v, fxs):
    state = v["state"]
    if state == "queued":
        x, y, angle = queue_pixel(v)
    else:
        x, y, angle = path_pos_at_dist(v["path"], v["dist"])
    x, y = int(x), int(y)
    if x < -100 or x > VIEW_W+100 or y < -100 or y > HEIGHT+100:
        return
    _render_car(surf, x, y, angle, v["color"], state, v.get("wait",0), fxs)

def _render_car(surf, x, y, angle_deg, col, state, wait, fxs):
    cw, ch = CAR_W, CAR_LEN
    body   = pygame.Surface((cw,ch),pygame.SRCALPHA)
    pygame.draw.rect(body,col,(0,2,cw,ch-4),border_radius=4)
    roof=tuple(min(255,c+45) for c in col)
    pygame.draw.rect(body,roof,(2,4,cw-4,ch-14),border_radius=3)
    pygame.draw.rect(body,(98,160,210,175),(2,ch-14,cw-4,8),border_radius=2)
    pygame.draw.circle(body,(255,248,182),(3,    ch-3),2)
    pygame.draw.circle(body,(255,248,182),(cw-3, ch-3),2)
    if state == "queued":
        pygame.draw.circle(body,(255,34,34),(3,   3),2)
        pygame.draw.circle(body,(255,34,34),(cw-3,3),2)
        g=pygame.Surface((cw,10),pygame.SRCALPHA)
        pygame.draw.rect(g,(255,34,34,45),(0,0,cw,10))
        body.blit(g,(0,0))
    else:
        pygame.draw.circle(body,(138,18,18),(3,   3),2)
        pygame.draw.circle(body,(138,18,18),(cw-3,3),2)
    rotated=pygame.transform.rotate(body,-(angle_deg-90))
    rr=rotated.get_rect(center=(x,y))
    surf.blit(rotated,rr.topleft)
    # Badge: only queued, only after 2s wait
    if state=="queued" and wait>2.0:
        bw,bh=28,13
        badge=pygame.Surface((bw,bh),pygame.SRCALPHA)
        pygame.draw.rect(badge,(185,24,24,210),(0,0,bw,bh),border_radius=3)
        t=fxs.render(f"{int(wait)}s",True,(255,255,255))
        badge.blit(t,(bw//2-t.get_width()//2,bh//2-t.get_height()//2))
        surf.blit(badge,(x-bw//2,y-20))

# ═══════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════
_chart_cache = None
_chart_last_n = -1

def build_chart(sd, w, h):
    global _chart_cache, _chart_last_n

    with sd.lock:
        waits = list(sd.wait_times)
        log   = list(sd.throughput_log)
        qlog  = list(getattr(sd, "queue_log", []))
        done  = len(sd.completed)

    # Cache optimization
    if abs(done - _chart_last_n) < 5 and _chart_cache is not None:
        return _chart_cache

    _chart_last_n = done

    # Create figure
    fig, axes = plt.subplots(
    3,
    1,
    figsize=(w / 100, h / 100),
    dpi=100
    )

    fig.patch.set_facecolor("#0c0e14")

    # Common axis styling
    for ax in axes:
        ax.set_facecolor("#171926")

        ax.tick_params(
            colors="#bfc7d5",
            labelsize=7
        )

        for sp in ax.spines.values():
            sp.set_color("#30364a")

        ax.grid(True, alpha=0.15)

# ─────────────────────────────
# WAIT TIME HISTOGRAM
# ─────────────────────────────
    axes[0].set_title(
        "Vehicle Wait Time Distribution",
        color="white",
        fontsize=10
    )

    axes[0].set_xlabel(
        "Wait Time (s)",
        color="#cfd6e6",
        fontsize=8
    )

    axes[0].set_ylabel(
        "Vehicles",
        color="#cfd6e6",
        fontsize=8
    )

    # DEFAULT VALUES
    avg_wait = 0

    if waits:
        bins = max(6, min(20, len(waits) // 3 + 1))

        axes[0].hist(
            waits,
            bins=bins
        )

        avg_wait = sum(waits) / len(waits)

        # Average line
        axes[0].axvline(
            avg_wait,
            linestyle="--",
            linewidth=2
        )

        # Bigger avg label
        axes[0].text(
            avg_wait,
            axes[0].get_ylim()[1] * 0.88,
            f"AVG: {avg_wait:.1f}s",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.85,
                edgecolor="white",
                boxstyle="round,pad=0.45"
            )
        )

    # ─────────────────────────────
    # THROUGHPUT GRAPH
    # ─────────────────────────────
    axes[1].set_title(
        "Intersection Throughput",
        color="white",
        fontsize=10
    )

    axes[1].set_xlabel(
        "Simulation Time (s)",
        color="#cfd6e6",
        fontsize=8
    )

    axes[1].set_ylabel(
        "Vehicles Passed",
        color="#cfd6e6",
        fontsize=8
    )

    if log:
        xs = [x[0] for x in log]
        ys = list(range(1, len(log) + 1))

        axes[1].plot(xs, ys, linewidth=2)

        axes[1].fill_between(
            xs,
            ys,
            alpha=0.15
        )

        axes[1].set_ylim(0, max(ys) * 1.15)

    axes[1].set_xlim(left=0)

    if log:
        axes[1].set_ylim(0, max(ys) * 1.15)

    # ─────────────────────────────
    # QUEUE SIZE GRAPH
    # ─────────────────────────────
    axes[2].set_title(
        "Vehicles Waiting Over Time",
        color="white",
        fontsize=10
    )

    axes[2].set_xlabel(
        "Simulation Time (s)",
        color="#cfd6e6",
        fontsize=8
    )

    axes[2].set_ylabel(
        "Queued Vehicles",
        color="#cfd6e6",
        fontsize=8
    )

    if qlog:
        sample_rate = max(1, len(qlog) // 350)
        sampled = qlog[::sample_rate]

        xs = [q[0] for q in sampled]
        ys = [q[1] for q in sampled]

        axes[2].plot(xs, ys, linewidth=2)

        axes[2].fill_between(
            xs,
            ys,
            alpha=0.15
        )

        peak = max(ys) if ys else 1

        axes[2].text(
        xs[-1] * 0.65,
        peak * 0.88,
        f"PEAK QUEUE: {peak}",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.85,
            edgecolor="white",
            boxstyle="round,pad=0.45"
        )
    )
    
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(bottom=0)

    fig.subplots_adjust(
    left=0.11,
    right=0.97,
    top=0.955,
    bottom=0.06,
    hspace=0.72
    )

    # Convert matplotlib → pygame surface
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()

    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()

    canvas_w, canvas_h = canvas.get_width_height()

    surf = pygame.image.frombuffer(
        raw_data,
        (canvas_w, canvas_h),
        "RGBA"
    ).convert_alpha()

    if canvas_w != int(w) or canvas_h != int(h):
        surf = pygame.transform.smoothscale(
            surf,
            (int(w), int(h))
        )

    plt.close(fig)

    _chart_cache = surf.copy()
    del surf

    return _chart_cache

# ═══════════════════════════════════════════════════════
#  STATS PANEL
# ═══════════════════════════════════════════════════════
def draw_panel(surf, sd, fonts, px, py, pw, ph, road_dirty_flag):
    font,fsm,fxs=fonts
    rrect(surf,C["panel"],(px,py,pw,ph),r=10)
    pygame.draw.rect(surf,C["border"],(px,py,pw,ph),1,border_radius=10)
    y=py+12

    def ctr(txt,fy,fnt,col):
        s=fnt.render(txt,True,col)
        surf.blit(s,(px+pw//2-s.get_width()//2,fy))
        return fy+s.get_height()+3
    def div(fy):
        pygame.draw.line(surf,C["border"],(px+10,fy),(px+pw-10,fy)); return fy+8
    def row(lbl,val,fy,vc=None):
        ls=fxs.render(lbl,True,C["text_dim"]); vs=fxs.render(str(val),True,vc or C["text"])
        surf.blit(ls,(px+12,fy)); surf.blit(vs,(px+pw-12-vs.get_width(),fy)); return fy+16

    y=ctr("TRAFFIC SIM",y,font,C["accent"])
    y=ctr("CS 324 · BatStateU",y,fxs,C["text_dim"])
    y=div(y); y=ctr(sd.scenario,y,fsm,C["accent2"]); y=div(y)

    with sd.lock:
        lights=list(sd.lights); timers=list(sd.timers); active=sd.active_dir
        sim_t=sd.sim_time; tv=sd.total_vehicles
        wait=sum(1 for v in sd.vehicles if v["state"]=="queued")
        move=sum(1 for v in sd.vehicles if v["state"]=="moving")
        appr=sum(1 for v in sd.vehicles if v["state"]=="approach")
        done=len(sd.completed); waits=list(sd.wait_times)

    # 4 small light boxes (2×2 grid)
    dirs_label=["N","E","S","W"]
    bw,bh=48,38
    bx0=px+6; by0=y
    for i in range(4):
        bx=bx0+(i%2)*(bw+4); by=by0+(i//2)*(bh+4)
        cm={"green":C["green_light"],"yellow":C["yellow_light"],"red":C["red_light"]}
        col=cm.get(lights[i],C["text_dim"])
        rrect(surf,(24,28,42),(bx,by,bw,bh),r=5)
        pygame.draw.rect(surf,col,(bx,by,bw,bh),2,border_radius=5)
        lbl=fxs.render(dirs_label[i],True,C["text_dim"])
        surf.blit(lbl,(bx+bw//2-lbl.get_width()//2,by+3))
        st=fxs.render(lights[i][:3].upper(),True,col)
        surf.blit(st,(bx+bw//2-st.get_width()//2,by+14))
        tm=fxs.render(f"{int(timers[i])}s",True,C["text_dim"])
        surf.blit(tm,(bx+bw//2-tm.get_width()//2,by+24))
    y=by0+bh*2+8+6; y=div(y)
    y=row("Sim Time",f"{sim_t:.1f}s",y)
    y=row("Spawned",tv,y); y=row("Completed",done,y,C["green_light"])
    y=row("Approaching",appr,y,(180,180,255))
    y=row("Waiting",wait,y,C["red_light"]); y=row("Moving",move,y,C["yellow_light"])
    y=div(y)
    avg=sum(waits)/len(waits) if waits else 0; mx=max(waits) if waits else 0
    y=row("Avg Wait",f"{avg:.1f}s",y); y=row("Max Wait",f"{mx:.1f}s",y); y=div(y)
    sp=fxs.render(f"Speed x{sd.speed_factor:.1f}",True,C["accent"])
    surf.blit(sp,(px+pw//2-sp.get_width()//2,y)); y+=16; y=div(y)

    # ── Lane toggle ──
    ln=fxs.render(f"Lanes/dir: {N_LANES}",True,C["accent2"])
    surf.blit(ln,(px+pw//2-ln.get_width()//2,y)); y+=14
    hint=fxs.render("[L] cycle lanes (2/3/4/6)",True,C["text_dim"])
    surf.blit(hint,(px+pw//2-hint.get_width()//2,y)); y+=16; y=div(y)

    for h in ["[1]Normal [2]Rush [3]Low",
               "[UP]Faster  [DOWN]Slower",
               "[C]Charts  [R]Reset  [Q]Quit"]:
        hs=fxs.render(h,True,C["text_dim"])
        surf.blit(hs,(px+pw//2-hs.get_width()//2,y)); y+=14

# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    global N_LANES, LANE_W, LANE_OFFS, HR
    global _chart_cache, _chart_last_n
    global CURRENT_THEME, C

    pygame.init()
    screen=pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("CS 324 — Traffic Light Simulation | BatStateU CICS")
    clock=pygame.time.Clock()
    font=pygame.font.SysFont("monospace",15,bold=True)
    fsm =pygame.font.SysFont("monospace",12,bold=True)
    fxs =pygame.font.SysFont("monospace",10)

    road_surf   = build_road_surface()
    road_dirty  = [False]

    threading.Thread(target=run_simulation,args=(SD,),daemon=True).start()

    px,py=WIDTH-PANEL_W-6,6; pw,ph=PANEL_W,HEIGHT-12
    show_charts=False

    running=True
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.KEYDOWN:
                if   event.key==pygame.K_q: running=False
                elif event.key==pygame.K_1: SD.scenario="Normal"
                elif event.key==pygame.K_2: SD.scenario="Rush"
                elif event.key==pygame.K_3: SD.scenario="Low"
                elif event.key==pygame.K_UP:
                    SD.speed_factor=min(8.0,round(SD.speed_factor+0.5,1))
                elif event.key==pygame.K_DOWN:
                    SD.speed_factor=max(0.5,round(SD.speed_factor-0.5,1))
                elif event.key==pygame.K_c:
                    show_charts=not show_charts
                elif event.key==pygame.K_r:
                    SD.reset_flag=True
                    _chart_cache=None; _chart_last_n=-1
                elif event.key==pygame.K_l:
                    # Cycle lane count, rebuild geometry and road
                    idx=(LANE_OPTIONS.index(N_LANES)+1)%len(LANE_OPTIONS)
                    N_LANES=LANE_OPTIONS[idx]
                    LANE_W,LANE_OFFS,HR=compute_geometry(N_LANES)
                    SD.reset_flag=True
                    _chart_cache=None; _chart_last_n=-1
                    road_dirty[0]=True
                elif event.key == pygame.K_t:
                    names = list(THEMES.keys())

                    idx = names.index(CURRENT_THEME)
                    CURRENT_THEME = names[(idx + 1) % len(names)]

                    C = THEMES[CURRENT_THEME]

                    road_surf = build_road_surface()

        # Rebuild road surface if lane count changed
        if road_dirty[0]:
            road_surf=build_road_surface()
            road_dirty[0]=False

        screen.fill(C["bg"])
        if show_charts:
            chart_top = 34

            screen.blit(
                build_chart(SD, VIEW_W, HEIGHT - chart_top),
                (0, chart_top)
            )
            header = pygame.Surface((VIEW_W, 34), pygame.SRCALPHA)
            pygame.draw.rect(header, (0, 0, 0, 120), (0, 0, VIEW_W, 34))
            screen.blit(header, (0, 0))

            lb = font.render(
                "TRAFFIC ANALYTICS DASHBOARD  •  Press [C] to return",
                True,
                (230, 235, 245)
            )

            screen.blit(lb, (18, 8))
        else:
            screen.blit(road_surf,(0,0))
            draw_lights(screen,SD,fxs)
            draw_vehicles(screen,SD,fxs)
        draw_panel(screen,SD,(font,fsm,fxs),px,py,pw,ph,road_dirty)
        fps=fxs.render(f"FPS {int(clock.get_fps())}",True,C["text_dim"])
        screen.blit(fps,(6,6))
        pygame.display.flip()
        clock.tick(60)

    SD.running=False
    pygame.quit()
    if SD.completed:
        df=pd.DataFrame([{k:v for k,v in c.items() if k!="path"} for c in SD.completed])
        df.to_csv("simulation_results.csv",index=False)
        print(f"Results saved ({len(SD.completed)} vehicles)")
    sys.exit(0)

if __name__=="__main__":
    main()