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
8. Pedestrians spawn at crosswalk edges during the all-red clearance phase.
   They walk across the road and disappear on the other side.
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
N_LANES      = 3

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
SLOT      = CAR_LEN + CAR_GAP
STOP_DIST = 12

CAR_SPEED = 1.0

# ═══════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════
SIM_FPS = 60.0
FRAME_T = 1.0 / SIM_FPS

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
#  PEDESTRIAN CONFIGURATION
# ═══════════════════════════════════════════════════════
PED_SPEED        = 0.6          # pixels per frame
PED_RADIUS = 7                  # drawn circle radius
PED_SPAWN_CHANCE = 0.55         # probability a pedestrian spawns each clear phase per crosswalk
PED_COLORS = [
    (255, 200, 120),  # warm skin
    (200, 150,  90),  # tan
    (140,  90,  50),  # dark
    (255, 220, 180),  # light
    (180, 130,  80),  # medium
]
SHIRT_COLORS = [
    (220,  60,  60),
    ( 60, 120, 220),
    ( 50, 180,  80),
    (200, 160,  30),
    (160,  60, 200),
    (240, 120,  30),
    ( 80, 200, 200),
]

# ═══════════════════════════════════════════════════════
#  COLOURS / THEMES
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
            (80,162,255),(255,102,65),(100,208,125),(255,198,60),
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
            (0,255,255),(255,0,180),(120,255,0),(255,140,0),
        ]
    }
}

CURRENT_THEME = "classic"
C = THEMES[CURRENT_THEME]

# ═══════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ═══════════════════════════════════════════════════════
def ib_coord(from_dir, lane):
    lane = max(0, min(lane, len(LANE_OFFS)-1))
    off = LANE_OFFS[lane]
    if from_dir == 0: return CX + off
    if from_dir == 2: return CX - off
    if from_dir == 1: return CY + off
    return             CY - off

def ob_coord(exit_dir, lane):
    lane = max(0, min(lane, len(LANE_OFFS)-1))
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

    exit_dir = {"right":    (from_dir+3)%4,
                "straight": (from_dir+2)%4,
                "left":     (from_dir+1)%4}[turn]

    ex_lane = min(lane, N_LANES - 1)
    ec = ob_coord(exit_dir, ex_lane)

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
        if from_dir in (0, 2):
            box_out = (ic, box_out[1])
            depart  = (ic, depart[1])
        else:
            box_out = (box_out[0], ic)
            depart  = (depart[0], ic)
        return [spawn, stop_pt, box_in, box_out, depart]

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
    dist = max(0.0, dist)
    acc  = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        sl = math.hypot(dx, dy)
        if sl < 1e-9:
            continue
        if dist <= acc + sl:
            t = (dist - acc) / sl
            return (path[i][0] + t*dx,
                    path[i][1] + t*dy,
                    math.degrees(math.atan2(dy, dx)))
        acc += sl

    dx = path[-1][0] - path[-2][0]
    dy = path[-1][1] - path[-2][1]
    return path[-1][0], path[-1][1], math.degrees(math.atan2(dy, dx))

# ═══════════════════════════════════════════════════════
#  QUEUE PIXEL POSITION
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
#  PEDESTRIAN HELPERS
# ═══════════════════════════════════════════════════════
def _crosswalk_endpoints(axis, side):
    """
    Smaller realistic pedestrian crossing paths.

    axis:
        "NS" → crosses vertical road (walks LEFT/RIGHT)
        "EW" → crosses horizontal road (walks UP/DOWN)
    """

    # Smaller crossing size
    cross_half = HR + 8

    # Keep pedestrians close to traffic lights
    signal_offset = HR + 28

    if axis == "NS":
        # Horizontal walking
        left_x  = CX - cross_half
        right_x = CX + cross_half

        y = CY - signal_offset if side == "west" else CY + signal_offset

        # Random walk direction
        if random.random() < 0.5:
            return left_x, y, right_x, y
        else:
            return right_x, y, left_x, y

    else:  # "EW"
        # Vertical walking
        top_y    = CY - cross_half
        bottom_y = CY + cross_half

        x = CX - signal_offset if side == "west" else CX + signal_offset

        if random.random() < 0.5:
            return x, top_y, x, bottom_y
        else:
            return x, bottom_y, x, top_y


def make_pedestrian(pid, axis, side):

    sx, sy, ex, ey = _crosswalk_endpoints(axis, side)

    dist = math.hypot(ex - sx, ey - sy)

    return {
        "id": pid,

        "sx": sx,
        "sy": sy,

        "ex": ex,
        "ey": ey,

        "total": dist,
        "walked": 0.0,

        # NEW
        "state": "waiting",
        "wait_timer": random.uniform(0.5, 2.0),

        "done": False,

        "skin": random.choice(PED_COLORS),
        "shirt": random.choice(SHIRT_COLORS),

        "bob": random.uniform(0, math.pi * 2),
    }


# ═══════════════════════════════════════════════════════
#  SIMULATION STATE
# ═══════════════════════════════════════════════════════
class SimData:
    def __init__(self):
        self.lock            = threading.Lock()
        self.vehicles        : list[dict] = []
        self.completed       : list[dict] = []
        self.pedestrians     : list[dict] = []   # ← NEW
        self.ped_id_counter  = 0                 # ← NEW
        self.ped_phase_active = False            # ← NEW: True during all-red clearance
        self.lights          = ["green","red","red","red"]
        self.timers          = [30,0,0,0]
        self.active_dir      = 0
        self.sim_time        = 0.0
        self.total_vehicles  = 0
        self.scenario        = "Normal"
        self.running         = True
        self.speed_factor    = 1.0
        self.wait_times      : list[float] = []
        self.throughput_log  : list[tuple] = []
        self.queue_log       : list[tuple] = []
        self.max_queue_seen  = 0
        self.reset_time      = 0.0
        self.reset_flag      = False

SD = SimData()

def pedestrian_crossing_active(sd):
    return any(not p["done"] for p in sd.pedestrians)

# ═══════════════════════════════════════════════════════
#  SIMPY ENGINE
# ═══════════════════════════════════════════════════════
def run_simulation(sd: SimData):
    env = simpy.Environment()

    def light_ctrl(env):
        cur = 0
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            G, Y, CLR = cfg["green"], cfg["yellow"], cfg["clear"]

            # Green
            with sd.lock:
                sd.lights      = ["red"]*4
                sd.lights[cur] = "green"
                sd.timers      = [0]*4
                sd.timers[cur] = G
                sd.active_dir  = cur
                sd.ped_phase_active = False
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
                sd.ped_phase_active = False
            for t in range(Y):
                if not sd.running: return
                yield env.timeout(1.0)
                with sd.lock:
                    sd.timers[cur] = max(0, Y-t-1)
                    sd.sim_time    = env.now

            # ── All-red clearance — PEDESTRIAN PHASE ──
            with sd.lock:
                sd.lights = ["red"]*4
                sd.timers = [CLR]*4
                sd.ped_phase_active = True

                # Spawn pedestrians at all four crosswalk edges
                crosswalks = [
                    ("NS", "west"),
                    ("NS", "east"),
                    ("EW", "west"),
                    ("EW", "east"),
                ]
                for axis, side in crosswalks:
                    if random.random() < PED_SPAWN_CHANCE:
                        sd.ped_id_counter += 1
                        sd.pedestrians.append(
                            make_pedestrian(sd.ped_id_counter, axis, side)
                        )

            for t in range(CLR):
                if not sd.running: return
                yield env.timeout(1.0)
                with sd.lock:
                    for i in range(4): sd.timers[i] = max(0, CLR-t-1)
                    sd.sim_time = env.now

            with sd.lock:
                sd.ped_phase_active = False

            cur = (cur + 1) % 4

    # ── Spawner ──
    def gen_vehicles(env, from_dir):
        vid = 0
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            yield env.timeout(random.expovariate(1.0 / cfg["arrival"]))

            if sd.reset_flag:
                yield env.timeout(0.1)
                continue
            vid += 1
            turn = random.choices(TURNS, TURN_PROBS)[0]

            if turn == "left":
                lane = N_LANES - 1
            elif turn == "right":
                lane = 0
            else:
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

    # ── Mover (vehicles + pedestrians) ──
    def mover(env):
        while sd.running:
            yield env.timeout(FRAME_T)
            with sd.lock:
                if sd.reset_flag:
                    sd.vehicles.clear()
                    sd.completed.clear()
                    sd.pedestrians.clear()        # ← clear peds on reset
                    sd.wait_times.clear()
                    sd.throughput_log.clear()
                    sd.queue_log.clear()
                    sd.sim_time       = 0.0
                    sd.total_vehicles = 0
                    sd.reset_flag     = False
                    sd.reset_time     = env.now

                lights = list(sd.lights)
                now    = env.now
                # ─────────────────────────────
                # Move pedestrians
                # ─────────────────────────────
                still_walking = []

                for p in sd.pedestrians:

                    if p["done"]:
                        continue

                    # WAITING STATE
                    if p["state"] == "waiting":

                        p["wait_timer"] -= FRAME_T

                        if p["wait_timer"] <= 0:
                            p["state"] = "crossing"

                        still_walking.append(p)
                        continue

                    # CROSSING STATE
                    if p["state"] == "crossing":

                        p["walked"] = min(
                            p["total"],
                            p["walked"] + PED_SPEED
                        )

                        p["bob"] += 0.25

                        if p["walked"] >= p["total"] - 0.5:
                            p["done"] = True
                        else:
                            still_walking.append(p)

                sd.pedestrians = still_walking

                # ── Vehicle logic (unchanged) ──
                queued_count = sum(
                    1 for v in sd.vehicles if v["state"] == "queued"
                )
                sd.max_queue_seen = max(sd.max_queue_seen, queued_count)
                sd.queue_log.append((now - sd.reset_time, queued_count))

                ql: dict[tuple,list] = {}
                for v in sd.vehicles:
                    if v["state"] == "queued":
                        key = (v["from_dir"], v["lane"])
                        ql.setdefault(key,[]).append(v)
                for q in ql.values():
                    q.sort(key=lambda v: v["arrive"])
                    for slot,v in enumerate(q): v["queue_slot"] = slot

                mov_lane = {}
                for v in sd.vehicles:
                    if v["state"] == "moving":
                        key = (v["from_dir"], v["lane"])
                        mov_lane.setdefault(key, []).append(v["dist"])
                for lst in mov_lane.values():
                    lst.sort()

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

                    if v["state"] == "approach":
                        q      = ql.get(key, [])
                        n_q    = len(q)
                        tail_d = v["stop_d"] - (n_q + 1) * SLOT

                        ahead_mv = [dd for dd in mov_lane.get((d, lane), []) if dd > v["dist"]]
                        if ahead_mv:
                            gap  = min(ahead_mv) - v["dist"]
                            safe = CAR_LEN + CAR_GAP
                            spd  = CAR_SPEED * max(0.0,(gap/safe)-0.05) if gap < safe else CAR_SPEED
                        else:
                            spd = CAR_SPEED

                        new_d = min(v["dist"] + spd, max(v["dist"], tail_d))
                        v["dist"] = new_d

                        if v["dist"] >= tail_d - 0.5:
                            v["dist"]        = tail_d
                            v["state"]       = "queued"
                            v["queue_slot"]  = n_q
                            ql.setdefault(key,[]).append(v)
                            ql[key].sort(key=lambda v: v["arrive"])
                            for i,qv in enumerate(ql[key]): qv["queue_slot"] = i

                    elif v["state"] == "queued":
                        v["wait"] = now - v["arrive"]
                        slot = v.get("queue_slot", 0)
                        q    = ql.get(key, [])

                        if grn and slot == 0 and not pedestrian_crossing_active(sd):
                            lane_clear = True
                            for ov in sd.vehicles:
                                if (ov is not v
                                        and ov["state"] == "moving"
                                        and ov["from_dir"] == d
                                        and ov["lane"] == lane):
                                    if abs(ov["dist"] - v["stop_d"]) < SLOT * 1.2:
                                        lane_clear = False
                                        break
                            if lane_clear:
                                v["state"] = "moving"
                                v["dist"]  = v["stop_d"]
                                mov_lane.get((d, lane), []).append(v["dist"])
                                mov_lane.get((d, lane), []).sort()
                                q.remove(v)
                                for i, rv in enumerate(q): rv["queue_slot"] = i

                        elif grn and slot > 0 and not pedestrian_crossing_active(sd):
                            leader = q[slot-1]
                            if (leader["state"] == "moving"
                                    and leader["dist"] >= v["stop_d"] + SLOT):
                                v["state"] = "moving"
                                v["dist"]  = v["stop_d"]
                                mov_lane.get((d, lane), []).append(v["dist"])
                                mov_lane.get((d, lane), []).sort()
                                in_box.add(d)
                                q.remove(v)
                                for i,rv in enumerate(q): rv["queue_slot"] = i

                    elif v["state"] == "moving":
                        ahead_mv = [dd for dd in mov_lane.get((d, lane), []) if dd > v["dist"] + 0.5]
                        if ahead_mv:
                            gap  = min(ahead_mv) - v["dist"]
                            safe = CAR_LEN + CAR_GAP
                            spd  = CAR_SPEED * max(0.0,(gap/safe)-0.05) if gap < safe else CAR_SPEED
                        else:
                            spd = CAR_SPEED

                        old_d     = v["dist"]
                        # ─────────────────────────
                        # PEDESTRIAN CROSSING YIELD
                        # ─────────────────────────
                        if pedestrian_crossing_active(sd):

                            cross_limit = v["stop_d"] - 8

                            # approaching the crosswalk
                            if old_d < cross_limit:
                                next_d = min(v["plen"], old_d + spd)

                                if next_d >= cross_limit:
                                    next_d = cross_limit
                                    spd = 0

                                v["dist"] = next_d
                            else:
                                v["dist"] = old_d

                        else:
                            v["dist"] = min(v["plen"], old_d + spd)
                        lane_list = mov_lane.get((d, lane), [])
                        if old_d in lane_list:
                            lane_list.remove(old_d)
                        mov_lane.get((d, lane), []).append(v["dist"])
                        mov_lane.get((d, lane), []).sort()

                        if v["dist"] >= v["plen"] - 0.5:
                            v["state"]  = "done"
                            v["depart"] = now
                            sd.wait_times.append(v["wait"])
                            sd.throughput_log.append((now, len(sd.completed)+1))
                            remove.append(v)

                for v in remove:
                    sd.vehicles.remove(v)
                    sd.completed.append(v)

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
#  STATIC ROAD SURFACE
# ═══════════════════════════════════════════════════════
def build_road_surface():
    surf = pygame.Surface((VIEW_W, HEIGHT))
    cx, cy = CX, CY
    surf.fill(C["grass"])

    pygame.draw.rect(surf, C["road"],     (cx-HR, 0,     HR*2,   HEIGHT))
    pygame.draw.rect(surf, C["road"],     (0,     cy-HR, VIEW_W, HR*2))
    pygame.draw.rect(surf, C["road_box"], (cx-HR, cy-HR, HR*2,   HR*2))

    dh = DIV_HALF

    pygame.draw.rect(surf, C["divider"], (cx-dh, 0,      DIV_W, cy-HR))
    pygame.draw.rect(surf, C["divider"], (cx-dh, cy+HR,  DIV_W, HEIGHT))
    pygame.draw.rect(surf, C["divider"], (0,      cy-dh, cx-HR, DIV_W))
    pygame.draw.rect(surf, C["divider"], (cx+HR,  cy-dh, VIEW_W, DIV_W))

    dk = C["lane_dash"]
    dl, dg = 18, 12

    for i in range(1, N_LANES):
        lx_ib = cx + dh + i * LANE_W
        lx_ob = cx - dh - i * LANE_W
        for y0, y1 in [(0, cy-HR), (cy+HR, HEIGHT)]:
            for y in range(y0, y1, dl+dg):
                w = min(dl, y1-y)
                pygame.draw.rect(surf, dk, (lx_ib-1, y, 2, w))
                pygame.draw.rect(surf, dk, (lx_ob-1, y, 2, w))
        ly_ib = cy + dh + i * LANE_W
        ly_ob = cy - dh - i * LANE_W
        for x0, x1 in [(0, cx-HR), (cx+HR, VIEW_W)]:
            for x in range(x0, x1, dl+dg):
                w = min(dl, x1-x)
                pygame.draw.rect(surf, dk, (x, ly_ib-1, w, 2))
                pygame.draw.rect(surf, dk, (x, ly_ob-1, w, 2))

    kc = C["kerb"]
    pygame.draw.line(surf, kc, (cx-HR, 0),    (cx-HR, HEIGHT), 2)
    pygame.draw.line(surf, kc, (cx+HR, 0),    (cx+HR, HEIGHT), 2)
    pygame.draw.line(surf, kc, (0, cy-HR),    (VIEW_W, cy-HR), 2)
    pygame.draw.line(surf, kc, (0, cy+HR),    (VIEW_W, cy+HR), 2)

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

    zebra  = (240, 240, 240)
    border = (90, 255, 140)

    # ====================================================
    # DYNAMIC VALUES
    # ====================================================

    crosswalk_w = N_LANES * LANE_W
    stripe_w    = 10
    stripe_gap  = 8

    lane_pad = 10

    total = crosswalk_w

    offset = HR + 6

    lane_thickness = 42

    # ====================================================
    # GREEN WALKING LANE
    # ====================================================

    glow = pygame.Surface((VIEW_W, HEIGHT), pygame.SRCALPHA)

    lane_col = (80, 255, 120, 40)

    # TOP
    pygame.draw.rect(
        glow,
        lane_col,
        (
            cx - total//2 - lane_pad,
            cy - offset - lane_thickness,
            total + lane_pad*2,
            lane_thickness
        )
    )

    # BOTTOM
    pygame.draw.rect(
        glow,
        lane_col,
        (
            cx - total//2 - lane_pad,
            cy + offset,
            total + lane_pad*2,
            lane_thickness
        )
    )

    # LEFT
    pygame.draw.rect(
        glow,
        lane_col,
        (
            cx - offset - lane_thickness,
            cy - total//2 - lane_pad,
            lane_thickness,
            total + lane_pad*2
        )
    )

    # RIGHT
    pygame.draw.rect(
        glow,
        lane_col,
        (
            cx + offset,
            cy - total//2 - lane_pad,
            lane_thickness,
            total + lane_pad*2
        )
    )

    surf.blit(glow, (0, 0))

    # ====================================================
    # OUTLINE
    # ====================================================

    pygame.draw.rect(
        surf,
        border,
        (
            cx - total//2 - lane_pad,
            cy - offset - lane_thickness,
            total + lane_pad*2,
            lane_thickness
        ),
        2
    )

    pygame.draw.rect(
        surf,
        border,
        (
            cx - total//2 - lane_pad,
            cy + offset,
            total + lane_pad*2,
            lane_thickness
        ),
        2
    )

    pygame.draw.rect(
        surf,
        border,
        (
            cx - offset - lane_thickness,
            cy - total//2 - lane_pad,
            lane_thickness,
            total + lane_pad*2
        ),
        2
    )

    pygame.draw.rect(
        surf,
        border,
        (
            cx + offset,
            cy - total//2 - lane_pad,
            lane_thickness,
            total + lane_pad*2
        ),
        2
    )

    # ====================================================
    # ZEBRA STRIPES
    # ====================================================

    stripes = max(4, total // (stripe_w + stripe_gap))

    start_x = cx - total//2
    start_y = cy - total//2

    # TOP + BOTTOM
    for i in range(stripes):

        x = start_x + i * (stripe_w + stripe_gap)

        pygame.draw.rect(
            surf,
            zebra,
            (
                x,
                cy - offset - lane_thickness,
                stripe_w,
                lane_thickness
            )
        )

        pygame.draw.rect(
            surf,
            zebra,
            (
                x,
                cy + offset,
                stripe_w,
                lane_thickness
            )
        )

    # LEFT + RIGHT
    for i in range(stripes):

        y = start_y + i * (stripe_w + stripe_gap)

        pygame.draw.rect(
            surf,
            zebra,
            (
                cx - offset - lane_thickness,
                y,
                lane_thickness,
                stripe_w
            )
        )

        pygame.draw.rect(
            surf,
            zebra,
            (
                cx + offset,
                y,
                lane_thickness,
                stripe_w
            )
        )
# ═══════════════════════════════════════════════════════
#  TRAFFIC LIGHTS
# ═══════════════════════════════════════════════════════
def draw_lights(surf, sd, fxs):
    cx, cy = CX, CY
    with sd.lock:
        lights = list(sd.lights)
        timers = list(sd.timers)
        active = sd.active_dir
        ped_phase = sd.ped_phase_active

    def pole(px, py, state, show_walk=False):
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

        # Pedestrian walk signal: small white walking figure below the pole
        if show_walk:
            wx, wy = px, py + 30
            sig_col = (80, 220, 120)
            pygame.draw.circle(surf, sig_col, (wx, wy),      4)        # head
            pygame.draw.line(surf,   sig_col, (wx, wy+4),  (wx, wy+12), 2)  # body
            pygame.draw.line(surf,   sig_col, (wx, wy+6),  (wx-5, wy+10), 2) # left arm
            pygame.draw.line(surf,   sig_col, (wx, wy+6),  (wx+5, wy+10), 2) # right arm
            pygame.draw.line(surf,   sig_col, (wx, wy+12), (wx-4, wy+19), 2) # left leg
            pygame.draw.line(surf,   sig_col, (wx, wy+12), (wx+4, wy+19), 2) # right leg

    pole(cx-HR-22, cy-HR-54, lights[0], show_walk=ped_phase)
    pole(cx+HR+8,  cy-HR-54, lights[1], show_walk=ped_phase)
    pole(cx+HR+8,  cy+HR+4,  lights[2], show_walk=ped_phase)
    pole(cx-HR-22, cy+HR+4,  lights[3], show_walk=ped_phase)

    if ped_phase:
        pygame.draw.rect(
            surf,
            (40, 180, 80),
            (cx-18, cy-18, 36, 36),
            border_radius=6
        )

        walk = fxs.render("WALK", True, (255,255,255))
        surf.blit(
            walk,
            (cx - walk.get_width()//2,
                cy - walk.get_height()//2)
        )

    tb=pygame.Surface((52,18),pygame.SRCALPHA)
    pygame.draw.rect(tb,(0,0,0,165),(0,0,52,18),border_radius=4)
    surf.blit(tb,(cx-26,cy-9))
    label = f"D{active}:{int(timers[active]):02d}s"
    ts=fxs.render(label,True,C["text"])
    surf.blit(ts,(cx-ts.get_width()//2,cy-8))

# ═══════════════════════════════════════════════════════
#  PEDESTRIAN RENDERER
# ═══════════════════════════════════════════════════════
def draw_pedestrians(surf, sd):

    with sd.lock:
        peds = list(sd.pedestrians)

    for p in peds:

        if p["done"]:
            continue

        # WAITING POSITION
        if p["state"] == "waiting":

            px = int(p["sx"])
            py = int(p["sy"])

        else:

            t = p["walked"] / max(1, p["total"])

            px = int(
                p["sx"] + t * (p["ex"] - p["sx"])
            )

            py = int(
                p["sy"] + t * (p["ey"] - p["sy"])
            )

        bob = int(math.sin(p["bob"]) * 2)

        skin  = p["skin"]
        shirt = p["shirt"]

        r = PED_RADIUS

        # shadow
        shadow = pygame.Surface((r*5, r*3), pygame.SRCALPHA)

        pygame.draw.ellipse(
            shadow,
            (0,0,0,80),
            (0,0,r*5,r*3)
        )

        surf.blit(
            shadow,
            (px-r*2, py+r*2)
        )

        # waiting glow
        if p["state"] == "waiting":

            glow = pygame.Surface((28,28), pygame.SRCALPHA)

            pygame.draw.circle(
                glow,
                (80,255,120,80),
                (14,14),
                12
            )

            surf.blit(glow, (px-14, py-14))

        # body
        pygame.draw.circle(
            surf,
            shirt,
            (px, py + bob + r),
            r
        )

        # head
        pygame.draw.circle(
            surf,
            skin,
            (px, py + bob - r + 1),
            r - 1
        )

        # legs
        leg_off = int(math.sin(p["bob"] * 2) * 3)

        pygame.draw.circle(
            surf,
            skin,
            (px - 2, py + bob + r*2 + leg_off),
            2
        )

        pygame.draw.circle(
            surf,
            skin,
            (px + 2, py + bob + r*2 - leg_off),
            2
        )
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
_chart_cache  = None
_chart_last_n = -1

def build_chart(sd, w, h):
    global _chart_cache, _chart_last_n

    with sd.lock:
        waits = list(sd.wait_times)
        log   = list(sd.throughput_log)
        qlog  = list(getattr(sd, "queue_log", []))
        done  = len(sd.completed)

    if abs(done - _chart_last_n) < 5 and _chart_cache is not None:
        return _chart_cache

    _chart_last_n = done

    fig, axes = plt.subplots(3, 1, figsize=(w/100, h/100), dpi=100)
    fig.patch.set_facecolor("#0c0e14")

    for ax in axes:
        ax.set_facecolor("#171926")
        ax.tick_params(colors="#bfc7d5", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#30364a")
        ax.grid(True, alpha=0.15)

    axes[0].set_title("Vehicle Wait Time Distribution", color="white", fontsize=10)
    axes[0].set_xlabel("Wait Time (s)", color="#cfd6e6", fontsize=8)
    axes[0].set_ylabel("Vehicles", color="#cfd6e6", fontsize=8)
    avg_wait = 0
    if waits:
        bins = max(6, min(20, len(waits)//3+1))
        axes[0].hist(waits, bins=bins)
        avg_wait = sum(waits)/len(waits)
        axes[0].axvline(avg_wait, linestyle="--", linewidth=2)
        axes[0].text(avg_wait, axes[0].get_ylim()[1]*0.88,
                     f"AVG: {avg_wait:.1f}s", fontsize=11, fontweight="bold",
                     color="white",
                     bbox=dict(facecolor="black",alpha=0.85,edgecolor="white",boxstyle="round,pad=0.45"))

    axes[1].set_title("Intersection Throughput", color="white", fontsize=10)
    axes[1].set_xlabel("Simulation Time (s)", color="#cfd6e6", fontsize=8)
    axes[1].set_ylabel("Vehicles Passed", color="#cfd6e6", fontsize=8)
    if log:
        xs = [x[0] for x in log]
        ys = list(range(1, len(log)+1))
        axes[1].plot(xs, ys, linewidth=2)
        axes[1].fill_between(xs, ys, alpha=0.15)
        axes[1].set_ylim(0, max(ys)*1.15)
    axes[1].set_xlim(left=0)

    axes[2].set_title("Vehicles Waiting Over Time", color="white", fontsize=10)
    axes[2].set_xlabel("Simulation Time (s)", color="#cfd6e6", fontsize=8)
    axes[2].set_ylabel("Queued Vehicles", color="#cfd6e6", fontsize=8)
    if qlog:
        sr      = max(1, len(qlog)//350)
        sampled = qlog[::sr]
        xs = [q[0] for q in sampled]
        ys = [q[1] for q in sampled]
        axes[2].plot(xs, ys, linewidth=2)
        axes[2].fill_between(xs, ys, alpha=0.15)
        peak = max(ys) if ys else 1
        axes[2].text(xs[-1]*0.65, peak*0.88,
                     f"PEAK QUEUE: {peak}", fontsize=11, fontweight="bold",
                     color="white",
                     bbox=dict(facecolor="black",alpha=0.85,edgecolor="white",boxstyle="round,pad=0.45"))
        axes[2].set_xlim(left=0)
        axes[2].set_ylim(bottom=0)

    fig.subplots_adjust(left=0.11,right=0.97,top=0.955,bottom=0.06,hspace=0.72)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer  = canvas.get_renderer()
    raw_data  = renderer.buffer_rgba()
    cw2, ch2  = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data,(cw2,ch2),"RGBA").convert_alpha()
    if cw2 != int(w) or ch2 != int(h):
        surf = pygame.transform.smoothscale(surf,(int(w),int(h)))
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
        n_peds=len(sd.pedestrians)
        ped_phase=sd.ped_phase_active

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

    # ── Pedestrian row ──
    ped_col = (80, 220, 120) if ped_phase else C["text_dim"]
    ped_lbl = "WALK" if ped_phase else "---"
    y=row("Pedestrians",f"{n_peds} ({ped_lbl})",y, ped_col)
    y=div(y)

    avg=sum(waits)/len(waits) if waits else 0; mx=max(waits) if waits else 0
    y=row("Avg Wait",f"{avg:.1f}s",y); y=row("Max Wait",f"{mx:.1f}s",y); y=div(y)
    sp=fxs.render(f"Speed x{sd.speed_factor:.1f}",True,C["accent"])
    surf.blit(sp,(px+pw//2-sp.get_width()//2,y)); y+=16; y=div(y)

    ln=fxs.render(f"Lanes/dir: {N_LANES}",True,C["accent2"])
    surf.blit(ln,(px+pw//2-ln.get_width()//2,y)); y+=14
    hint=fxs.render("[L] cycle lanes (2/3/4/6)",True,C["text_dim"])
    surf.blit(hint,(px+pw//2-hint.get_width()//2,y)); y+=16; y=div(y)

    for h in ["[1]Normal [2]Rush [3]Low",
               "[UP]Faster  [DOWN]Slower",
               "[C]Charts  [R]Reset  [Q]Quit",
               "[T]Theme"]:
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

    road_surf  = build_road_surface()
    road_dirty = [False]

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
                    idx=(LANE_OPTIONS.index(N_LANES)+1)%len(LANE_OPTIONS)
                    N_LANES=LANE_OPTIONS[idx]
                    LANE_W,LANE_OFFS,HR=compute_geometry(N_LANES)
                    SD.reset_flag=True
                    _chart_cache=None; _chart_last_n=-1
                    road_dirty[0]=True
                elif event.key==pygame.K_t:
                    names=list(THEMES.keys())
                    idx=names.index(CURRENT_THEME)
                    CURRENT_THEME=names[(idx+1)%len(names)]
                    C=THEMES[CURRENT_THEME]
                    road_surf=build_road_surface()

        if road_dirty[0]:
            road_surf=build_road_surface()
            road_dirty[0]=False

        screen.fill(C["bg"])
        if show_charts:
            chart_top=34
            screen.blit(build_chart(SD,VIEW_W,HEIGHT-chart_top),(0,chart_top))
            header=pygame.Surface((VIEW_W,34),pygame.SRCALPHA)
            pygame.draw.rect(header,(0,0,0,120),(0,0,VIEW_W,34))
            screen.blit(header,(0,0))
            lb=font.render("TRAFFIC ANALYTICS DASHBOARD  •  Press [C] to return",True,(230,235,245))
            screen.blit(lb,(18,8))
        else:
            screen.blit(road_surf,(0,0))
            draw_lights(screen,SD,fxs)
            draw_pedestrians(screen,SD)      # ← draw peds BEFORE vehicles (under cars)
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