"""
CS 324 — Modeling and Simulation
Traffic Light System Simulation
Batangas State University — CICS

LANE LAYOUT (drive on the right, 2 inbound + 2 outbound per arm)
Each road arm cross-section from centre outward:

  [DIV] | IB-inner | IB-outer | KERB | OB-outer | OB-inner | [DIV]

Wait — that's wrong. Drive-on-right means:

  Centre divider
  ────────────────────────────────────
  IB-inner   IB-outer   (cars coming IN toward intersection)
  ────────────────────────────────────
  OB-inner   OB-outer   (cars going OUT away from intersection)
  ────────────────────────────────────

For N-S road (vertical):
  East of CX = INBOUND   (cars travelling south, from north)
  West of CX = OUTBOUND  (cars travelling north, leaving south)

For E-W road (horizontal):
  South of CY = INBOUND  (cars travelling west, from east)
  North of CY = OUTBOUND (cars travelling east, leaving west)

Each half has 2 lanes: inner (closer to divider) and outer (closer to kerb).
"""

import simpy, pygame, random, math, sys, threading, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd

# ════════════════════════════════════════════════════════
#  WINDOW
# ════════════════════════════════════════════════════════
WIDTH, HEIGHT = 1280, 780
PANEL_W       = 218
VIEW_W        = WIDTH - PANEL_W - 8
CX            = VIEW_W // 2
CY            = HEIGHT  // 2

# ════════════════════════════════════════════════════════
#  ROAD GEOMETRY
#  2 inbound lanes + 2 outbound lanes per arm
#  Each lane is LANE_W px wide
#  Centre divider is DIV_W px wide
# ════════════════════════════════════════════════════════
LANE_W  = 32        # one lane width (px)
DIV_W   = 6         # yellow centre divider (px)
N_IB    = 2         # inbound lanes per arm
N_OB    = 2         # outbound lanes per arm

# Half-road width from centre divider edge:
#   N_IB lanes (inbound side)  OR  N_OB lanes (outbound side)
HALF_IB = N_IB * LANE_W          # 64 px  — inbound half
HALF_OB = N_OB * LANE_W          # 64 px  — outbound half
DIV_HALF = DIV_W // 2            # 3 px

# Total half-road (from road centreline to kerb):
HR = DIV_HALF + HALF_IB          # = 3 + 64 = 67 px
# (same for outbound side since N_IB == N_OB)

# Inbound lane centres (offset from road centreline, on the INBOUND side)
# Lane 0 = innermost (next to divider), Lane 1 = outermost (next to kerb)
IB_OFF = [DIV_HALF + LANE_W//2 + i*LANE_W for i in range(N_IB)]
# = [3+16, 3+16+32] = [19, 51]

# Outbound lane centres (same offsets, but on the OTHER side of the divider)
OB_OFF = IB_OFF[:]   # mirror: OB_OFF[0] = innermost outbound, etc.

# ════════════════════════════════════════════════════════
#  PHYSICS / QUEUE
# ════════════════════════════════════════════════════════
STOP_DIST  = 10       # px from box edge to stop line
CAR_LEN    = 26
CAR_W      = 14
CAR_GAP    = 6
SLOT_SZ    = CAR_LEN + CAR_GAP   # 32 px

MIN_GAP    = 0.07     # min progress separation between moving cars
BASE_SPD   = 0.003    # progress per mover tick (tuned for visual speed)

# ════════════════════════════════════════════════════════
#  TIMING
#  The SimPy loop sleeps REAL_TICK_S seconds per sim-tick.
#  speed_factor=1.0 → 1 sim-second takes 1 real-second.
#  SIM_TICK = 0.05 sim-sec → sleep 0.05 real-sec at 1x speed.
# ════════════════════════════════════════════════════════
SIM_TICK    = 0.05    # sim-seconds per mover tick
REAL_SEC    = 0.05    # real seconds per sim-second at speed_factor=1.0

# ════════════════════════════════════════════════════════
#  COLOURS
# ════════════════════════════════════════════════════════
C = {
    "bg":           (12,  14,  20),
    "grass":        (24,  40,  24),
    "road":         (36,  38,  47),
    "road_box":     (28,  30,  39),
    "divider":      (200, 175,  30),
    "lane_dash":    (85,  89,  70),
    "kerb":         (50,  53,  63),
    "stop_line":    (220, 220, 220),
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
        (80,  162, 255), (255, 102,  65), (100, 208, 125),
        (255, 198,  60), (168, 100, 255), (255, 145,  80),
        (60,  208, 208), (188, 188, 198), (255, 120, 162),
        (120, 255, 190), (255, 170,  95), (95,  195, 255),
    ],
}

# ════════════════════════════════════════════════════════
#  SCENARIOS
# ════════════════════════════════════════════════════════
SCENARIOS = {
    "Normal Traffic": {"green": 30, "yellow": 5, "red": 35, "arrival": 3.5},
    "Rush Hour":      {"green": 45, "yellow": 5, "red": 50, "arrival": 1.2},
    "Low Traffic":    {"green": 18, "yellow": 3, "red": 21, "arrival": 9.0},
}

#  from_dir:  0 = from North  (car travels South)
#             1 = from East   (car travels West)
#             2 = from South  (car travels North)
#             3 = from West   (car travels East)
DIR_NAMES  = ["North", "East", "South", "West"]

# Turns: right = 90° CW,  straight = same direction,  left = 90° CCW
# NO U-turns (that would be (from_dir+2)%4 and is never generated)
TURNS      = ["right", "straight", "left"]
TURN_PROBS = [0.25,    0.50,       0.25]


# ════════════════════════════════════════════════════════
#  LANE COORDINATE HELPERS
#
#  INBOUND traffic for a given from_dir occupies the "right" half
#  of that road arm (drive on the right):
#
#   from_dir=0 (→South): inbound x = CX + IB_OFF[lane]   (east side)
#   from_dir=2 (→North): inbound x = CX - IB_OFF[lane]   (west side)
#   from_dir=1 (→West):  inbound y = CY + IB_OFF[lane]   (south side)
#   from_dir=3 (→East):  inbound y = CY - IB_OFF[lane]   (north side)
#
#  OUTBOUND traffic (exiting vehicles) uses the OTHER half:
#   Vehicles exiting toward North travel on west side → CX - OB_OFF[lane]
#   Vehicles exiting toward South travel on east side → CX + OB_OFF[lane]
#   Vehicles exiting toward East  travel on north side→ CY - OB_OFF[lane]
#   Vehicles exiting toward West  travel on south side→ CY + OB_OFF[lane]
# ════════════════════════════════════════════════════════

def ib_x(from_dir, lane):
    """Inbound lane x-coord (for NS roads) or y-coord (for EW roads)."""
    off = IB_OFF[lane]
    if from_dir == 0: return CX + off   # southbound → east side
    if from_dir == 2: return CX - off   # northbound → west side
    if from_dir == 1: return CY + off   # westbound  → south side
    return             CY - off          # eastbound  → north side


def ob_coord(exit_toward, lane):
    """
    Outbound lane coord for a vehicle EXITING toward exit_toward.
    exit_toward=0 → heading North → west side → CX - OB_OFF[lane]
    exit_toward=2 → heading South → east side → CX + OB_OFF[lane]
    exit_toward=1 → heading East  → north side→ CY - OB_OFF[lane]
    exit_toward=3 → heading West  → south side→ CY + OB_OFF[lane]
    """
    off = OB_OFF[lane]
    if exit_toward == 0: return CX - off   # northbound outbound: west side
    if exit_toward == 2: return CX + off   # southbound outbound: east side
    if exit_toward == 1: return CY - off   # eastbound  outbound: north side
    return               CY + off           # westbound  outbound: south side


# ════════════════════════════════════════════════════════
#  PATH BUILDER  —  no U-turns, correct lane geometry
# ════════════════════════════════════════════════════════
def _bezier(p0, p1, p2, n=16):
    return [
        (
            (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0],
            (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1],
        )
        for t in (i/n for i in range(n+1))
    ]


def build_path(from_dir: int, turn: str, lane: int) -> list[tuple]:
    """
    Build full waypoint list for one vehicle.

    from_dir : approach direction (0-3)
    turn     : 'right' | 'straight' | 'left'
    lane     : 0=inner inbound lane, 1=outer inbound lane

    Exit lane is always the INNER outbound lane (lane 0) of the
    departure road arm — vehicles merge into the closest lane.
    """
    off  = IB_OFF[lane]
    sl   = HR + STOP_DIST       # stop-line distance from CX/CY

    # ── Approach points ──
    if from_dir == 0:    # southbound (from North), east side
        ix = CX + off
        spawn   = (ix, -50)
        stop_pt = (ix, CY - sl)
        box_in  = (ix, CY - HR + 2)
    elif from_dir == 2:  # northbound (from South), west side
        ix = CX - off
        spawn   = (ix, HEIGHT + 50)
        stop_pt = (ix, CY + sl)
        box_in  = (ix, CY + HR - 2)
    elif from_dir == 1:  # westbound (from East), south side
        iy = CY + off
        spawn   = (VIEW_W + 50, iy)
        stop_pt = (CX + sl, iy)
        box_in  = (CX + HR - 2, iy)
    else:                # eastbound (from West), north side
        iy = CY - off
        spawn   = (-50, iy)
        stop_pt = (CX - sl, iy)
        box_in  = (CX - HR + 2, iy)

    # ── Exit direction (no U-turn) ──
    exit_dir = (from_dir + {"right": 1, "straight": 0, "left": 3}[turn]) % 4

    # ── Exit/departure points — always inner outbound lane (lane=0) ──
    ex_off = OB_OFF[0]
    if exit_dir == 0:    # heading North → west side outbound
        ex = CX - ex_off
        box_out = (ex, CY - HR + 2)
        depart  = (ex, -50)
    elif exit_dir == 2:  # heading South → east side outbound
        ex = CX + ex_off
        box_out = (ex, CY + HR - 2)
        depart  = (ex, HEIGHT + 50)
    elif exit_dir == 1:  # heading East → north side outbound
        ey = CY - ex_off
        box_out = (CX + HR - 2, ey)
        depart  = (VIEW_W + 50, ey)
    else:                # heading West → south side outbound
        ey = CY + ex_off
        box_out = (CX - HR + 2, ey)
        depart  = (-50, ey)

    # ── Straight: no curve needed ──
    if turn == "straight":
        return [spawn, stop_pt, box_in, box_out, depart]

    # ── Turn: quadratic Bezier through intersection box ──
    bix, biy   = box_in
    box_x, box_y = box_out
    # Control point = geometric corner of the turn
    if from_dir in (0, 2):   # travelling vertically → turning horizontal
        cp = (bix, box_y)
    else:                     # travelling horizontally → turning vertical
        cp = (box_x, biy)

    n = 10 if turn == "right" else 18
    curve = _bezier(box_in, cp, box_out, n=n)
    return [spawn, stop_pt] + curve + [depart]


# ════════════════════════════════════════════════════════
#  PATH INTERPOLATION
# ════════════════════════════════════════════════════════
def interp_path(path, t):
    if t <= 0: return path[0]
    if t >= 1: return path[-1]
    segs, total = [], 0.0
    for i in range(len(path)-1):
        l = math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        segs.append(l); total += l
    if total == 0: return path[0]
    target, acc = t * total, 0.0
    for i, sl in enumerate(segs):
        if acc + sl >= target or i == len(segs)-1:
            lt = (target - acc) / sl if sl > 0 else 0
            return (path[i][0] + lt*(path[i+1][0]-path[i][0]),
                    path[i][1] + lt*(path[i+1][1]-path[i][1]))
        acc += sl
    return path[-1]


def path_heading(path, t):
    p0 = interp_path(path, max(0.0, t - 0.02))
    p1 = interp_path(path, min(1.0, t + 0.02))
    return math.degrees(math.atan2(p1[1]-p0[1], p1[0]-p0[0]))


# ════════════════════════════════════════════════════════
#  SIMULATION STATE
# ════════════════════════════════════════════════════════
class SimData:
    def __init__(self):
        self.lock           = threading.Lock()
        self.vehicles       : list[dict] = []
        self.completed      : list[dict] = []
        self.light_ns       = "green"
        self.light_ew       = "red"
        self.light_timer    = 0
        self.sim_time       = 0.0
        self.total_vehicles = 0
        self.scenario       = "Normal Traffic"
        self.running        = True
        self.speed_factor   = 1.0    # 1.0 = real-time
        self.wait_times     : list[float] = []
        self.throughput_log : list[tuple] = []

SD = SimData()


# ════════════════════════════════════════════════════════
#  SIMPY ENGINE
# ════════════════════════════════════════════════════════
def run_simulation(sd: SimData):
    env = simpy.Environment()

    # ── Traffic light controller ──
    def light_ctrl(env):
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            G, Y, R = cfg["green"], cfg["yellow"], cfg["red"]
            for ns, ew, dur in [
                ("green",  "red",    G),
                ("yellow", "red",    Y),
                ("red",    "green",  R),
                ("red",    "yellow", Y),
            ]:
                with sd.lock:
                    sd.light_ns    = ns
                    sd.light_ew    = ew
                    sd.light_timer = dur
                for t in range(dur):
                    if not sd.running: return
                    yield env.timeout(1.0)          # 1 sim-second per tick
                    with sd.lock:
                        sd.light_timer = dur - t - 1
                        sd.sim_time    = env.now

    # ── Vehicle spawner ──
    def gen_vehicles(env, from_dir):
        vid = 0
        while sd.running:
            cfg  = SCENARIOS[sd.scenario]
            yield env.timeout(random.expovariate(1.0 / cfg["arrival"]))
            vid  += 1
            turn  = random.choices(TURNS, TURN_PROBS)[0]
            lane  = random.randint(0, N_IB - 1)
            path  = build_path(from_dir, turn, lane)
            with sd.lock:
                sd.total_vehicles += 1
                sd.vehicles.append({
                    "id":         f"{DIR_NAMES[from_dir][0]}{vid}",
                    "from_dir":   from_dir,
                    "lane":       lane,
                    "turn":       turn,
                    "path":       path,
                    "state":      "queued",
                    "progress":   0.0,
                    "arrive":     env.now,
                    "depart":     None,
                    "wait":       0.0,
                    "queue_slot": 0,
                    "color":      random.choice(C["car_colors"]),
                })

    # ── Vehicle mover (runs every SIM_TICK sim-seconds) ──
    def mover(env):
        while sd.running:
            yield env.timeout(SIM_TICK)
            with sd.lock:
                l_ns = sd.light_ns
                l_ew = sd.light_ew
                now  = env.now

                # Per-direction AND per-lane queued lists (arrival order)
                # Key: (from_dir, lane)
                q_only: dict[tuple, list] = {}
                for v in sd.vehicles:
                    if v["state"] == "queued":
                        key = (v["from_dir"], v["lane"])
                        q_only.setdefault(key, []).append(v)
                for q in q_only.values():
                    q.sort(key=lambda v: v["arrive"])

                # Assign queue_slot per lane independently
                for q in q_only.values():
                    for slot, v in enumerate(q):
                        v["queue_slot"] = slot

                # Progress map per direction for gap-keeping (moving vehicles)
                prog: dict[int, list] = {0:[], 1:[], 2:[], 3:[]}
                for v in sd.vehicles:
                    if v["state"] == "moving":
                        prog[v["from_dir"]].append(v["progress"])
                for lst in prog.values():
                    lst.sort()

                remove = []
                for v in sd.vehicles:
                    d     = v["from_dir"]
                    light = l_ns if d in (0, 2) else l_ew
                    key   = (d, v["lane"])
                    qd    = q_only.get(key, [])

                    if v["state"] == "queued":
                        v["wait"] = now - v["arrive"]
                        if light == "green":
                            pos = qd.index(v) if v in qd else -1
                            if pos == 0:
                                # First in this lane — release if road ahead clear
                                if not prog[d] or min(prog[d]) > MIN_GAP:
                                    v["state"]    = "moving"
                                    v["progress"] = 0.001
                                    prog[d].append(0.001)
                                    prog[d].sort()
                                    qd.remove(v)
                            elif pos > 0:
                                leader = qd[pos - 1]
                                if (leader["state"] == "moving"
                                        and leader["progress"] > MIN_GAP * (pos + 1)):
                                    v["state"]    = "moving"
                                    v["progress"] = 0.001
                                    prog[d].append(0.001)
                                    prog[d].sort()
                                    qd.remove(v)

                    elif v["state"] == "moving":
                        ahead = [p for p in prog[d] if p > v["progress"] + 0.001]
                        if ahead:
                            gap   = min(ahead) - v["progress"]
                            speed = (BASE_SPD * max(0.0, (gap / MIN_GAP) - 0.05)
                                     if gap < MIN_GAP else BASE_SPD)
                        else:
                            speed = BASE_SPD

                        old_p         = v["progress"]
                        v["progress"] = min(1.0, old_p + speed)
                        try:    prog[d].remove(old_p)
                        except: pass
                        prog[d].append(v["progress"])
                        prog[d].sort()

                        if v["progress"] >= 1.0:
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
    for d in range(4):
        env.process(gen_vehicles(env, d))

    # ── Paced SimPy loop ──
    # Run all events up to the next mover tick, then sleep the correct
    # real-world duration so sim-time matches real-time at speed_factor=1.
    real_per_simtick = SIM_TICK * REAL_SEC  # real seconds per mover tick at 1x
    while sd.running:
        # Advance sim by one full SIM_TICK worth of events
        target = env.now + SIM_TICK
        while env.peek() <= target:
            env.step()
        # Sleep to pace real time
        sleep_t = real_per_simtick / sd.speed_factor
        time.sleep(sleep_t)


# ════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════════════
def rrect(surf, color, rect, r=8, alpha=None):
    if alpha is not None:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)


# ════════════════════════════════════════════════════════
#  STATIC ROAD SURFACE
# ════════════════════════════════════════════════════════
def build_road_surface() -> pygame.Surface:
    surf = pygame.Surface((VIEW_W, HEIGHT))
    cx, cy = CX, CY
    surf.fill(C["grass"])

    # ── Road base ──
    pygame.draw.rect(surf, C["road"],     (cx-HR, 0,     HR*2,   HEIGHT))
    pygame.draw.rect(surf, C["road"],     (0,     cy-HR, VIEW_W, HR*2))
    pygame.draw.rect(surf, C["road_box"], (cx-HR, cy-HR, HR*2,   HR*2))

    dh = DIV_HALF

    # ── Yellow solid centre divider (outside box only) ──
    pygame.draw.rect(surf, C["divider"], (cx-dh, 0,      DIV_W, cy-HR))
    pygame.draw.rect(surf, C["divider"], (cx-dh, cy+HR,  DIV_W, HEIGHT))
    pygame.draw.rect(surf, C["divider"], (0,      cy-dh, cx-HR, DIV_W))
    pygame.draw.rect(surf, C["divider"], (cx+HR,  cy-dh, VIEW_W,DIV_W))

    dk = C["lane_dash"]
    dl, dg = 22, 14

    # ── Dashed white lane separators ──
    # Vertical N-S road: IB = east side (cx+dh..cx+HR), OB = west side
    for i in range(1, N_IB):
        lx_ib = cx + dh + i * LANE_W   # IB lane separator (east)
        lx_ob = cx - dh - i * LANE_W   # OB lane separator (west)
        for y_start, y_end in [(0, cy-HR), (cy+HR, HEIGHT)]:
            for y in range(y_start, y_end, dl+dg):
                pygame.draw.rect(surf, dk, (lx_ib-1, y, 2, min(dl, y_end-y)))
                pygame.draw.rect(surf, dk, (lx_ob-1, y, 2, min(dl, y_end-y)))

    # Horizontal E-W road: IB = south side (cy+dh..cy+HR), OB = north side
    for i in range(1, N_OB):
        ly_ib = cy + dh + i * LANE_W
        ly_ob = cy - dh - i * LANE_W
        for x_start, x_end in [(0, cx-HR), (cx+HR, VIEW_W)]:
            for x in range(x_start, x_end, dl+dg):
                pygame.draw.rect(surf, dk, (x, ly_ib-1, min(dl, x_end-x), 2))
                pygame.draw.rect(surf, dk, (x, ly_ob-1, min(dl, x_end-x), 2))

    # ── Kerb edge lines ──
    kc = (55, 58, 68)
    pygame.draw.line(surf, kc, (cx-HR, 0),     (cx-HR, HEIGHT),  2)
    pygame.draw.line(surf, kc, (cx+HR, 0),     (cx+HR, HEIGHT),  2)
    pygame.draw.line(surf, kc, (0,     cy-HR), (VIEW_W, cy-HR),  2)
    pygame.draw.line(surf, kc, (0,     cy+HR), (VIEW_W, cy+HR),  2)

    # ── Stop lines — span inbound half of each approach ──
    wl = C["stop_line"]
    st = 3
    soff = HR + STOP_DIST
    pygame.draw.rect(surf, wl, (cx+dh,       cy-soff-st, HALF_IB, st))  # from-N
    pygame.draw.rect(surf, wl, (cx-HR,       cy+soff,    HALF_IB, st))  # from-S
    pygame.draw.rect(surf, wl, (cx+soff,     cy+dh,      st, HALF_IB))  # from-E
    pygame.draw.rect(surf, wl, (cx-soff-st,  cy-HR,      st, HALF_IB))  # from-W

    _draw_arrows(surf, cx, cy)
    _draw_crosswalks(surf, cx, cy)
    return surf


def _draw_crosswalks(surf, cx, cy):
    n, sh, sg = 5, 5, 4
    total = n * (sh + sg)
    ofs   = 5
    for i in range(n):
        s = pygame.Surface((HR*2, sh), pygame.SRCALPHA)
        pygame.draw.rect(s, (218, 220, 228, 80), (0, 0, HR*2, sh))
        surf.blit(s, (cx-HR, cy - HR - ofs - total + i*(sh+sg)))
        surf.blit(s, (cx-HR, cy + HR + ofs + i*(sh+sg)))
        s2 = pygame.Surface((sh, HR*2), pygame.SRCALPHA)
        pygame.draw.rect(s2, (218, 220, 228, 80), (0, 0, sh, HR*2))
        surf.blit(s2, (cx + HR + ofs + i*(sh+sg),           cy - HR))
        surf.blit(s2, (cx - HR - ofs - total + i*(sh+sg),   cy - HR))


def _draw_arrows(surf, cx, cy):
    col  = (75, 80, 96)
    sz   = 8
    dist = 55

    def arrow(x, y, deg):
        r   = math.radians(deg)
        tip = (x + sz*math.cos(r),         y + sz*math.sin(r))
        l   = (x + sz*.5*math.cos(r+2.3),  y + sz*.5*math.sin(r+2.3))
        ri  = (x + sz*.5*math.cos(r-2.3),  y + sz*.5*math.sin(r-2.3))
        pygame.draw.polygon(surf, col, [tip, l, ri])

    for lane in range(N_IB):
        off = IB_OFF[lane]
        arrow(cx + off, cy - HR - dist,  90)   # from-N → south
        arrow(cx - off, cy + HR + dist, 270)   # from-S → north
        arrow(cx + HR + dist, cy + off, 180)   # from-E → west
        arrow(cx - HR - dist, cy - off,   0)   # from-W → east


# ════════════════════════════════════════════════════════
#  TRAFFIC LIGHTS
# ════════════════════════════════════════════════════════
def draw_lights(surf, sd: SimData, fxs):
    cx, cy = CX, CY
    with sd.lock:
        l_ns = sd.light_ns; l_ew = sd.light_ew; timer = sd.light_timer

    def pole(px, py, state):
        pygame.draw.rect(surf, (46, 50, 65), (px-2, py, 4, 24), border_radius=2)
        hw, hh = 18, 50
        hx, hy = px - hw//2, py - hh
        rrect(surf, (16, 19, 29), (hx, hy, hw, hh), r=5)
        pygame.draw.rect(surf, (38, 43, 60), (hx, hy, hw, hh), 1, border_radius=5)
        lmap = {
            "red":    [C["red_light"],    (26,26,26),        (26,26,26)],
            "yellow": [(26,26,26),         C["yellow_light"], (26,26,26)],
            "green":  [(26,26,26),         (26,26,26),        C["green_light"]],
        }
        for i, col in enumerate(lmap.get(state, [(26,26,26)]*3)):
            lcy = hy + 10 + i*14
            if col != (26,26,26):
                g = pygame.Surface((20,20), pygame.SRCALPHA)
                pygame.draw.circle(g, (*col, 55), (10,10), 10)
                surf.blit(g, (px-10, lcy-10))
            pygame.draw.circle(surf, col, (px, lcy), 6)

    pole(cx - HR - 22, cy - HR - 54, l_ns)
    pole(cx + HR +  8, cy + HR +  4, l_ns)
    pole(cx + HR +  8, cy - HR - 54, l_ew)
    pole(cx - HR - 22, cy + HR +  4, l_ew)

    tb = pygame.Surface((46, 18), pygame.SRCALPHA)
    pygame.draw.rect(tb, (0,0,0,165), (0,0,46,18), border_radius=4)
    surf.blit(tb, (cx-23, cy-9))
    ts = fxs.render(f"{int(timer):02d}s", True, C["text"])
    surf.blit(ts, (cx - ts.get_width()//2, cy-8))


# ════════════════════════════════════════════════════════
#  VEHICLE RENDERER
# ════════════════════════════════════════════════════════
def draw_vehicles(surf, sd: SimData, fxs):
    with sd.lock:
        vehs = list(sd.vehicles)
    for v in vehs:
        if v["state"] == "queued":  _draw_vehicle(surf, v, fxs)
    for v in vehs:
        if v["state"] == "moving":  _draw_vehicle(surf, v, fxs)


def _draw_vehicle(surf, v, fxs):
    d     = v["from_dir"]
    state = v["state"]
    path  = v["path"]

    if state == "queued":
        sx, sy = path[1]   # stop-line point
        slot   = v.get("queue_slot", 0)
        if   d == 0: sy -= slot * SLOT_SZ
        elif d == 2: sy += slot * SLOT_SZ
        elif d == 1: sx += slot * SLOT_SZ
        else:        sx -= slot * SLOT_SZ
        x, y  = int(sx), int(sy)
        angle = {0: 90, 1: 180, 2: 270, 3: 0}[d]
    else:
        p     = v["progress"]
        pt    = interp_path(path, p)
        x, y  = int(pt[0]), int(pt[1])
        angle = path_heading(path, p)

    if x < -120 or x > VIEW_W+120 or y < -120 or y > HEIGHT+120:
        return

    _render_car(surf, x, y, angle, v["color"], state, v.get("wait", 0.0), fxs)


def _render_car(surf, x, y, angle_deg, col, state, wait, fxs):
    cw, ch = CAR_W, CAR_LEN
    body   = pygame.Surface((cw, ch), pygame.SRCALPHA)

    pygame.draw.rect(body, col, (0, 2, cw, ch-4), border_radius=4)
    roof = tuple(min(255, c+45) for c in col)
    pygame.draw.rect(body, roof, (2, 4, cw-4, ch-14), border_radius=3)
    pygame.draw.rect(body, (98, 160, 210, 175), (2, ch-17, cw-4, 9), border_radius=2)
    # Headlights (front = bottom of surface)
    pygame.draw.circle(body, (255, 248, 182), (3,    ch-3), 2)
    pygame.draw.circle(body, (255, 248, 182), (cw-3, ch-3), 2)
    # Tail/brake lights (rear = top of surface)
    if state == "queued":
        bl = (255, 34, 34)
        pygame.draw.circle(body, bl, (3,    3), 2)
        pygame.draw.circle(body, bl, (cw-3, 3), 2)
        g = pygame.Surface((cw, 10), pygame.SRCALPHA)
        pygame.draw.rect(g, (255,34,34,45), (0,0,cw,10))
        body.blit(g, (0,0))
    else:
        pygame.draw.circle(body, (138, 18, 18), (3,    3), 2)
        pygame.draw.circle(body, (138, 18, 18), (cw-3, 3), 2)

    rotated = pygame.transform.rotate(body, -(angle_deg - 90))
    rr = rotated.get_rect(center=(x, y))
    surf.blit(rotated, rr.topleft)

    if state == "queued" and wait > 2.0:
        bw, bh = 28, 13
        badge  = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(badge, (185, 24, 24, 210), (0,0,bw,bh), border_radius=3)
        t = fxs.render(f"{int(wait)}s", True, (255,255,255))
        badge.blit(t, (bw//2-t.get_width()//2, bh//2-t.get_height()//2))
        surf.blit(badge, (x-bw//2, y-22))


# ════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════
_chart_cache  = None
_chart_last_n = -1

def build_chart(sd, w, h):
    global _chart_cache, _chart_last_n
    with sd.lock:
        waits = list(sd.wait_times)
        log   = list(sd.throughput_log)
        done  = len(sd.completed)
    if done == _chart_last_n and _chart_cache is not None:
        return _chart_cache
    _chart_last_n = done

    fig, axes = plt.subplots(1, 2, figsize=(w/100, h/100), dpi=100)
    fig.patch.set_facecolor("#0c0e14")
    for ax, kind, data, title, xl, yl, col in [
        (axes[0], "hist", waits, "Vehicle Wait Times",    "Wait (s)",     "Count",    "#5090ff"),
        (axes[1], "line", log,   "Cumulative Throughput", "Sim Time (s)", "Vehicles", "#50ffa0"),
    ]:
        ax.set_facecolor("#171926")
        ax.set_title(title, color="#dce1f0", fontsize=9, pad=5)
        ax.set_xlabel(xl,   color="#787e96", fontsize=7)
        ax.set_ylabel(yl,   color="#787e96", fontsize=7)
        ax.tick_params(colors="#787e96", labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#282b40")
        if kind == "hist" and data:
            ax.hist(data, bins=max(6, min(20, len(data)//3+1)),
                    color=col, edgecolor="#080a12", lw=0.5)
        elif kind == "line" and data:
            xs = [l[0] for l in data]; ys = list(range(1, len(data)+1))
            ax.plot(xs, ys, color=col, lw=1.3)
            ax.fill_between(xs, ys, alpha=0.10, color=col)

    fig.tight_layout(pad=1.5)
    canvas = agg.FigureCanvasAgg(fig); canvas.draw()
    surf = pygame.image.frombuffer(canvas.buffer_rgba(), canvas.get_width_height(), "RGBA")
    plt.close(fig)
    _chart_cache = surf.copy()
    return _chart_cache


# ════════════════════════════════════════════════════════
#  STATS PANEL
# ════════════════════════════════════════════════════════
def draw_panel(surf, sd, fonts, px, py, pw, ph):
    font, fsm, fxs = fonts
    rrect(surf, C["panel"], (px,py,pw,ph), r=10)
    pygame.draw.rect(surf, C["border"], (px,py,pw,ph), 1, border_radius=10)
    y = py + 14

    def ctr(txt, fy, fnt, col):
        s = fnt.render(txt, True, col)
        surf.blit(s, (px+pw//2 - s.get_width()//2, fy))
        return fy + s.get_height() + 4

    def div(fy):
        pygame.draw.line(surf, C["border"], (px+10,fy),(px+pw-10,fy))
        return fy + 9

    def row(lbl, val, fy, vc=None):
        ls = fxs.render(lbl, True, C["text_dim"])
        vs = fxs.render(str(val), True, vc or C["text"])
        surf.blit(ls, (px+12, fy))
        surf.blit(vs, (px+pw-12-vs.get_width(), fy))
        return fy + 17

    y = ctr("TRAFFIC SIM",        y, font, C["accent"])
    y = ctr("CS 324 · BatStateU", y, fxs,  C["text_dim"])
    y = div(y)
    y = ctr(sd.scenario,          y, fsm,  C["accent2"])
    y = div(y)

    with sd.lock:
        l_ns=sd.light_ns; l_ew=sd.light_ew; timer=sd.light_timer
        sim_t=sd.sim_time; tv=sd.total_vehicles
        wait =sum(1 for v in sd.vehicles if v["state"]=="queued")
        move =sum(1 for v in sd.vehicles if v["state"]=="moving")
        done =len(sd.completed); waits=list(sd.wait_times)

    def lbox(lbl, state, bx, by):
        cm = {"green":C["green_light"],"yellow":C["yellow_light"],"red":C["red_light"]}
        col = cm.get(state, C["text_dim"])
        rrect(surf,(24,28,42),(bx,by,88,46),r=6)
        pygame.draw.rect(surf,col,(bx,by,88,46),2,border_radius=6)
        ls = fxs.render(lbl, True, C["text_dim"])
        surf.blit(ls, (bx+44-ls.get_width()//2, by+4))
        dot = pygame.Surface((14,14), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*col,210),(7,7),7)
        surf.blit(dot, (bx+37, by+22))
        ss = fxs.render(state.upper(), True, col)
        surf.blit(ss, (bx+44-ss.get_width()//2, by+27))

    lbox("N-S", l_ns, px+6,     y)
    lbox("E-W", l_ew, px+pw-94, y)
    y += 54
    ts = fxs.render(f"Phase: {int(timer):02d}s", True, C["text"])
    surf.blit(ts, (px+pw//2-ts.get_width()//2, y)); y += 18
    y = div(y)
    y = row("Sim Time",   f"{sim_t:.1f}s", y)
    y = row("Spawned",    tv,              y)
    y = row("Completed",  done,            y, C["green_light"])
    y = row("Waiting",    wait,            y, C["red_light"])
    y = row("Moving",     move,            y, C["yellow_light"])
    y = div(y)
    avg = sum(waits)/len(waits) if waits else 0
    mx  = max(waits)            if waits else 0
    y = row("Avg Wait", f"{avg:.1f}s", y)
    y = row("Max Wait", f"{mx:.1f}s",  y)
    y = div(y)
    sp = fxs.render(f"Speed x{sd.speed_factor:.1f}", True, C["accent"])
    surf.blit(sp, (px+pw//2-sp.get_width()//2, y)); y += 18
    y = div(y)
    for hint in [
        "[1] Normal [2] Rush [3] Low",
        "[UP] Faster  [DOWN] Slower",
        "[C] Charts [R] Reset [Q] Quit",
    ]:
        hs = fxs.render(hint, True, C["text_dim"])
        surf.blit(hs, (px+pw//2-hs.get_width()//2, y)); y += 15


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CS 324 — Traffic Light Simulation | BatStateU CICS")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 15, bold=True)
    fsm  = pygame.font.SysFont("monospace", 12, bold=True)
    fxs  = pygame.font.SysFont("monospace", 10)

    road_surf = build_road_surface()
    threading.Thread(target=run_simulation, args=(SD,), daemon=True).start()

    px, py = WIDTH - PANEL_W - 6, 6
    pw, ph = PANEL_W, HEIGHT - 12
    show_charts = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_q: running = False
                elif event.key == pygame.K_1: SD.scenario = "Normal Traffic"
                elif event.key == pygame.K_2: SD.scenario = "Rush Hour"
                elif event.key == pygame.K_3: SD.scenario = "Low Traffic"
                elif event.key == pygame.K_UP:
                    SD.speed_factor = min(8.0, round(SD.speed_factor + 0.5, 1))
                elif event.key == pygame.K_DOWN:
                    SD.speed_factor = max(0.5, round(SD.speed_factor - 0.5, 1))
                elif event.key == pygame.K_c:
                    show_charts = not show_charts
                elif event.key == pygame.K_r:
                    with SD.lock:
                        SD.vehicles.clear(); SD.completed.clear()
                        SD.wait_times.clear(); SD.throughput_log.clear()
                        SD.total_vehicles = 0

        screen.fill(C["bg"])
        if show_charts:
            screen.blit(build_chart(SD, VIEW_W, HEIGHT), (0,0))
            lb = fxs.render("Press [C] to return to simulation", True, C["text_dim"])
            screen.blit(lb, (VIEW_W//2 - lb.get_width()//2, HEIGHT-26))
        else:
            screen.blit(road_surf, (0,0))
            draw_lights(screen, SD, fxs)
            draw_vehicles(screen, SD, fxs)

        draw_panel(screen, SD, (font,fsm,fxs), px, py, pw, ph)
        fps = fxs.render(f"FPS {int(clock.get_fps())}", True, C["text_dim"])
        screen.blit(fps, (6, 6))
        pygame.display.flip()
        clock.tick(60)

    SD.running = False
    pygame.quit()
    if SD.completed:
        df = pd.DataFrame([
            {k: v for k,v in c.items() if k != "path"}
            for c in SD.completed
        ])
        df.to_csv("simulation_results.csv", index=False)
        print(f"Results saved: simulation_results.csv ({len(SD.completed)} vehicles)")
    sys.exit(0)


if __name__ == "__main__":
    main()