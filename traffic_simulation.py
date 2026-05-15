"""
CS 324 — Modeling and Simulation
Traffic Light System Simulation | Batangas State University — CICS

ROAD LAYOUT (drive on right, 3 inbound + 3 outbound per arm)

  N-S road cross-section (looking north):
  ┌──────────────────────────────────────────────────────┐
  │ OB-L OB-M OB-R │ DIV │ IB-L IB-M IB-R              │
  │ ←←←  ←←←  ←←← │─────│ →→→  →→→  →→→               │
  │ (northbound)    │     │ (southbound)                  │
  └──────────────────────────────────────────────────────┘

  IB = Inbound  (approaching intersection)
  OB = Outbound (leaving intersection)
  DIV = Yellow centre divider

For N-S road:  IB east of CX  (CX + offset), OB west of CX  (CX - offset)
For E-W road:  IB south of CY (CY + offset), OB north of CY (CY - offset)

Turns (no U-turns ever):
  from_dir = source direction (0=N,1=E,2=S,3=W)
  travel   = (from_dir+2)%4
  right    = (from_dir+3)%4
  straight = (from_dir+2)%4  [same as travel]
  left     = (from_dir+1)%4
  U-turn   = from_dir        ← NEVER generated
"""

import simpy, pygame, random, math, sys, threading, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd

# ══════════════════════════════════════════════════
#  WINDOW
# ══════════════════════════════════════════════════
WIDTH, HEIGHT = 1280, 780
PANEL_W       = 220
VIEW_W        = WIDTH - PANEL_W - 8
CX            = VIEW_W // 2
CY            = HEIGHT  // 2

# ══════════════════════════════════════════════════
#  ROAD GEOMETRY — 3 lanes each direction per arm
# ══════════════════════════════════════════════════
LANE_W   = 28        # px per lane (narrower to fit 3+3)
N_LANES  = 3         # lanes per direction
DIV_W    = 8         # yellow centre divider
DIV_HALF = DIV_W // 2   # 4

# Lane centre offsets from road centre (positive = inbound side)
# Lane 0 = innermost (next to divider), Lane N_LANES-1 = outermost (kerb)
LANE_OFFS = [DIV_HALF + LANE_W // 2 + i * LANE_W for i in range(N_LANES)]
# = [4+14, 4+14+28, 4+14+56] = [18, 46, 74]

# Half-road width (centre line → outer kerb)
HR = DIV_HALF + N_LANES * LANE_W   # = 4 + 84 = 88

# ══════════════════════════════════════════════════
#  VEHICLE / QUEUE CONSTANTS
# ══════════════════════════════════════════════════
CAR_LEN    = 22      # px
CAR_W      = 12      # px
CAR_GAP    = 6       # px gap between queued cars
SLOT       = CAR_LEN + CAR_GAP   # 28 px per queue slot
STOP_DIST  = 14      # px from box edge to stop line

# Speed: px per sim-frame at 1x speed
# Path ~= 600 px; at 1.0 px/frame & 60fps → ~10 real-seconds to cross
CAR_SPEED  = 1.0

# ══════════════════════════════════════════════════
#  TIMING
# ══════════════════════════════════════════════════
SIM_FPS  = 60.0
FRAME_T  = 1.0 / SIM_FPS   # sim-seconds per mover tick

# ══════════════════════════════════════════════════
#  SCENARIOS
# ══════════════════════════════════════════════════
SCENARIOS = {
    "Normal Traffic": {"green": 30, "yellow": 4, "clear": 3, "red": 30, "arrival": 3.5},
    "Rush Hour":      {"green": 40, "yellow": 4, "clear": 3, "red": 40, "arrival": 1.5},
    "Low Traffic":    {"green": 20, "yellow": 3, "clear": 3, "red": 20, "arrival": 8.0},
}

# ══════════════════════════════════════════════════
#  DIRECTIONS
# ══════════════════════════════════════════════════
DIR_NAMES  = ["North", "East", "South", "West"]
TURNS      = ["right", "straight", "left"]
TURN_PROBS = [0.25,    0.50,       0.25]

# ══════════════════════════════════════════════════
#  COLOURS
# ══════════════════════════════════════════════════
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
        (80, 162, 255), (255, 102,  65), (100, 208, 125),
        (255,198,  60), (168, 100, 255), (255, 145,  80),
        (60, 208, 208), (192, 192, 202), (255, 120, 162),
        (120,255, 192), (255, 170,  95), (95,  195, 255),
    ],
}

# ══════════════════════════════════════════════════
#  COORDINATE HELPERS
# ══════════════════════════════════════════════════
def ib_coord(from_dir: int, lane: int) -> int:
    """Pixel centre of inbound lane (x for NS, y for EW)."""
    off = LANE_OFFS[lane]
    if from_dir == 0: return CX + off   # southbound → east
    if from_dir == 2: return CX - off   # northbound → west
    if from_dir == 1: return CY + off   # westbound  → south
    return             CY - off          # eastbound  → north

def ob_coord(exit_dir: int, turn: str, lane: int) -> int:
    """Pixel centre of outbound lane for given exit direction."""
    # Straight keeps same lane index; turns always use innermost (lane 0)
    el  = lane if turn == "straight" else 0
    off = LANE_OFFS[el]
    if exit_dir == 0: return CX - off   # heading north → west side
    if exit_dir == 2: return CX + off   # heading south → east side
    if exit_dir == 1: return CY - off   # heading east  → north side
    return             CY + off          # heading west  → south side

def stop_px(from_dir: int) -> int:
    """Pixel coordinate of the stop line for this approach."""
    d = HR + STOP_DIST
    if from_dir == 0: return CY - d
    if from_dir == 2: return CY + d
    if from_dir == 1: return CX + d
    return             CX - d

# ══════════════════════════════════════════════════
#  PATH BUILDER
# ══════════════════════════════════════════════════
def _bezier(p0, p1, p2, n=16):
    return [
        ((1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0],
         (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1])
        for t in (i/n for i in range(n+1))
    ]

def build_path(from_dir: int, turn: str, lane: int) -> list:
    ic  = ib_coord(from_dir, lane)
    stp = stop_px(from_dir)

    # Approach waypoints
    if from_dir == 0:
        spawn, stop_pt, box_in = (ic,-60), (ic,stp), (ic, CY-HR+2)
    elif from_dir == 2:
        spawn, stop_pt, box_in = (ic,HEIGHT+60), (ic,stp), (ic, CY+HR-2)
    elif from_dir == 1:
        spawn, stop_pt, box_in = (VIEW_W+60,ic), (stp,ic), (CX+HR-2, ic)
    else:
        spawn, stop_pt, box_in = (-60,ic), (stp,ic), (CX-HR+2, ic)

    # Exit direction — never equals from_dir (no U-turn)
    exit_dir = {"right":    (from_dir+3)%4,
                "straight": (from_dir+2)%4,
                "left":     (from_dir+1)%4}[turn]

    ec = ob_coord(exit_dir, turn, lane)

    if exit_dir == 0:
        box_out, depart = (ec, CY-HR+2), (ec, -60)
    elif exit_dir == 2:
        box_out, depart = (ec, CY+HR-2), (ec, HEIGHT+60)
    elif exit_dir == 1:
        box_out, depart = (CX+HR-2, ec), (VIEW_W+60, ec)
    else:
        box_out, depart = (CX-HR+2, ec), (-60, ec)

    if turn == "straight":
        # Enforce same axis all the way through
        if from_dir in (0, 2):
            box_out = (ic, box_out[1]); depart = (ic, depart[1])
        else:
            box_out = (box_out[0], ic); depart = (depart[0], ic)
        return [spawn, stop_pt, box_in, box_out, depart]

    bix, biy   = box_in
    bx2, by2   = box_out
    cp = (bix, by2) if from_dir in (0,2) else (bx2, biy)
    n  = 10 if turn == "right" else 18
    return [spawn, stop_pt] + _bezier(box_in, cp, box_out, n=n) + [depart]

# ══════════════════════════════════════════════════
#  PATH UTILITIES
# ══════════════════════════════════════════════════
def path_length(path):
    return sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
               for i in range(len(path)-1))

def path_pos_at_dist(path, dist):
    acc = 0.0
    for i in range(len(path)-1):
        dx = path[i+1][0]-path[i][0]
        dy = path[i+1][1]-path[i][1]
        sl = math.hypot(dx, dy)
        if sl == 0: continue
        if acc + sl >= dist or i == len(path)-2:
            t = max(0.0, min(1.0, (dist-acc)/sl))
            return (path[i][0]+t*dx, path[i][1]+t*dy,
                    math.degrees(math.atan2(dy, dx)))
        acc += sl
    return path[-1][0], path[-1][1], 0.0

def stop_dist_along_path(path):
    p0, p1 = path[0], path[1]
    return math.hypot(p1[0]-p0[0], p1[1]-p0[1])

# ══════════════════════════════════════════════════
#  SIMULATION STATE
# ══════════════════════════════════════════════════
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
        self.speed_factor   = 1.0
        self.wait_times     : list[float] = []
        self.throughput_log : list[tuple] = []

SD = SimData()

# ══════════════════════════════════════════════════
#  SIMPY ENGINE
# ══════════════════════════════════════════════════
def run_simulation(sd: SimData):
    env = simpy.Environment()

    # ── Light controller ──
    def light_ctrl(env):
        while sd.running:
            cfg = SCENARIOS[sd.scenario]
            G, Y, CLR, R = cfg["green"], cfg["yellow"], cfg["clear"], cfg["red"]
            for ns, ew, dur in [
                ("green",  "red",   G),
                ("yellow", "red",   Y),
                ("red",    "red",   CLR),  # all-red clearance
                ("red",    "green", R),
                ("red",    "yellow",Y),
                ("red",    "red",   CLR),
            ]:
                with sd.lock:
                    sd.light_ns = ns; sd.light_ew = ew; sd.light_timer = dur
                for t in range(dur):
                    if not sd.running: return
                    yield env.timeout(1.0)
                    with sd.lock:
                        sd.light_timer = max(0, dur-t-1)
                        sd.sim_time    = env.now

    # ── Spawner ──
    def gen_vehicles(env, from_dir):
        vid = 0
        while sd.running:
            cfg  = SCENARIOS[sd.scenario]
            yield env.timeout(random.expovariate(1.0/cfg["arrival"]))
            vid += 1
            turn = random.choices(TURNS, TURN_PROBS)[0]
            lane = random.randint(0, N_LANES-1)
            path = build_path(from_dir, turn, lane)
            plen = path_length(path)
            s0   = stop_dist_along_path(path)
            with sd.lock:
                sd.total_vehicles += 1
                sd.vehicles.append({
                    "id":         f"{DIR_NAMES[from_dir][0]}{vid}",
                    "from_dir":   from_dir,
                    "lane":       lane,
                    "turn":       turn,
                    "path":       path,
                    "plen":       plen,
                    "stop_d":     s0,
                    "dist":       0.0,
                    "state":      "queued",
                    "arrive":     env.now,
                    "depart":     None,
                    "wait":       0.0,
                    "queue_slot": 0,
                    "color":      random.choice(C["car_colors"]),
                })

    # ── Mover ──
    def mover(env):
        while sd.running:
            yield env.timeout(FRAME_T)
            with sd.lock:
                l_ns = sd.light_ns
                l_ew = sd.light_ew
                now  = env.now

                # ── 1. Build per-(dir,lane) queue lists, assign slots ──
                ql: dict[tuple, list] = {}
                for v in sd.vehicles:
                    if v["state"] == "queued":
                        key = (v["from_dir"], v["lane"])
                        ql.setdefault(key, []).append(v)
                for q in ql.values():
                    q.sort(key=lambda v: v["arrive"])
                    for slot, v in enumerate(q):
                        v["queue_slot"] = slot

                # ── 2. Moving vehicles: per-direction pixel positions ──
                # We track per-(dir,lane) so passing between lanes is
                # never confused. But gap-keeping is direction-wide since
                # once cars enter the box they mix paths.
                moving_px: dict[int, list] = {0:[],1:[],2:[],3:[]}
                for v in sd.vehicles:
                    if v["state"] == "moving":
                        moving_px[v["from_dir"]].append(v["dist"])
                for lst in moving_px.values():
                    lst.sort()

                remove = []

                for v in sd.vehicles:
                    d     = v["from_dir"]
                    green = (l_ns if d in (0,2) else l_ew) == "green"
                    key   = (d, v["lane"])
                    q     = ql.get(key, [])
                    slot  = v.get("queue_slot", 0)

                    if v["state"] == "queued":
                        v["wait"] = now - v["arrive"]

                        if green and slot == 0:
                            # Only release if no moving car is within one slot
                            # ahead of the stop line on this direction
                            too_close = any(
                                dd < v["stop_d"] + SLOT * 2
                                for dd in moving_px[d]
                            )
                            if not too_close:
                                v["state"] = "moving"
                                # Start just past stop line so no visual
                                # overlap with the queued car behind it
                                v["dist"]  = v["stop_d"] + SLOT
                                moving_px[d].append(v["dist"])
                                moving_px[d].sort()
                                q.remove(v)
                                for i, rv in enumerate(q):
                                    rv["queue_slot"] = i

                        elif green and slot > 0:
                            leader = q[slot-1]
                            # Follow leader: release only when leader has
                            # cleared enough space (2 slots ahead of stop)
                            if (leader["state"] == "moving"
                                    and leader["dist"] > v["stop_d"] + SLOT * 2):
                                v["state"] = "moving"
                                v["dist"]  = v["stop_d"] + SLOT
                                moving_px[d].append(v["dist"])
                                moving_px[d].sort()
                                q.remove(v)
                                for i, rv in enumerate(q):
                                    rv["queue_slot"] = i
                        # Red/yellow: do nothing — car stays frozen at slot pos

                    elif v["state"] == "moving":
                        # Gap-following: don't pass the car ahead
                        ahead = [dd for dd in moving_px[d]
                                 if dd > v["dist"] + 1.0]
                        if ahead:
                            gap = min(ahead) - v["dist"]
                            safe = CAR_LEN + CAR_GAP
                            spd  = (CAR_SPEED * max(0.0, (gap/safe) - 0.05)
                                    if gap < safe else CAR_SPEED)
                        else:
                            spd = CAR_SPEED

                        old_d     = v["dist"]
                        v["dist"] = min(v["plen"], old_d + spd)

                        try:    moving_px[d].remove(old_d)
                        except: pass
                        moving_px[d].append(v["dist"])
                        moving_px[d].sort()

                        if v["dist"] >= v["plen"] - 1:
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

    # Paced loop: each FRAME_T of sim = FRAME_T real-seconds at speed_factor=1
    while sd.running:
        target = env.now + FRAME_T
        while env.peek() <= target and sd.running:
            env.step()
        time.sleep(FRAME_T / sd.speed_factor)

# ══════════════════════════════════════════════════
#  QUEUE PIXEL POSITION (renderer only)
# ══════════════════════════════════════════════════
def queue_px(v) -> tuple[float, float, float]:
    """
    Exact pixel position for a queued car, based purely on
    (from_dir, lane, queue_slot). Completely independent of v["dist"].
    Slot 0 sits one SLOT behind the stop line.
    Slot N sits (N+1) SLOTs behind the stop line.
    Cars can stack off-screen; they are simply not drawn if out of bounds.
    """
    d    = v["from_dir"]
    ic   = ib_coord(d, v["lane"])
    stp  = stop_px(d)
    slot = v.get("queue_slot", 0)
    ofs  = (slot + 1) * SLOT   # distance behind stop line

    if d == 0: return float(ic), float(stp - ofs), 90.0
    if d == 2: return float(ic), float(stp + ofs), 270.0
    if d == 1: return float(stp + ofs), float(ic), 180.0
    return      float(stp - ofs), float(ic), 0.0

# ══════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════
def rrect(surf, color, rect, r=8, alpha=None):
    if alpha is not None:
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0,0,rect[2],rect[3]), border_radius=r)
        surf.blit(s, (rect[0], rect[1]))
    else:
        pygame.draw.rect(surf, color, rect, border_radius=r)

# ══════════════════════════════════════════════════
#  STATIC ROAD SURFACE
# ══════════════════════════════════════════════════
def build_road_surface() -> pygame.Surface:
    surf = pygame.Surface((VIEW_W, HEIGHT))
    cx, cy = CX, CY
    surf.fill(C["grass"])

    # Road base
    pygame.draw.rect(surf, C["road"],     (cx-HR, 0,     HR*2,   HEIGHT))
    pygame.draw.rect(surf, C["road"],     (0,     cy-HR, VIEW_W, HR*2))
    pygame.draw.rect(surf, C["road_box"], (cx-HR, cy-HR, HR*2,   HR*2))

    dh = DIV_HALF

    # Yellow centre divider (outside box only)
    pygame.draw.rect(surf, C["divider"], (cx-dh, 0,      DIV_W, cy-HR))
    pygame.draw.rect(surf, C["divider"], (cx-dh, cy+HR,  DIV_W, HEIGHT))
    pygame.draw.rect(surf, C["divider"], (0,      cy-dh, cx-HR, DIV_W))
    pygame.draw.rect(surf, C["divider"], (cx+HR,  cy-dh, VIEW_W,DIV_W))

    # Dashed white lane separators — one line between each adjacent lane pair
    dk = C["lane_dash"]
    dl, dg = 20, 12   # dash length, gap

    for i in range(1, N_LANES):
        # N-S road
        lx_ib = cx + dh + i * LANE_W   # IB (east side)
        lx_ob = cx - dh - i * LANE_W   # OB (west side)
        for y0, y1 in [(0, cy-HR), (cy+HR, HEIGHT)]:
            for y in range(y0, y1, dl+dg):
                w = min(dl, y1-y)
                pygame.draw.rect(surf, dk, (lx_ib-1, y, 2, w))
                pygame.draw.rect(surf, dk, (lx_ob-1, y, 2, w))
        # E-W road
        ly_ib = cy + dh + i * LANE_W   # IB (south side)
        ly_ob = cy - dh - i * LANE_W   # OB (north side)
        for x0, x1 in [(0, cx-HR), (cx+HR, VIEW_W)]:
            for x in range(x0, x1, dl+dg):
                w = min(dl, x1-x)
                pygame.draw.rect(surf, dk, (x, ly_ib-1, w, 2))
                pygame.draw.rect(surf, dk, (x, ly_ob-1, w, 2))

    # Kerb edges
    kc = C["kerb"]
    pygame.draw.line(surf, kc, (cx-HR,0),    (cx-HR,HEIGHT),  2)
    pygame.draw.line(surf, kc, (cx+HR,0),    (cx+HR,HEIGHT),  2)
    pygame.draw.line(surf, kc, (0,cy-HR),    (VIEW_W,cy-HR),  2)
    pygame.draw.line(surf, kc, (0,cy+HR),    (VIEW_W,cy+HR),  2)

    # Stop lines — span only the inbound half per approach
    wl = C["stop_line"]
    st = 3
    ib_span = N_LANES * LANE_W   # total inbound width
    soff    = HR + STOP_DIST
    pygame.draw.rect(surf, wl, (cx+dh,       cy-soff-st, ib_span, st))  # from-N
    pygame.draw.rect(surf, wl, (cx-HR,       cy+soff,    ib_span, st))  # from-S
    pygame.draw.rect(surf, wl, (cx+soff,     cy+dh,      st, ib_span))  # from-E
    pygame.draw.rect(surf, wl, (cx-soff-st,  cy-HR,      st, ib_span))  # from-W

    _draw_arrows(surf, cx, cy)
    _draw_crosswalks(surf, cx, cy)
    return surf

def _draw_arrows(surf, cx, cy):
    col  = (78, 82, 98)
    sz   = 8
    dist = 55

    def arrow(x, y, deg):
        r   = math.radians(deg)
        tip = (x + sz*math.cos(r),        y + sz*math.sin(r))
        l   = (x + sz*.5*math.cos(r+2.3), y + sz*.5*math.sin(r+2.3))
        ri  = (x + sz*.5*math.cos(r-2.3), y + sz*.5*math.sin(r-2.3))
        pygame.draw.polygon(surf, col, [tip, l, ri])

    for lane in range(N_LANES):
        off = LANE_OFFS[lane]
        arrow(cx+off,    cy-HR-dist,  90)   # IB from-N (→S)
        arrow(cx-off,    cy+HR+dist, 270)   # IB from-S (→N)
        arrow(cx+HR+dist, cy+off,   180)   # IB from-E (→W)
        arrow(cx-HR-dist, cy-off,     0)   # IB from-W (→E)

def _draw_crosswalks(surf, cx, cy):
    n, sh, sg = 5, 5, 4
    total = n*(sh+sg)
    ofs   = 5
    for i in range(n):
        s = pygame.Surface((HR*2, sh), pygame.SRCALPHA)
        pygame.draw.rect(s, (218,220,228,75), (0,0,HR*2,sh))
        surf.blit(s, (cx-HR, cy-HR-ofs-total+i*(sh+sg)))
        surf.blit(s, (cx-HR, cy+HR+ofs+i*(sh+sg)))
        s2 = pygame.Surface((sh, HR*2), pygame.SRCALPHA)
        pygame.draw.rect(s2, (218,220,228,75), (0,0,sh,HR*2))
        surf.blit(s2, (cx+HR+ofs+i*(sh+sg),         cy-HR))
        surf.blit(s2, (cx-HR-ofs-total+i*(sh+sg),   cy-HR))

# ══════════════════════════════════════════════════
#  TRAFFIC LIGHTS
# ══════════════════════════════════════════════════
def draw_lights(surf, sd: SimData, fxs):
    cx, cy = CX, CY
    with sd.lock:
        l_ns=sd.light_ns; l_ew=sd.light_ew; timer=sd.light_timer

    def pole(px, py, state):
        pygame.draw.rect(surf, (46,50,65),(px-2,py,4,24),border_radius=2)
        hw, hh = 18, 50
        hx, hy = px-hw//2, py-hh
        rrect(surf,(16,19,29),(hx,hy,hw,hh),r=5)
        pygame.draw.rect(surf,(38,43,60),(hx,hy,hw,hh),1,border_radius=5)
        lm = {"red":   [C["red_light"],  (26,26,26),       (26,26,26)],
              "yellow":[(26,26,26),       C["yellow_light"],(26,26,26)],
              "green": [(26,26,26),       (26,26,26),       C["green_light"]]}
        for i,col in enumerate(lm.get(state,[(26,26,26)]*3)):
            lcy = hy+10+i*14
            if col != (26,26,26):
                g = pygame.Surface((20,20),pygame.SRCALPHA)
                pygame.draw.circle(g,(*col,55),(10,10),10)
                surf.blit(g,(px-10,lcy-10))
            pygame.draw.circle(surf,col,(px,lcy),6)

    pole(cx-HR-22, cy-HR-54, l_ns)
    pole(cx+HR+8,  cy+HR+4,  l_ns)
    pole(cx+HR+8,  cy-HR-54, l_ew)
    pole(cx-HR-22, cy+HR+4,  l_ew)

    tb = pygame.Surface((46,18),pygame.SRCALPHA)
    pygame.draw.rect(tb,(0,0,0,165),(0,0,46,18),border_radius=4)
    surf.blit(tb,(cx-23,cy-9))
    ts = fxs.render(f"{int(timer):02d}s",True,C["text"])
    surf.blit(ts,(cx-ts.get_width()//2,cy-8))

# ══════════════════════════════════════════════════
#  VEHICLE RENDERER
# ══════════════════════════════════════════════════
def draw_vehicles(surf, sd: SimData, fxs):
    with sd.lock:
        vehs = list(sd.vehicles)
    # Draw queued first (back layer), moving on top (front layer)
    for v in vehs:
        if v["state"] == "queued": _draw_veh(surf, v, fxs)
    for v in vehs:
        if v["state"] == "moving": _draw_veh(surf, v, fxs)

def _draw_veh(surf, v, fxs):
    if v["state"] == "queued":
        x, y, angle = queue_px(v)
    else:
        x, y, angle = path_pos_at_dist(v["path"], v["dist"])
    x, y = int(x), int(y)
    # Skip off-screen (queues can extend off-screen — that's intentional)
    if x < -80 or x > VIEW_W+80 or y < -80 or y > HEIGHT+80:
        return
    _render_car(surf, x, y, angle, v["color"], v["state"], v.get("wait",0), fxs)

def _render_car(surf, x, y, angle_deg, col, state, wait, fxs):
    cw, ch = CAR_W, CAR_LEN
    body   = pygame.Surface((cw, ch), pygame.SRCALPHA)
    pygame.draw.rect(body, col, (0,2,cw,ch-4), border_radius=4)
    roof = tuple(min(255,c+45) for c in col)
    pygame.draw.rect(body, roof, (2,4,cw-4,ch-14), border_radius=3)
    pygame.draw.rect(body, (98,160,210,175), (2,ch-15,cw-4,8), border_radius=2)
    # Headlights (front = bottom of surface before rotation)
    pygame.draw.circle(body,(255,248,182),(3,    ch-3),2)
    pygame.draw.circle(body,(255,248,182),(cw-3, ch-3),2)
    # Tail/brake lights (rear = top of surface)
    if state == "queued":
        pygame.draw.circle(body,(255,34,34),(3,    3),2)
        pygame.draw.circle(body,(255,34,34),(cw-3, 3),2)
        g = pygame.Surface((cw,10),pygame.SRCALPHA)
        pygame.draw.rect(g,(255,34,34,45),(0,0,cw,10))
        body.blit(g,(0,0))
    else:
        pygame.draw.circle(body,(138,18,18),(3,    3),2)
        pygame.draw.circle(body,(138,18,18),(cw-3, 3),2)

    rotated = pygame.transform.rotate(body, -(angle_deg-90))
    rr = rotated.get_rect(center=(x, y))
    surf.blit(rotated, rr.topleft)

    # Wait badge: only shown on queued vehicles that have waited > 2s
    # Badge disappears the moment the vehicle starts moving
    if state == "queued" and wait > 2.0:
        ws   = int(wait)
        bw, bh = 28, 13
        badge  = pygame.Surface((bw,bh),pygame.SRCALPHA)
        pygame.draw.rect(badge,(185,24,24,210),(0,0,bw,bh),border_radius=3)
        t = fxs.render(f"{ws}s",True,(255,255,255))
        badge.blit(t,(bw//2-t.get_width()//2,bh//2-t.get_height()//2))
        surf.blit(badge,(x-bw//2,y-20))

# ══════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════
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
    fig, axes = plt.subplots(1,2,figsize=(w/100,h/100),dpi=100)
    fig.patch.set_facecolor("#0c0e14")
    for ax,kind,data,title,xl,yl,col in [
        (axes[0],"hist",waits,"Vehicle Wait Times","Wait (s)","Count","#5090ff"),
        (axes[1],"line",log,"Cumulative Throughput","Sim Time (s)","Vehicles","#50ffa0"),
    ]:
        ax.set_facecolor("#171926")
        ax.set_title(title,color="#dce1f0",fontsize=9,pad=5)
        ax.set_xlabel(xl,color="#787e96",fontsize=7)
        ax.set_ylabel(yl,color="#787e96",fontsize=7)
        ax.tick_params(colors="#787e96",labelsize=6)
        for sp in ax.spines.values(): sp.set_color("#282b40")
        if kind=="hist" and data:
            ax.hist(data,bins=max(6,min(20,len(data)//3+1)),
                    color=col,edgecolor="#080a12",lw=0.5)
        elif kind=="line" and data:
            xs=[l[0] for l in data]; ys=list(range(1,len(data)+1))
            ax.plot(xs,ys,color=col,lw=1.3)
            ax.fill_between(xs,ys,alpha=0.10,color=col)
    fig.tight_layout(pad=1.5)
    canvas=agg.FigureCanvasAgg(fig); canvas.draw()
    s=pygame.image.frombuffer(canvas.buffer_rgba(),canvas.get_width_height(),"RGBA")
    plt.close(fig); _chart_cache=s.copy()
    return _chart_cache

# ══════════════════════════════════════════════════
#  STATS PANEL
# ══════════════════════════════════════════════════
def draw_panel(surf, sd, fonts, px, py, pw, ph):
    font,fsm,fxs = fonts
    rrect(surf,C["panel"],(px,py,pw,ph),r=10)
    pygame.draw.rect(surf,C["border"],(px,py,pw,ph),1,border_radius=10)
    y = py+14

    def ctr(txt,fy,fnt,col):
        s=fnt.render(txt,True,col)
        surf.blit(s,(px+pw//2-s.get_width()//2,fy))
        return fy+s.get_height()+4
    def div(fy):
        pygame.draw.line(surf,C["border"],(px+10,fy),(px+pw-10,fy))
        return fy+9
    def row(lbl,val,fy,vc=None):
        ls=fxs.render(lbl,True,C["text_dim"])
        vs=fxs.render(str(val),True,vc or C["text"])
        surf.blit(ls,(px+12,fy)); surf.blit(vs,(px+pw-12-vs.get_width(),fy))
        return fy+17

    y=ctr("TRAFFIC SIM",y,font,C["accent"])
    y=ctr("CS 324 · BatStateU",y,fxs,C["text_dim"])
    y=div(y); y=ctr(sd.scenario,y,fsm,C["accent2"]); y=div(y)

    with sd.lock:
        l_ns=sd.light_ns; l_ew=sd.light_ew; timer=sd.light_timer
        sim_t=sd.sim_time; tv=sd.total_vehicles
        wait=sum(1 for v in sd.vehicles if v["state"]=="queued")
        move=sum(1 for v in sd.vehicles if v["state"]=="moving")
        done=len(sd.completed); waits=list(sd.wait_times)

    def lbox(lbl,state,bx,by):
        cm={"green":C["green_light"],"yellow":C["yellow_light"],"red":C["red_light"]}
        col=cm.get(state,C["text_dim"])
        rrect(surf,(24,28,42),(bx,by,88,46),r=6)
        pygame.draw.rect(surf,col,(bx,by,88,46),2,border_radius=6)
        ls=fxs.render(lbl,True,C["text_dim"])
        surf.blit(ls,(bx+44-ls.get_width()//2,by+4))
        dot=pygame.Surface((14,14),pygame.SRCALPHA)
        pygame.draw.circle(dot,(*col,210),(7,7),7)
        surf.blit(dot,(bx+37,by+22))
        ss=fxs.render(state.upper(),True,col)
        surf.blit(ss,(bx+44-ss.get_width()//2,by+27))

    lbox("N-S",l_ns,px+6,y); lbox("E-W",l_ew,px+pw-94,y); y+=54
    ts=fxs.render(f"Phase: {int(timer):02d}s",True,C["text"])
    surf.blit(ts,(px+pw//2-ts.get_width()//2,y)); y+=18; y=div(y)
    y=row("Sim Time",f"{sim_t:.1f}s",y)
    y=row("Spawned",tv,y)
    y=row("Completed",done,y,C["green_light"])
    y=row("Waiting",wait,y,C["red_light"])
    y=row("Moving",move,y,C["yellow_light"])
    y=div(y)
    avg=sum(waits)/len(waits) if waits else 0
    mx=max(waits) if waits else 0
    y=row("Avg Wait",f"{avg:.1f}s",y)
    y=row("Max Wait",f"{mx:.1f}s",y); y=div(y)
    sp=fxs.render(f"Speed x{sd.speed_factor:.1f}",True,C["accent"])
    surf.blit(sp,(px+pw//2-sp.get_width()//2,y)); y+=18; y=div(y)
    for hint in ["[1]Normal [2]Rush [3]Low",
                 "[UP]Faster  [DOWN]Slower",
                 "[C]Charts [R]Reset [Q]Quit"]:
        hs=fxs.render(hint,True,C["text_dim"])
        surf.blit(hs,(px+pw//2-hs.get_width()//2,y)); y+=15

# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CS 324 — Traffic Light Simulation | BatStateU CICS")
    clock  = pygame.time.Clock()
    font = pygame.font.SysFont("monospace",15,bold=True)
    fsm  = pygame.font.SysFont("monospace",12,bold=True)
    fxs  = pygame.font.SysFont("monospace",10)

    road_surf = build_road_surface()
    threading.Thread(target=run_simulation,args=(SD,),daemon=True).start()

    px,py = WIDTH-PANEL_W-6, 6
    pw,ph = PANEL_W, HEIGHT-12
    show_charts = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running=False
            elif event.type == pygame.KEYDOWN:
                if   event.key==pygame.K_q: running=False
                elif event.key==pygame.K_1: SD.scenario="Normal Traffic"
                elif event.key==pygame.K_2: SD.scenario="Rush Hour"
                elif event.key==pygame.K_3: SD.scenario="Low Traffic"
                elif event.key==pygame.K_UP:
                    SD.speed_factor=min(8.0,round(SD.speed_factor+0.5,1))
                elif event.key==pygame.K_DOWN:
                    SD.speed_factor=max(0.5,round(SD.speed_factor-0.5,1))
                elif event.key==pygame.K_c: show_charts=not show_charts
                elif event.key==pygame.K_r:
                    with SD.lock:
                        SD.vehicles.clear(); SD.completed.clear()
                        SD.wait_times.clear(); SD.throughput_log.clear()
                        SD.total_vehicles=0

        screen.fill(C["bg"])
        if show_charts:
            screen.blit(build_chart(SD,VIEW_W,HEIGHT),(0,0))
            lb=fxs.render("Press [C] to return",True,C["text_dim"])
            screen.blit(lb,(VIEW_W//2-lb.get_width()//2,HEIGHT-26))
        else:
            screen.blit(road_surf,(0,0))
            draw_lights(screen,SD,fxs)
            draw_vehicles(screen,SD,fxs)
        draw_panel(screen,SD,(font,fsm,fxs),px,py,pw,ph)
        fps=fxs.render(f"FPS {int(clock.get_fps())}",True,C["text_dim"])
        screen.blit(fps,(6,6))
        pygame.display.flip()
        clock.tick(60)

    SD.running=False
    pygame.quit()
    if SD.completed:
        df=pd.DataFrame([{k:v for k,v in c.items() if k!="path"}
                         for c in SD.completed])
        df.to_csv("simulation_results.csv",index=False)
        print(f"Results saved ({len(SD.completed)} vehicles)")
    sys.exit(0)

if __name__=="__main__":
    main()