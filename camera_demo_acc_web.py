#!/usr/bin/env python3
"""
Real-time YOLO camera demo: distance estimation + in-lane colouring + YOLOPv2 lanes.

Single unified model (Vistas 8-class): car, truck, bus, motorcycle, bicycle,
person, traffic_sign, traffic_light.

Vehicles get pinhole-distance labels and zone colours (green/orange/red) when
in-lane; signs get magenta boxes; lights get yellow boxes.

Usage:
    .venv/bin/python camera_demo_acc.py [--camera 0] [--width 1280] [--height 720]

Controls:
    Q / ESC   quit
    S         save screenshot
    +/-       raise/lower confidence threshold
    F         toggle fullscreen
    R         cycle inference resolution: 1280 -> 640 -> 320 -> 160
    D         toggle debug panel
"""

import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import queue
import threading
from flask import Flask, Response, request

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()
key_queue = queue.Queue()

def generate_frames():
    target_delay = 1.0 / 25.0
    while True:
        t0 = time.time()
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        elapsed = time.time() - t0
        time.sleep(max(0.001, target_delay - elapsed))

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO ACC Demo (Headless)</title>
    <style>
        body { background-color: #111; color: #eee; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
        img { max-width: 100%; border: 2px solid #444; border-radius: 8px; }
        .controls { margin-top: 15px; font-size: 14px; color: #aaa; }
    </style>
</head>
<body>
    <h2>YOLO ACC Demo (Headless Web)</h2>
    <img src="/video_feed" />
    <div class="controls">
        Click here, then press keys: <b>Q/ESC</b> (quit), <b>S</b> (screenshot), <b>+/-</b> (confidence), <b>F</b> (fullscreen - ignored), <b>R</b> (resolution), <b>D</b> (debug)
    </div>
    <script>
        document.addEventListener('keydown', function(event) {
            fetch('/api/key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key: event.key.toLowerCase() })
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/key', methods=['POST'])
def api_key():
    data = request.json
    key_str = data.get('key', '')
    key_queue.put(key_str)
    return {"status": "ok"}
from ultralytics import YOLO

# ------------------------------------------------------------------
# CONFIG — TUNE THESE FOR YOUR CAMERA
WEIGHTS = "weights/yolo12m_vistas_best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INFERENCE_IMGSZ = 640

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# DISTANCE ESTIMATION — PINHOLE CAMERA MODEL
# ------------------------------------------------------------------
FOCAL_LENGTH_PX = 267.0

REF_HEIGHTS = {
    0: 1.5,   # car
    1: 3.5,   # truck
    2: 3.0,   # bus
    3: 1.1,   # motorcycle
    4: 1.1,   # bicycle
    5: 1.7,   # person
}

ZONE_RED_M = 2.0
ZONE_ORANGE_M = 5.0

LANE_HORIZON_FRAC = 0.515
LANE_MASK_CONF_THRESHOLD = 0.80  # YOLOPv2 lane-mask sigmoid cutoff for "is a lane pixel" (default 0.5 via round())
PEDESTRIAN_IGNORE_ABOVE_FRAC = 0.58  # pedestrians whose box sits above this fraction of frame height are too far away to matter — excluded from hazard checks (lane-line detection is unaffected)
MAX_LATERAL_PXS = 80.0

TRACK_MATCH_FRAC = 0.12
TRACK_MISS_MAX = 6

RESOLUTIONS = (1280, 640, 320, 160)

BAR_H = 48
MAX_WIN_W, MAX_WIN_H = 1920, 1080

VEHICLE_CLASS_IDS = (0, 1, 2, 3, 4)   # car truck bus motorcycle bicycle
SIGN_CLASS_ID     = 6
LIGHT_CLASS_ID    = 7
PERSON_CLASS_ID   = 5
OFF_LANE_COLOR = (60, 60, 60)

# Pedestrian-in-crosswalk hazard: two roughly-horizontal lane-line bands
# (crosswalk stripes / stop line) crossing the ego lane, with a person
# detected ahead of the car in that zone -> force a hold-brake stop
# regardless of the normal closing-rate/distance decision logic.
CROSSWALK_MIN_LINES     = 2
CROSSWALK_SPAN_FRAC     = 0.55   # row must be this much of lane width to count as a line
CROSSWALK_LINE_MIN_ROWS = 2      # min band thickness (rows) to filter noise
# Require most of the pedestrian's box to actually sit in the hazard zone
# (not just clip it with a corner) before it's treated as a hazard.
CROSSWALK_PERSON_MIN_OVERLAP_FRAC = 0.8
# How much further forward (up-frame, beyond the detected line) the hazard
# zone reaches, as a fraction of frame height — i.e. how far in front of the
# car a pedestrian can be and still count.
CROSSWALK_LOOKAHEAD_EXTEND_FRAC = 0.35
# Pedestrian path judgement: a pedestrian is a hazard only when actually in
# (or moving into) the ego lane. Walking parallel to the car — same direction
# on the shoulder/sidewalk, or opposite direction on the far side of the road —
# must NOT trip the brake. The ego lane is given by `left_xs`/`right_xs`;
# `lateral_pxs` (px/s) separates crossing (lateral) from walking-along motion.
PEDESTRIAN_LANE_MARGIN_FRAC       = 0.15   # slack beyond the ego-lane edge for tracking jitter
PEDESTRIAN_CROSSING_LATERAL_PXS   = 40.0   # min lateral px/s to count as crossing the road
PEDESTRIAN_CROSSING_MIN_TRACK_AGE = 2      # frames of tracking required to trust a crossing estimate
CROSSING_VERTICAL_TAU = 80.0             # px/s; how fast the required lateral speed grows with vertical (along-road) motion (shared: pedestrians + crossing vehicles)

STEADY_RATE_MPS   = 0.5
HEAVY_BRAKE_MPS   = 3.0
HOLD_DISTANCE_M   = 5.0

# Cross-traffic hazard: a vehicle moving fast laterally across the frame
# (e.g. through an intersection) rather than along the road with us. Lateral
# speed threshold is well above MAX_LATERAL_PXS (which merely disqualifies a
# track from "in lane"/lead selection) — this is a much higher bar meant to
# confidently distinguish genuine crossing traffic from lane drift/curves.
CROSSING_VEHICLE_LATERAL_PXS = 180.0   # px/s lateral speed to count as "crossing"
CROSSING_VEHICLE_MIN_TRACK_AGE = 3     # frames of consistent tracking required to trust the velocity estimate
CROSSING_VEHICLE_MIN_HORIZONTAL_LINES = 1  # horizontal lane bands (stop line / crosswalk stripe) required to treat lateral motion as crossing traffic
EGO_STILL_LINE_DY_PX = 3.0             # max frame-to-frame vertical drift (px) of the horizontal lines before the ego counts as "moving" (suppresses crossing flags)

HUD_PAD = 8
HUD_LINE_H = 20

DEBUG_KEY = 'd'


# ------------------------------------------------------------------
# COLOUR / DISTANCE HELPERS
# ------------------------------------------------------------------
def get_zone_color(distance_m):
    if distance_m < ZONE_RED_M:
        return (0, 0, 255)
    if distance_m < ZONE_ORANGE_M:
        return (0, 165, 255)
    return (0, 255, 0)

def estimate_distance_pinhole(bbox_h_px, cls_id):
    if bbox_h_px <= 0 or cls_id not in REF_HEIGHTS:
        return None
    return (REF_HEIGHTS[cls_id] * FOCAL_LENGTH_PX) / bbox_h_px


# ------------------------------------------------------------------
# TRACKER
# ------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    cls_id: int
    cx: float
    cy: float
    age: int = 0
    miss: int = 0
    dist: float | None = None
    dist_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    cx_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    cy_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    t_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    lateral_pxs: float = 0.0
    vertical_pxs: float = 0.0

class BoxTracker:
    def __init__(self):
        self.tracks = {}
        self._next_id = 1

    def update(self, boxes_xyxy, cls_ids, frame_w, t_now):
        dets = []
        for i, (box, cls_id) in enumerate(zip(boxes_xyxy, cls_ids)):
            x1, y1, x2, y2 = box
            dets.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, int(cls_id), i))

        max_match = frame_w * TRACK_MATCH_FRAC
        live = [tr for tr in self.tracks.values() if tr.miss <= TRACK_MISS_MAX]
        pairs = []
        for tr in live:
            for cx, cy, cls_id, di in dets:
                dx = tr.cx - cx
                dy = tr.cy - cy
                if math.hypot(dx, dy) <= max_match and tr.cls_id == cls_id:
                    pairs.append((math.hypot(dx, dy), tr.tid, di, cx, cy))
        pairs.sort(key=lambda p: p[0])

        matched_tids = set()
        matched_dis = set()
        det_to_track = {}
        for _, tid, di, cx, cy in pairs:
            if tid in matched_tids or di in matched_dis:
                continue
            tr = self.tracks[tid]
            tr.cx, tr.cy = cx, cy
            tr.cx_hist.append(cx)
            tr.cy_hist.append(cy)
            tr.t_hist.append(t_now)
            tr.lateral_pxs = self._slope(tr.cx_hist, tr.t_hist)
            tr.vertical_pxs = self._slope(tr.cy_hist, tr.t_hist)
            h = boxes_xyxy[di][3] - boxes_xyxy[di][1]
            tr.dist = estimate_distance_pinhole(h, tr.cls_id)
            if tr.dist is not None:
                tr.dist_hist.append(tr.dist)
            tr.age += 1
            tr.miss = 0
            matched_tids.add(tid)
            matched_dis.add(di)
            det_to_track[di] = tr

        for cx, cy, cls_id, di in dets:
            if di in matched_dis:
                continue
            tid = self._next_id
            self._next_id += 1
            tr = Track(tid=tid, cls_id=cls_id, cx=cx, cy=cy, age=1, miss=0)
            tr.cx_hist.append(cx)
            tr.cy_hist.append(cy)
            tr.t_hist.append(t_now)
            h = boxes_xyxy[di][3] - boxes_xyxy[di][1]
            tr.dist = estimate_distance_pinhole(h, tr.cls_id)
            if tr.dist is not None:
                tr.dist_hist.append(tr.dist)
            self.tracks[tid] = tr
            det_to_track[di] = tr

        for tr in self.tracks.values():
            if tr.tid not in matched_tids and tr.miss <= TRACK_MISS_MAX:
                tr.miss += 1
        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr.miss <= TRACK_MISS_MAX}
        return det_to_track

    @staticmethod
    def _slope(hist, t_hist):
        if len(hist) < 2: return 0.0
        dt = t_hist[-1] - t_hist[0]
        if dt <= 1e-3: return 0.0
        return (hist[-1] - hist[0]) / dt


def get_ego_lane(ll_mask):
    """Extract the ego lane mask and boundaries dynamically from YOLOPv2 lane mask.

    Also returns `locked`: per-row bool, True only where BOTH the left and
    right boundary were pinned by an actual lane-line pixel that row (not
    carried forward from the previous row) — i.e. where the "outside lines"
    are clearly visible, as opposed to a guess/hold-over.
    """
    H, W = ll_mask.shape
    ego_mask = np.zeros_like(ll_mask, dtype=np.uint8)
    
    left_xs = np.zeros(H, dtype=np.int32)
    right_xs = np.full(H, W - 1, dtype=np.int32)
    locked = np.zeros(H, dtype=bool)
    
    cx = W // 2
    left_x = int(W * 0.2)
    right_x = int(W * 0.8)
    
    # Scan bottom row to lock onto actual lane lines if they exist
    bot_row = ll_mask[H-1, :]
    nz = np.nonzero(bot_row)[0]
    if len(nz) > 0:
        l_cands = nz[nz < cx]
        if len(l_cands) > 0: left_x = l_cands[-1]
        r_cands = nz[nz >= cx]
        if len(r_cands) > 0: right_x = r_cands[0]
        
    search_rad = int(W * 0.15)
    horizon = int(H * LANE_HORIZON_FRAC)
    
    for y in range(H - 1, horizon - 1, -1):
        row = ll_mask[y, :]
        nz = np.nonzero(row)[0]
        l_found = r_found = False
        if len(nz) > 0:
            l_cands = nz[(nz >= left_x - search_rad) & (nz < cx)]
            if len(l_cands) > 0: left_x = l_cands[-1]; l_found = True
            
            r_cands = nz[(nz >= cx) & (nz <= right_x + search_rad)]
            if len(r_cands) > 0: right_x = r_cands[0]; r_found = True
            
        if left_x >= right_x: 
            left_x = max(0, right_x - 10)
            
        cx = (left_x + right_x) // 2
        
        left_xs[y] = left_x
        right_xs[y] = right_x
        locked[y] = l_found and r_found
        ego_mask[y, left_x:right_x] = 1
        
    return ego_mask, left_xs, right_xs, locked

def detect_lane_horizontal_lines(ll_mask, left_xs, right_xs, horizon_y, frame_h):
    """Find horizontal-line bands (crosswalk stripes / stop line) crossing the ego lane.

    A row counts as "line" when its lane-mask coverage spans most of the lane
    width at that row (unlike normal lane-boundary marks, which only light up
    near the left/right edges). Consecutive hit rows are grouped into bands;
    returns a list of (y_top, y_bottom) ordered top-to-bottom (far-to-near).
    """
    bands = []
    band_start = None
    for y in range(max(0, horizon_y), frame_h):
        lx, rx = int(left_xs[y]), int(right_xs[y])
        lane_w = rx - lx
        row_hit = lane_w > 0 and np.count_nonzero(ll_mask[y, lx:rx]) >= lane_w * CROSSWALK_SPAN_FRAC
        if row_hit:
            if band_start is None:
                band_start = y
        elif band_start is not None:
            if y - band_start >= CROSSWALK_LINE_MIN_ROWS:
                bands.append((band_start, y - 1))
            band_start = None
    if band_start is not None and frame_h - band_start >= CROSSWALK_LINE_MIN_ROWS:
        bands.append((band_start, frame_h - 1))
    return bands


def _ego_lane_bounds_at(left_xs, right_xs, locked, y, frame_h):
    """Ego-lane lateral bounds `(lane_l, lane_r)` at row `y`, or None when no
    *locked* row (both boundaries pinned by real lane pixels) is available.

    Inside a crosswalk the outside lane lines are frequently obscured by the
    stripe paint, so `left_xs`/`right_xs` there are unreliable. Prefer the
    nearest locked row at or below `y` (closer to the car = more trustworthy),
    then the nearest locked row above.
    """
    y = int(np.clip(y, 0, frame_h - 1))
    if locked[y]:
        return float(left_xs[y]), float(right_xs[y])
    below = np.nonzero(locked[y:])[0]
    if len(below):
        yy = y + int(below[0])
        return float(left_xs[yy]), float(right_xs[yy])
    above = np.nonzero(locked[:y + 1])[0]
    if len(above):
        yy = int(above[-1])
        return float(left_xs[yy]), float(right_xs[yy])
    return None

def _required_lateral(vertical_pxs, floor_pxs):
    """Lateral speed (px/s) required to classify motion as "crossing the road".

    The bar starts at `floor_pxs` when vertical (along-road) motion is zero and
    grows exponentially with `|vertical_pxs|`: an object with no Y-change is
    moving perpendicular and needs only the floor, while one moving along the
    road must show a drastically larger lateral speed before it counts as
    crossing rather than perspective convergence toward the vanishing point.
    """
    return floor_pxs * math.exp(abs(vertical_pxs) / CROSSING_VERTICAL_TAU)

def _horizontal_lines_ref_y(lines):
    """Representative vertical position of the detected horizontal bands,
    used to measure ego motion frame-to-frame. Averaging the bands' top edges
    is more robust to single-stripe jitter than any one band."""
    if not lines:
        return None
    return float(sum(b[0] for b in lines) / len(lines))


def _pedestrian_in_ego_path(box, tr, left_xs, right_xs, locked, frame_h, horizon_y):
    """Whether a pedestrian is actually a threat to the ego vehicle.

    A person standing or walking INSIDE the ego lane (plus a small margin for
    tracking jitter) is a hazard regardless of heading. Someone beside the lane
    is a hazard only when moving laterally INTO it (crossing) — walking
    parallel to the car (same or opposite direction) is not.
    """
    x1, _y1, x2, y2 = box
    y_check = int(np.clip(y2, 0, frame_h - 1))
    if y_check < horizon_y:
        return False
    bounds = _ego_lane_bounds_at(left_xs, right_xs, locked, y_check, frame_h)
    if bounds is None:
        return False
    lane_l, lane_r = bounds
    lane_w = max(lane_r - lane_l, 1.0)
    margin = lane_w * PEDESTRIAN_LANE_MARGIN_FRAC
    cx = (x1 + x2) * 0.5

    if lane_l - margin <= cx <= lane_r + margin:
        return True

    if tr is None or tr.age < PEDESTRIAN_CROSSING_MIN_TRACK_AGE:
        return False
    lat = tr.lateral_pxs
    required_lateral = _required_lateral(tr.vertical_pxs, PEDESTRIAN_CROSSING_LATERAL_PXS)
    if abs(lat) < required_lateral:
        return False
    if cx < lane_l:
        return lat > 0.0   # left of the lane, moving right into it
    return lat < 0.0       # right of the lane, moving left into it


def find_crosswalk_pedestrian_hazard(lines, boxes_xyxy, cls_ids, det_to_track, left_xs, right_xs, locked, frame_h, horizon_y):
    """Marked-crosswalk mode: a person is a hazard when in the vertical span
    bounded by the two detected parallel lines (plus an early-warning lookahead
    margin above the far line) AND actually in (or moving into) the ego lane —
    see `_pedestrian_in_ego_path`. Requires >= CROSSWALK_MIN_LINES bands.

    A pedestrian box only counts once at least CROSSWALK_PERSON_MIN_OVERLAP_FRAC
    of its height falls inside the vertical zone — a box merely clipping the
    boundary doesn't count.

    Pedestrians whose box sits entirely above PEDESTRIAN_IGNORE_ABOVE_FRAC of
    the frame (i.e. too far away) are excluded regardless of overlap/lane fit.
    """
    if len(lines) < CROSSWALK_MIN_LINES:
        return []
    gap_top, gap_bottom = lines[0][0], lines[-1][1]  # topmost line's top .. nearest line's bottom
    zone_top = max(horizon_y, gap_top - int(frame_h * CROSSWALK_LOOKAHEAD_EXTEND_FRAC))
    zone_bottom = gap_bottom
    ignore_above_y = frame_h * PEDESTRIAN_IGNORE_ABOVE_FRAC
    hazard = []
    for i, (box, cls_id) in enumerate(zip(boxes_xyxy, cls_ids)):
        if int(cls_id) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = box
        if y2 < ignore_above_y:
            continue
        box_h = y2 - y1
        if box_h <= 0:
            continue
        overlap = min(y2, zone_bottom) - max(y1, zone_top)
        if overlap <= 0:
            continue
        if (overlap / box_h) < CROSSWALK_PERSON_MIN_OVERLAP_FRAC:
            continue
        if _pedestrian_in_ego_path(box, det_to_track.get(i), left_xs, right_xs, locked, frame_h, horizon_y):
            hazard.append(i)
    return hazard


def find_jaywalker_hazard(boxes_xyxy, cls_ids, det_to_track, left_xs, right_xs, locked, frame_h, horizon_y):
    """No clear marked crosswalk (< CROSSWALK_MIN_LINES parallel lines):
    fall back to the outside (lane-boundary) lines directly to judge whether
    a pedestrian is in the vehicle's path (jaywalking). Only rows where those
    lines are clearly tracked (`locked`) are trusted — an unlocked/held-over
    row can't reliably place a jaywalker in-lane.

    Pedestrians whose box sits entirely above PEDESTRIAN_IGNORE_ABOVE_FRAC of
    the frame (i.e. too far away) are excluded regardless of lane fit.
    """
    ignore_above_y = frame_h * PEDESTRIAN_IGNORE_ABOVE_FRAC
    hazard = []
    for i, (box, cls_id) in enumerate(zip(boxes_xyxy, cls_ids)):
        if int(cls_id) != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = box
        if y2 < ignore_above_y:
            continue
        if _pedestrian_in_ego_path(box, det_to_track.get(i), left_xs, right_xs, locked, frame_h, horizon_y):
            hazard.append(i)
    return hazard



def track_in_lane(tr, y_bot, y_top, frame_h, left_xs, right_xs):
    if tr.cls_id not in VEHICLE_CLASS_IDS: return False
    if y_bot < LANE_HORIZON_FRAC * frame_h: return False
    
    y_check = int(np.clip(y_bot, 0, frame_h - 1))
    if tr.cx < left_xs[y_check] or tr.cx > right_xs[y_check]: 
        return False
        
    if y_top >= LANE_HORIZON_FRAC * frame_h:
        y_top_check = int(np.clip(y_top, 0, frame_h - 1))
        if tr.cx < left_xs[y_top_check] or tr.cx > right_xs[y_top_check]: 
            return False
            
    return abs(tr.lateral_pxs) <= MAX_LATERAL_PXS


def find_crossing_vehicle_hazard(boxes_xyxy, cls_ids, det_to_track, horizon_y, horizontal_lines, ego_line_dy):
    """Flag vehicles moving fast laterally across the frame — cross traffic
    through an intersection. Only active when a horizontal lane-marking band
    (stop line / crosswalk stripe — the green horizontal lane line) is detected,
    signalling an intersection, AND the ego is ~stopped there (the bands aren't
    drifting vertically). On open road, or while driving through an
    intersection, lateral motion is our own parallax past parked/stopped cars,
    not crossing traffic.

    Requires a few frames of consistent tracking (`CROSSING_VEHICLE_MIN_TRACK_AGE`)
    so the velocity estimate isn't just first-frame jitter, that the box is on
    the road (below the horizon), and applies the same Y-motion gate as
    pedestrians: a vehicle with lots of vertical (along-road) motion must show
    a drastically larger lateral speed to count as crossing.
    """
    if len(horizontal_lines) < CROSSING_VEHICLE_MIN_HORIZONTAL_LINES:
        return []
    if abs(ego_line_dy) > EGO_STILL_LINE_DY_PX:
        return []
    hazard = []
    for i, (box, cls_id) in enumerate(zip(boxes_xyxy, cls_ids)):
        if int(cls_id) not in VEHICLE_CLASS_IDS:
            continue
        tr = det_to_track.get(i)
        if tr is None or tr.age < CROSSING_VEHICLE_MIN_TRACK_AGE:
            continue
        _, _, _, y2 = box
        if y2 < horizon_y:
            continue
        if abs(tr.lateral_pxs) >= _required_lateral(tr.vertical_pxs, CROSSING_VEHICLE_LATERAL_PXS):
            hazard.append(i)
    return hazard

def select_lead(tracks, det_to_track, boxes_xyxy, cls_ids, frame_h, left_xs, right_xs):
    lead, lead_dist = None, None
    for di, tr in det_to_track.items():
        if tr is None or tr not in tracks.values(): continue
        x1, y1, x2, y2 = boxes_xyxy[di]
        if not track_in_lane(tr, float(y2), float(y1), frame_h, left_xs, right_xs): continue
        if tr.dist is None: continue
        if lead_dist is None or tr.dist < lead_dist:
            lead, lead_dist = tr, tr.dist
    return lead

def closing_rate_mps(tr):
    if len(tr.dist_hist) < 2 or len(tr.t_hist) < 2: return 0.0
    dt = tr.t_hist[-1] - tr.t_hist[0]
    if dt <= 1e-3: return 0.0
    return (tr.dist_hist[0] - tr.dist_hist[-1]) / dt

@dataclass
class Decision:
    action: str
    reason: str
    lead_tid: int | None
    lead_dist: float | None
    closing_rate: float | None

def decide_action(lead):
    """Advisory only — a suggestion, not a defensive guarantee. Three
    outcomes: NO_SUGGESTION (nothing to flag), LIGHT_BRAKE (mild closing or
    steady & close), HEAVY_BRAKE (closing fast). No accel/hold-speed
    suggestions — separating or comfortably-spaced leads are NO_SUGGESTION."""
    if lead is None:
        return Decision("NO_SUGGESTION", "No in-lane lead", None, None, None)
    if lead.dist is None:
        return Decision("NO_SUGGESTION", "Lead detected, awaiting range", lead.tid, None, None)
    cr, d = closing_rate_mps(lead), lead.dist
    if cr > STEADY_RATE_MPS:
        if cr >= HEAVY_BRAKE_MPS:
            return Decision("HEAVY_BRAKE", f"Lead closing fast ({cr:+.1f} m/s) at {d:.1f} m", lead.tid, d, cr)
        return Decision("LIGHT_BRAKE", f"Lead closing ({cr:+.1f} m/s) at {d:.1f} m", lead.tid, d, cr)
    if d < HOLD_DISTANCE_M:
        return Decision("LIGHT_BRAKE", f"Lead steady & short at {d:.1f} m", lead.tid, d, cr)
    return Decision("NO_SUGGESTION", f"Lead steady at {d:.1f} m", lead.tid, d, cr)

# ------------------------------------------------------------------
# YOLOPv2 (LANE DETECTION)
# ------------------------------------------------------------------
def infer_yolopv2(model, frame, device):
    """Run YOLOPv2 natively in PyTorch to get lane line mask."""
    # Resize to 1280x720, then letterbox to 640x384
    img0 = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img0, (640, 360), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    
    img_tensor = torch.from_numpy(img).to(device)
    if model.parameters().__next__().dtype == torch.float16:
        img_tensor = img_tensor.half()
    else:
        img_tensor = img_tensor.float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        _, _, ll = model(img_tensor)
        
    # Extract the valid region and upsample
    ll_predict = ll[:, :, 12:372, :]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear', align_corners=False)
    ll_seg_mask = (ll_seg_mask >= LANE_MASK_CONF_THRESHOLD).squeeze()
    ll_mask = ll_seg_mask.byte().cpu().numpy()
    return ll_mask

def draw_dynamic_lane(annotated, ego_mask, ll_mask):
    """Overlay the dynamic ego lane and YOLOPv2 lane mask."""
    h, w = annotated.shape[:2]
    if ll_mask.shape != (h, w):
        ll_mask = cv2.resize(ll_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if ego_mask.shape != (h, w):
        ego_mask = cv2.resize(ego_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    overlay = annotated.copy()
    
    # Dark gray for ego lane drivable area
    overlay[ego_mask == 1] = (70, 70, 70)
    # Vibrant green for lane boundaries
    overlay[ll_mask == 1] = (0, 255, 0)
    
    active = (ego_mask == 1) | (ll_mask == 1)
    annotated[active] = cv2.addWeighted(annotated[active], 0.6, overlay[active], 0.4, 0).squeeze()
    return annotated

# ------------------------------------------------------------------
# CAMERA & DRAWING
# ------------------------------------------------------------------
class CameraCapture:
    def __init__(self, source=0, target_w=1280, target_h=720):
        self.target_w, self.target_h = target_w, target_h
        self.frame, self.running, self.cam_fps = None, True, 0.0
        self._lock = threading.Lock()
        if isinstance(source, int) and sys.platform.startswith("linux"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(30):
            ok, f = self.cap.read()
            if ok: break
            time.sleep(0.05)
        else:
            self.cap.release()
            raise RuntimeError(f"Camera {source} opened but no frames arriving.")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        t0, count = time.time(), 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.001)
                continue
            if frame.shape[1] != self.target_w or frame.shape[0] != self.target_h:
                frame = cv2.resize(frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            with self._lock:
                self.frame = frame
            count += 1
            if time.time() - t0 >= 1.0:
                self.cam_fps, count, t0 = count / (time.time() - t0), 0, time.time()

    def get(self):
        with self._lock: return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

def draw_overlay(annotated, inference_fps, conf, imgsz, num_dets):
    canvas = cv2.copyMakeBorder(annotated, BAR_H, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(canvas, f"YOLO: {inference_fps:.1f} fps  |  res: {imgsz}  |  conf: {conf:.2f}  |  Dets: {num_dets}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[Q]uit  [S]creenshot  [+/-]conf  [F]ullscreen  [R]esolution  [D]ebug", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def draw_detections(frame, boxes_xyxy, confs, cls_ids, names, det_to_track, left_xs, right_xs, hazard_indices=frozenset(), crossing_hazard_indices=frozenset()):
    """Draw all detections from the unified model.

    Vehicles: distance-zone colour when in-lane, gray otherwise.
    Traffic signs: magenta.
    Traffic lights: yellow.
    Persons flagged in `hazard_indices` (crosswalk pedestrian hazard): forced red.
    Vehicles flagged in `crossing_hazard_indices` (cross traffic): forced red.
    """
    frame_h, frame_w = frame.shape[:2]
    for i, ((x1, y1, x2, y2), conf, cls_id) in enumerate(zip(boxes_xyxy, confs, cls_ids)):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id_i = int(cls_id)

        if i in crossing_hazard_indices:
            # Vehicle crossing our path — always red, highest-priority hazard
            color      = (0, 0, 255)
            text_color = (255, 255, 255)
            thickness  = 3
            label      = f"[CROSSING VEHICLE] {conf:.2f}"
            font_scale = 0.6
        elif i in hazard_indices:
            # Pedestrian in crosswalk hazard zone — always red, regardless of zone/lane colouring
            color      = (0, 0, 255)
            text_color = (255, 255, 255)
            thickness  = 3
            label      = f"[PEDESTRIAN HAZARD] {conf:.2f}"
            font_scale = 0.6
        elif cls_id_i == SIGN_CLASS_ID:
            # Traffic sign — magenta, no distance
            color      = (255, 0, 255)
            text_color = (255, 255, 255)
            thickness  = 2
            label      = f"[sign] {conf:.2f}"
            font_scale = 0.5
        elif cls_id_i == LIGHT_CLASS_ID:
            # Traffic light — yellow, no distance
            color      = (0, 215, 255)
            text_color = (0, 0, 0)
            thickness  = 2
            label      = f"[light] {conf:.2f}"
            font_scale = 0.5
        else:
            # Vehicle / person — distance + zone colour
            dist     = estimate_distance_pinhole(y2 - y1, cls_id_i)
            dist_str = f"{dist:.1f}m" if dist is not None else "?m"
            tr       = det_to_track.get(i)
            in_lane  = tr is not None and track_in_lane(tr, float(y2), float(y1), frame_h, left_xs, right_xs)
            if in_lane and dist is not None:
                color, text_color, thickness = get_zone_color(dist), (0, 0, 0), 2
            else:
                color, text_color, thickness = OFF_LANE_COLOR, (255, 255, 255), 1
            label      = f"{names[cls_id_i]} {conf:.2f}  {dist_str}"
            font_scale = 0.6

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        if y1 - th - 8 >= 0:
            rect_tl, rect_br, text_org = (x1, y1 - th - 8), (x1 + tw + 4, y1), (x1 + 2, y1 - 4)
        else:
            rect_tl, rect_br, text_org = (x1, y1), (x1 + tw + 4, y1 + th + 8), (x1 + 2, y1 + th + 4)
        cv2.rectangle(frame, rect_tl, rect_br, color, -1)
        cv2.putText(frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
    return frame

def action_color(action):
    if action == "HEAVY_BRAKE": return (0, 0, 255)
    if action == "LIGHT_BRAKE": return (0, 165, 255)
    return (255, 200, 0)

def _semi_panel(annotated, tl, br, fill_alpha, border_color, border_thick=1):
    overlay = annotated.copy()
    cv2.rectangle(overlay, tl, br, (0, 0, 0), -1)
    cv2.addWeighted(overlay, fill_alpha, annotated, 1 - fill_alpha, 0, annotated)
    cv2.rectangle(annotated, tl, br, border_color, border_thick)

def draw_hud(annotated, decision):
    h, w = annotated.shape[:2]
    (tw1, th1), _ = cv2.getTextSize(decision.action, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (tw2, th2), _ = cv2.getTextSize(decision.reason, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    box_w, box_h = max(tw1, tw2) + 2 * HUD_PAD, th1 + th2 + 3 * HUD_PAD
    br = (w - HUD_PAD, h - HUD_PAD)
    tl = (br[0] - box_w, br[1] - box_h)
    color = action_color(decision.action)
    _semi_panel(annotated, tl, br, 0.55, color)
    cv2.putText(annotated, decision.action, (tl[0] + HUD_PAD, tl[1] + HUD_PAD + th1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(annotated, decision.reason, (tl[0] + HUD_PAD, tl[1] + 2 * HUD_PAD + th1 + th2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated

def draw_debug_panel(annotated, decision, show):
    if not show: return annotated
    h, w = annotated.shape[:2]
    lines = [
        "=== ACC DECISION DEBUG ===",
        f"Action      : {decision.action}",
        f"Reason      : {decision.reason}",
        f"Lead TID    : {decision.lead_tid if decision.lead_tid is not None else 'none'}",
        f"Lead dist   : {decision.lead_dist if decision.lead_dist is not None else '--'} m",
        f"Closing rate: {decision.closing_rate if decision.closing_rate is not None else '--'} m/s",
        "---",
        "Thresholds:",
        f"  STEADY   <= {STEADY_RATE_MPS} m/s",
        f"  HEAVY    >= {HEAVY_BRAKE_MPS} m/s",
        f"  HOLD_DIST= {HOLD_DISTANCE_M} m",
    ]
    panel_w = min(320, max(220, w // 2 - 2 * HUD_PAD))
    tl, br = (HUD_PAD, h // 2 - (HUD_LINE_H * len(lines)) // 2), (HUD_PAD + panel_w, h // 2 + (HUD_LINE_H * len(lines)) // 2)
    _semi_panel(annotated, tl, br, 0.55, (160, 160, 160))
    for i, ln in enumerate(lines):
        y = tl[1] + (i + 1) * HUD_LINE_H - 5
        col = (0, 255, 255) if i == 0 else (180, 180, 180) if ln == "---" else (220, 220, 220)
        cv2.putText(annotated, ln, (tl[0] + HUD_PAD, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, action_color(decision.action) if i == 1 else col, 2 if i == 1 else 1, cv2.LINE_AA)
    return annotated

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--imgsz", type=int, default=INFERENCE_IMGSZ)
    args = parser.parse_args()

    print(f"[Info] Loading unified detection model: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    
    print(f"[Info] Loading YOLOPv2 Lane Detection model...")
    try:
        yolopv2_model = torch.jit.load("weights/yolopv2.pt", map_location=DEVICE)
        yolopv2_model.eval()
    except Exception as e:
        print(f"[Error] Failed to load YOLOPv2 model: {e}")
        raise SystemExit(1)
        
    print(f"[Info] Device: {DEVICE}")

    print(f"[Info] Opening camera {args.camera} at {args.width}x{args.height} ...")
    try:
        cam = CameraCapture(source=args.camera, target_w=args.width, target_h=args.height)
    except RuntimeError as e:
        print(f"[Error] {e}")
        raise SystemExit(1)
    print("[Info] Camera ready.")

    conf, imgsz, half, show_debug = CONF_THRESHOLD, args.imgsz, DEVICE != "cpu", False
    res_idx = RESOLUTIONS.index(imgsz) if imgsz in RESOLUTIONS else 1
    inference_times = deque(maxlen=30)
    frame_count, t_start = 0, time.time()
    tracker = BoxTracker()
    last_action = None
    prev_line_ref_y = None

    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    print("[Info] Web server started at http://localhost:5000 (Headless mode)")

    running = True
    take_screenshot = False
    while running:
        while not key_queue.empty():
            key = key_queue.get()
            if key in ('q', 'escape'): running = False
            elif key == 's':
                take_screenshot = True
            elif key in ('+', '='):
                conf = min(1.0, conf + 0.05)
                print(f"[Config] conf = {conf:.2f}")
            elif key == '-':
                conf = max(0.05, conf - 0.05)
                print(f"[Config] conf = {conf:.2f}")
            elif key == 'f':
                print(f"[Config] fullscreen = ignored in headless mode")
            elif key == 'r':
                res_idx = (res_idx + 1) % len(RESOLUTIONS)
                imgsz = RESOLUTIONS[res_idx]
                print(f"[Config] inference resolution = {imgsz}")
            elif key == 'd':
                show_debug = not show_debug
                print(f"[Config] debug panel = {show_debug}")

        frame = cam.get()
        if frame is None:
            time.sleep(0.016)
            continue

        t0 = time.time()
        # Run YOLO Object Detection
        results = model.predict(source=frame, conf=conf, iou=IOU_THRESHOLD, imgsz=imgsz, device=DEVICE, verbose=False, half=half)[0]
        
        annotated = frame.copy()
        
        # Run YOLOPv2 Lane Detection
        crosswalk_lines = []
        locked = None
        horizon_y = int(frame.shape[0] * LANE_HORIZON_FRAC)
        try:
            ll_mask = infer_yolopv2(yolopv2_model, frame, DEVICE)
            ego_mask, left_xs, right_xs, locked = get_ego_lane(ll_mask)
            annotated = draw_dynamic_lane(annotated, ego_mask, ll_mask)
            crosswalk_lines = detect_lane_horizontal_lines(ll_mask, left_xs, right_xs, horizon_y, frame.shape[0])
        except Exception as e:
            print(f"[Error] YOLOPv2 Inference failed: {e}")
            h, w = frame.shape[:2]
        # (signs and lights now come from the unified model below)
        # Ego-motion proxy: vertical drift of the horizontal lane lines since
        # last frame. Large drift = we're driving through the intersection, so
        # apparent lateral motion of parked/stopped cars is parallax, not
        # crossing traffic (see find_crossing_vehicle_hazard).
        line_ref_y = _horizontal_lines_ref_y(crosswalk_lines)
        ego_line_dy = 0.0
        if line_ref_y is None:
            prev_line_ref_y = None
        else:
            if prev_line_ref_y is not None:
                ego_line_dy = line_ref_y - prev_line_ref_y
            prev_line_ref_y = line_ref_y

        inference_times.append(time.time() - t0)
        num_dets = len(results.boxes) if results.boxes else 0
        inf_fps  = 1.0 / (sum(inference_times) / len(inference_times)) if inference_times else 0

        pedestrian_hazard_kind = None
        crossing_hazard_indices = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes, confs, cls_ids = results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()
            det_to_track = tracker.update(boxes, cls_ids, frame.shape[1], t0)
            hazard_indices = []
            if locked is not None:
                if len(crosswalk_lines) >= CROSSWALK_MIN_LINES:
                    hazard_indices = find_crosswalk_pedestrian_hazard(crosswalk_lines, boxes, cls_ids, det_to_track, left_xs, right_xs, locked, frame.shape[0], horizon_y)
                    if hazard_indices: pedestrian_hazard_kind = "crosswalk"
                else:
                    hazard_indices = find_jaywalker_hazard(boxes, cls_ids, det_to_track, left_xs, right_xs, locked, frame.shape[0], horizon_y)
                    if hazard_indices: pedestrian_hazard_kind = "jaywalker"
            crossing_hazard_indices = find_crossing_vehicle_hazard(boxes, cls_ids, det_to_track, horizon_y, crosswalk_lines, ego_line_dy)
            annotated = draw_detections(annotated, boxes, confs, cls_ids, model.names, det_to_track, left_xs, right_xs, hazard_indices, crossing_hazard_indices)
            lead = select_lead(tracker.tracks, det_to_track, boxes, cls_ids, frame.shape[0], left_xs, right_xs)
        else:
            hazard_indices = []
            lead = None

        decision = decide_action(lead)
        if hazard_indices:
            reason_txt = "Pedestrian in crosswalk ahead" if pedestrian_hazard_kind == "crosswalk" else "Pedestrian jaywalking ahead"
            decision = Decision("HEAVY_BRAKE", f"{reason_txt} ({len(hazard_indices)})", decision.lead_tid, decision.lead_dist, decision.closing_rate)
        if crossing_hazard_indices:
            decision = Decision("HEAVY_BRAKE", f"Vehicle crossing our path ({len(crossing_hazard_indices)})", decision.lead_tid, decision.lead_dist, decision.closing_rate)

        if decision.action != last_action:
            cr_str = f"{decision.closing_rate:+.2f}" if decision.closing_rate is not None else "None"
            print(f"[Decision] {decision.action} - {decision.reason}  (d={decision.lead_dist}, cr={cr_str} m/s, lead={decision.lead_tid})")
            last_action = decision.action

        annotated = draw_overlay(annotated, inf_fps, conf, imgsz, num_dets)
        annotated = draw_hud(annotated, decision)
        annotated = draw_debug_panel(annotated, decision, show_debug)

        if take_screenshot:
            ss_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(ss_path, annotated)
            print(f"[Screenshot] {ss_path}")
            take_screenshot = False
        global latest_frame
        with frame_lock:
            latest_frame = annotated.copy()
        
        frame_count += 1
        time.sleep(0.001)

    cam.stop()
    elapsed = time.time() - t_start
    print(f"\n[Done] {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps avg)")

if __name__ == "__main__":
    main()
