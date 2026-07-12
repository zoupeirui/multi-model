"""
eye_direction_detector.py
=========================
Real-time LEFT / CENTER / RIGHT gaze classifier for a single-eye close-up camera,
following the Pupil Labs image-processing pipeline (Top-Hat glint suppression →
adaptive dark-pupil segmentation → ellipse fit) AND adding the **Pupil-Center /
Corneal-Reflection (PCCR)** vector that Tobii / commercial eye trackers use to
get robust gaze direction in a NON-head-mounted setup.

Why PCCR (and not glints alone, not pupil position alone):
- Glints are reflections of the IR LEDs on the cornea. They translate with the
  HEAD but barely with the EYE rotation (the cornea is a small near-spherical
  cap so its image position is almost head-locked).
- The pupil center moves with the EYE rotation.
- The vector  v = pupil_center - glint_center  therefore encodes gaze direction
  while being largely invariant to head translation. This is exactly what the
  user observed: when the eye rotates left/right, the glints "drift onto the
  white" relative to the pupil — that drift IS the signal.

Workflow:
  1.  python eye_direction_detector.py --camera 1 --mirror --select-on-start
  2.  Drag a tight ROI around the eye, press SPACE/ENTER.
  3.  (Optional) Calibrate:
        Press '1', look hard left  for ~1.2 s while it samples.
        Press '2', look straight   for ~1.2 s.
        Press '3', look hard right for ~1.2 s.
        Press 's' to save calibration to eye_calib.json.
  4.  Run. The status bar shows LEFT / CENTER / RIGHT / INVALID.
  5.  Trackbars let you tune thresholds live. 'd' toggles debug masks.

CLI:
  --camera N         camera index (default 0)
  --mirror           horizontal flip (set this for selfie-cam feel)
  --select-on-start  prompt for ROI selection at startup
  --video PATH       use a video file instead of a live camera (for offline tests)
  --calib PATH       load calibration from json
  --list             list available cameras and exit

Public API (for future ROS 2 wrapper):
  result = detector.process(roi_bgr, roi_origin_xy)
  result.valid   : bool
  result.label   : "LEFT" | "CENTER" | "RIGHT" | "INVALID"
  result.dx_norm : float       (pupil_x - glint_x) / pupil_radius
  result.confidence : float    in [0, 1]
"""



import argparse
import collections
import dataclasses
import json
import os
import sys
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration / tunables
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    # --- Glint detector (morphological Top-Hat: brightness-invariant) ---
    tophat_kernel: int = 15            # Top-Hat structuring element diameter
    # tophat_thr was 25 → raised to 35 to reject dim skin reflections that
    # otherwise pollute the glint cluster centroid and make dx_norm jitter.
    tophat_thr: int = 35
    # glint_min_area was 3 → 5 (drops single-pixel speckle noise)
    glint_min_area: int = 5
    glint_max_area: int = 400
    # glint_max_blobs was 3 → 2. We only have 2 IR LEDs; the 3rd "spare"
    # slot was being filled by skin / iris highlights and pulling the
    # cluster centroid off the corneal reflection.
    glint_max_blobs: int = 2

    # --- Pupil detector (glint-anchored search) ---
    # The pupil sits AT the glints because the cornea reflects the LEDs and the
    # cornea sits over the pupil. We search a window of this radius around the
    # glint cluster. This radius should cover the maximum pupil-glint offset
    # under extreme gaze (typically ≤ 1.5 pupil diameters).
    pupil_search_radius: int = 90

    # Within the search window, the pupil is the very darkest fraction.
    # Threshold = window_min_gray + pupil_thr_offset. Anything within
    # `pupil_thr_offset` gray levels of the darkest pixel is considered
    # "pupil". The iris (50+ gray levels brighter than the pupil) is
    # excluded. This is more robust than percentile to varying iris darkness.
    pupil_thr_offset: int = 25
    # Pupil-only area bounds (NOT iris). The pupil at this resolution is
    # ~30-50 px diameter → core area ~700-2000 px. After morph-close it can
    # absorb adjacent iris and reach ~10000. The full iris is 20000+, so
    # 12000 keeps us safely below the iris.
    pupil_min_area: int = 100
    pupil_max_area: int = 12000
    pupil_min_circularity: float = 0.20
    pupil_min_aspect: float = 0.25
    erode_kernel: int = 9

    # --- Decision thresholds (default; calibration overrides) ---
    deadband: float = 0.30             # |dx_norm| <= deadband → CENTER (uncalibrated)
    invalid_below_conf: float = 0.30

    # --- dx_norm temporal smoothing (EMA) ---
    # The discrete label smoother only operates on LEFT/CENTER/RIGHT/INVALID,
    # so a single noisy dx_norm sample that crosses `deadband` can flip the
    # raw label. An EMA over dx_norm itself stops that.
    #   ema_new = alpha * raw + (1 - alpha) * ema_old
    # alpha=1.0 disables smoothing; smaller values smooth more but lag more.
    dx_ema_alpha: float = 0.4
    # If we go this long without a confidence-passing frame, reset the EMA
    # next time we get one — protects against stale state after long blinks
    # or after the ROI bootstrap re-acquires from a different head position.
    dx_ema_stale_sec: float = 1.0

    # --- Smoothing: hysteresis-based state machine ---
    # Switch to a new state only after this many CONSECUTIVE raw observations
    # of that state. Lower = more responsive, higher = more stable.
    # At 30 fps: 3 frames = 100 ms latency; at 10 fps: 3 frames = 300 ms.
    frames_to_switch: int = 3
    # Need this many CONSECUTIVE INVALID observations before declaring INVALID.
    # Higher = more tolerant of brief detection outages (blinks, saccades).
    frames_to_invalid: int = 6

    # --- Dwell-time gating (for "target lock") ---
    # Once the smoothed state has been LEFT or RIGHT continuously for this
    # many seconds, the result is marked `locked = True`. This corresponds
    # to the "凝视 1.5 秒 → 目标锁定" mechanism in the multimodal demo script.
    dwell_target_sec: float = 1.5

    # --- Display / runtime ---
    show_debug: bool = False
    show_trackbars: bool = False       # trackbars in a separate window (toggle with 't')


@dataclasses.dataclass
class DetectionResult:
    valid: bool
    label: str                          # "LEFT" | "CENTER" | "RIGHT" | "INVALID"
    dx_norm: float
    pupil_center: Optional[Tuple[float, float]]   # in ROI coords
    pupil_radius: float
    pupil_ellipse: Optional[Tuple[Tuple[float, float],
                                  Tuple[float, float], float]]
    glint_center: Optional[Tuple[float, float]]   # in ROI coords
    confidence: float
    reason: str = ""
    # Time spent continuously in the current smoothed state (seconds).
    # Reset to 0 whenever the smoothed state changes.
    dwell_seconds: float = 0.0
    # True once dwell_seconds >= dwell_target_sec AND state ∈ {LEFT, RIGHT}.
    # This is the signal a downstream fusion node should consume to "lock"
    # onto a side (e.g., dog/car) per the demo script.
    locked: bool = False


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class EyeDirectionDetector:
    LABELS = ("LEFT", "CENTER", "RIGHT", "INVALID")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # --- Hysteresis state machine (replaces the old majority-vote window) ---
        # current_state: the smoothed label we are currently outputting
        self._current_state: str = "INVALID"
        # pending_state / pending_count: how many consecutive frames of a
        # *different* state we've seen, building up evidence to switch
        self._pending_state: str = "INVALID"
        self._pending_count: int = 0
        # Time when the current state was first entered (for dwell tracking)
        self._state_enter_time: Optional[float] = None
        # --- dx_norm EMA state ---
        # Smoothed gaze value used as input to _classify(). None = not seeded
        # yet (cold start, or stale > cfg.dx_ema_stale_sec since last update).
        self._dx_ema: Optional[float] = None
        self._dx_ema_last_t: float = 0.0
        # calibration: dx_norm targets for the three classes (None → use deadband)
        self.calib = {"LEFT": None, "CENTER": None, "RIGHT": None}

    # ---------- glint detection ----------

    def _detect_glints(self, gray: np.ndarray):
        cfg = self.cfg
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (cfg.tophat_kernel, cfg.tophat_kernel))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, mask = cv2.threshold(tophat, cfg.tophat_thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: List[Tuple[float, float, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (cfg.glint_min_area <= area <= cfg.glint_max_area):
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            blobs.append((cx, cy, area))

        if not blobs:
            return None, mask

        # Keep the largest few; weight-average their centroids
        blobs.sort(key=lambda b: -b[2])
        blobs = blobs[: cfg.glint_max_blobs]
        wsum = sum(b[2] for b in blobs)
        gx = sum(b[0] * b[2] for b in blobs) / wsum
        gy = sum(b[1] * b[2] for b in blobs) / wsum
        return (gx, gy), mask

    # ---------- pupil detection ----------

    def _detect_pupil(self, gray: np.ndarray, glint_mask: np.ndarray,
                      glint_center: Optional[Tuple[float, float]] = None):
        """
        Detect the pupil. CRITICAL: the pupil is always at-or-very-near the
        glints (glints are corneal reflections; the cornea sits over the pupil).
        We therefore search in a window AROUND the glints, not globally — this
        prevents grabbing the much larger iris as the "pupil".
        """
        cfg = self.cfg
        h, w = gray.shape

        # Fill glints with neighbourhood minimum so they don't punch holes
        eroded = cv2.erode(gray, np.ones((cfg.erode_kernel, cfg.erode_kernel), np.uint8))
        gray_no_glint = np.where(glint_mask > 0, eroded, gray).astype(np.uint8)
        blurred = cv2.GaussianBlur(gray_no_glint, (7, 7), 0)

        # Build the search ROI for pupil: a window centred on the glints.
        # If glints not detected, fall back to the full frame.
        if glint_center is not None:
            gx, gy = glint_center
            r_search = cfg.pupil_search_radius
            x0 = max(0, int(gx - r_search))
            y0 = max(0, int(gy - r_search))
            x1 = min(w, int(gx + r_search))
            y1 = min(h, int(gy + r_search))
        else:
            x0, y0, x1, y1 = 0, 0, w, h

        sub = blurred[y0:y1, x0:x1]
        if sub.size == 0:
            return None, np.zeros_like(gray)

        # Min-relative threshold: pupil is the set of pixels within
        # `pupil_thr_offset` gray levels of the darkest pixel in the window.
        # This guarantees we capture the pupil core but not the iris around it.
        min_val = float(sub.min())
        thr_val = min_val + cfg.pupil_thr_offset
        _, sub_dark = cv2.threshold(sub, thr_val, 255, cv2.THRESH_BINARY_INV)
        # MORPH_CLOSE only — fill any tiny holes left by glint inpainting
        sub_dark = cv2.morphologyEx(
            sub_dark, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        # Re-embed the sub-mask into a full-size mask for return/visualisation
        dark = np.zeros_like(gray)
        dark[y0:y1, x0:x1] = sub_dark

        contours, _ = cv2.findContours(sub_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        best = None
        best_score = 0.0
        for c in contours:
            if len(c) < 8:
                continue
            area = cv2.contourArea(c)
            if not (cfg.pupil_min_area <= area <= cfg.pupil_max_area):
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4.0 * np.pi * area / (peri * peri)
            if circ < cfg.pupil_min_circularity:
                continue
            try:
                (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
            except cv2.error:
                continue
            ar = min(ax1, ax2) / max(ax1, ax2) if max(ax1, ax2) > 0 else 0.0
            if ar < cfg.pupil_min_aspect:
                continue

            # Translate to original image coordinates
            cx_g = cx + x0
            cy_g = cy + y0

            # Score: prefer larger, rounder, and CLOSER TO GLINT
            if glint_center is not None:
                d_glint = np.hypot(cx_g - glint_center[0], cy_g - glint_center[1])
                # exponential decay over a "pupil-radius-ish" length scale
                glint_pull = float(np.exp(-d_glint / max(20.0, cfg.pupil_search_radius * 0.4)))
            else:
                glint_pull = 1.0
            score = circ * np.sqrt(area) * glint_pull

            if score > best_score:
                best_score = score
                best = ((cx_g, cy_g), max(ax1, ax2) / 2.0,
                        ((cx_g, cy_g), (ax1, ax2), ang),
                        circ, ar, area)

        return best, dark

    # ---------- decision ----------

    def _classify(self, dx_norm: float) -> str:
        cL, cC, cR = self.calib["LEFT"], self.calib["CENTER"], self.calib["RIGHT"]
        if cL is not None and cC is not None and cR is not None:
            # mid-points between center-left and center-right
            thr_L = cC - 0.5 * (cC - cL)
            thr_R = cC + 0.5 * (cR - cC)
            if dx_norm < thr_L:
                return "LEFT"
            if dx_norm > thr_R:
                return "RIGHT"
            return "CENTER"
        # uncalibrated deadband
        d = self.cfg.deadband
        if dx_norm < -d:
            return "LEFT"
        if dx_norm > d:
            return "RIGHT"
        return "CENTER"

    def _smooth(self, raw: str, now: float) -> str:
        """
        Hysteresis state machine. Switch to a new state only after
        cfg.frames_to_switch consecutive observations of it. INVALID raw
        observations need cfg.frames_to_invalid to flip the output to
        INVALID — this absorbs brief detection outages (blinks, saccades).
        """
        cfg = self.cfg
        if raw == self._current_state:
            # confirming current state — clear pending count
            self._pending_state = self._current_state
            self._pending_count = 0
            return self._current_state

        # raw is different from current — accumulate evidence
        if raw == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw
            self._pending_count = 1

        threshold = (cfg.frames_to_invalid if raw == "INVALID"
                     else cfg.frames_to_switch)

        if self._pending_count >= threshold:
            # commit the switch
            self._current_state = self._pending_state
            self._pending_count = 0
            self._state_enter_time = now

        return self._current_state

    def _build_result(self, raw: str, now: float, dx_norm: float, confidence: float,
                      pupil_center, pupil_r, ellipse, glint_center,
                      reason: str = "") -> DetectionResult:
        """Apply smoothing + dwell tracking and return DetectionResult."""
        smoothed = self._smooth(raw, now)
        # Dwell: time spent in the current smoothed state since entering it.
        if self._state_enter_time is None:
            self._state_enter_time = now
        dwell = max(0.0, now - self._state_enter_time)
        locked = (smoothed in ("LEFT", "RIGHT")
                  and dwell >= self.cfg.dwell_target_sec)
        return DetectionResult(
            valid=(smoothed != "INVALID"),
            label=smoothed,
            dx_norm=dx_norm,
            pupil_center=pupil_center,
            pupil_radius=pupil_r,
            pupil_ellipse=ellipse,
            glint_center=glint_center,
            confidence=confidence,
            reason=reason,
            dwell_seconds=dwell,
            locked=locked,
        )

    # ---------- public API ----------

    def process(self, roi_bgr: np.ndarray,
                roi_origin: Tuple[int, int] = (0, 0)) -> DetectionResult:
        cfg = self.cfg
        now = time.time()

        if roi_bgr is None or roi_bgr.size == 0:
            return self._build_result("INVALID", now, 0.0, 0.0,
                                      None, 0.0, None, None, "empty roi")

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        glint_center, _glint_mask = self._detect_glints(gray)
        pupil_pack, _dark_mask = self._detect_pupil(gray, _glint_mask, glint_center)

        if pupil_pack is None:
            return self._build_result("INVALID", now, 0.0, 0.0,
                                      None, 0.0, None, glint_center, "pupil not found")

        pupil_center, pupil_r, ellipse, circ, ar, area = pupil_pack
        conf_pupil = 0.5 * circ + 0.5 * ar
        conf_glint = 1.0 if glint_center is not None else 0.0
        confidence = 0.6 * conf_pupil + 0.4 * conf_glint

        if glint_center is None:
            return self._build_result("INVALID", now, 0.0, conf_pupil,
                                      pupil_center, pupil_r, ellipse, None,
                                      "glint not found")

        if confidence < cfg.invalid_below_conf:
            return self._build_result("INVALID", now, 0.0, confidence,
                                      pupil_center, pupil_r, ellipse, glint_center,
                                      f"low confidence ({confidence:.2f})")

        dx = pupil_center[0] - glint_center[0]
        dx_norm = dx / pupil_r if pupil_r > 1e-3 else 0.0

        # --- EMA-smooth dx_norm before classification ---
        # All branches above here either returned INVALID or have confidence
        # >= invalid_below_conf, so any dx_norm reaching this point is from
        # a frame whose pupil+glint both passed quality gates.
        if (self._dx_ema is None
                or (now - self._dx_ema_last_t) > cfg.dx_ema_stale_sec):
            # Cold start, or last good update was too long ago to trust
            self._dx_ema = dx_norm
        else:
            a = cfg.dx_ema_alpha
            self._dx_ema = a * dx_norm + (1.0 - a) * self._dx_ema
        self._dx_ema_last_t = now

        # Classify using the SMOOTHED value, but report it as dx_norm in the
        # result so downstream sees what the classifier actually saw.
        raw = self._classify(self._dx_ema)
        return self._build_result(raw, now, self._dx_ema, confidence,
                                  pupil_center, pupil_r, ellipse, glint_center)

    # ---------- calibration management ----------

    def add_calib_sample(self, label: str, dx_norm: float, samples: List[float]):
        """Caller appends dx_norm to samples, then calls finalize_calib."""
        samples.append(dx_norm)

    def finalize_calib(self, label: str, samples: List[float]) -> bool:
        if len(samples) < 5:
            return False
        # use the median to be robust against blink/saccade outliers
        self.calib[label] = float(np.median(samples))
        return True

    def calib_thresholds(self) -> Optional[Tuple[float, float]]:
        cL, cC, cR = self.calib["LEFT"], self.calib["CENTER"], self.calib["RIGHT"]
        if None in (cL, cC, cR):
            return None
        return (cC - 0.5 * (cC - cL), cC + 0.5 * (cR - cC))

    def save_calib(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.calib, f, indent=2)

    def load_calib(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        for k in ("LEFT", "CENTER", "RIGHT"):
            v = data.get(k, None)
            self.calib[k] = float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def list_cameras(max_idx: int = 8) -> List[int]:
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                found.append(i)
        cap.release()
    return found


def open_video_source(args) -> cv2.VideoCapture:
    """Open a camera with the right backend for the current OS."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {args.video}")
        return cap

    # Pick the right backend per OS:
    # - Windows: DirectShow (CAP_DSHOW) is the most reliable for USB UVC cams
    # - Linux/Jetson: V4L2 is the native UVC path
    # - macOS / others: let OpenCV auto-pick
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = 0  # auto

    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}. "
                           f"Try --list to see available indices.")

    # Resolution. On Jetson Nano 2GB, lowering to 640x480 noticeably reduces
    # CPU load — pass --width/--height to override these defaults.
    target_w = getattr(args, "width", 1280) or 1280
    target_h = getattr(args, "height", 720) or 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    # Smaller buffer reduces input lag and the chance of stale frames
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass
    return cap


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "LEFT":    (0, 220, 255),    # yellow-orange
    "RIGHT":   (255, 180, 0),    # cyan-blue
    "CENTER":  (0, 255, 0),      # green
    "INVALID": (60, 60, 220),    # red
}


def draw_overlay(frame: np.ndarray,
                 roi_origin: Tuple[int, int],
                 roi_size: Tuple[int, int],
                 result: DetectionResult,
                 fps: float,
                 calibrated: bool,
                 calib_state: Optional[str],
                 cfg: Optional[Config] = None,
                 calib_data: Optional[dict] = None) -> None:
    H, W = frame.shape[:2]
    x0, y0 = roi_origin
    rw, rh = roi_size

    # --- ROI rectangle + pupil/glint markers ---
    cv2.rectangle(frame, (x0, y0), (x0 + rw, y0 + rh), (200, 200, 0), 2)

    if result.pupil_ellipse is not None:
        (cx, cy), (ax1, ax2), ang = result.pupil_ellipse
        cx_g, cy_g = int(cx + x0), int(cy + y0)
        try:
            cv2.ellipse(frame, (cx_g, cy_g),
                        (max(1, int(ax1 / 2)), max(1, int(ax2 / 2))),
                        ang, 0, 360, (0, 255, 0), 2)
        except cv2.error:
            pass
        cv2.drawMarker(frame, (cx_g, cy_g), (0, 0, 255),
                       cv2.MARKER_CROSS, 24, 2)

    if result.glint_center is not None:
        gx, gy = int(result.glint_center[0] + x0), int(result.glint_center[1] + y0)
        cv2.circle(frame, (gx, gy), 8, (0, 255, 255), 2)
        cv2.drawMarker(frame, (gx, gy), (0, 255, 255),
                       cv2.MARKER_TRIANGLE_UP, 18, 2)
        if result.pupil_center is not None:
            px, py = int(result.pupil_center[0] + x0), int(result.pupil_center[1] + y0)
            cv2.arrowedLine(frame, (gx, gy), (px, py), (0, 0, 255), 2, tipLength=0.3)

    # --- Top-left status panel ---
    color = LABEL_COLORS.get(result.label, (255, 255, 255))
    cv2.rectangle(frame, (10, 10), (440, 95), (20, 20, 20), -1)
    cv2.putText(frame, f"GAZE: {result.label}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame,
                f"dx_norm={result.dx_norm:+.2f}  conf={result.confidence:.2f}  fps={fps:4.1f}",
                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame,
                "CALIBRATED" if calibrated else "uncalibrated (default deadband)",
                (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 255, 180) if calibrated else (180, 180, 180), 1)

    # --- Calibration prompt banner ---
    if calib_state is not None:
        cv2.rectangle(frame, (10, 105), (440, 140), (10, 10, 80), -1)
        cv2.putText(frame, f"CALIB: {calib_state}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 2)

    # --- Bottom-center gaze-axis bar ---
    if cfg is not None:
        _draw_gaze_axis(frame, result, cfg, calib_data)

    # --- Dwell-time progress (when in LEFT or RIGHT) ---
    if result.label in ("LEFT", "RIGHT") and cfg is not None:
        _draw_dwell_bar(frame, result, cfg.dwell_target_sec)

    # --- LOCKED banner (big, central) ---
    if result.locked:
        _draw_locked_banner(frame, result.label)


def _draw_gaze_axis(frame: np.ndarray, result: DetectionResult,
                    cfg: Config, calib_data: Optional[dict]) -> None:
    """Bottom-center horizontal bar showing where dx_norm sits between LEFT and RIGHT."""
    H, W = frame.shape[:2]
    bar_w = 480
    bar_h = 26
    bar_x = W // 2 - bar_w // 2
    bar_y = H - 80

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (35, 35, 35), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (90, 90, 90), 1)

    # Determine left/right thresholds and the dx_norm range to display
    if calib_data and all(v is not None for v in calib_data.values()):
        lo = float(calib_data["LEFT"]) * 1.2
        hi = float(calib_data["RIGHT"]) * 1.2
        thr_l = calib_data["CENTER"] - 0.5 * (calib_data["CENTER"] - calib_data["LEFT"])
        thr_r = calib_data["CENTER"] + 0.5 * (calib_data["RIGHT"] - calib_data["CENTER"])
    else:
        lo, hi = -2.0, 2.0
        thr_l, thr_r = -cfg.deadband, cfg.deadband

    def to_x(v: float) -> int:
        v = max(lo, min(hi, v))
        return int(bar_x + (v - lo) / (hi - lo) * bar_w)

    # Dead-zone (CENTER) shaded green
    cv2.rectangle(frame, (to_x(thr_l), bar_y + 2),
                  (to_x(thr_r), bar_y + bar_h - 2), (15, 60, 15), -1)
    # Threshold tick marks
    for v, lbl in [(thr_l, "L"), (thr_r, "R")]:
        x = to_x(v)
        cv2.line(frame, (x, bar_y), (x, bar_y + bar_h), (200, 200, 200), 1)

    # Marker for current dx_norm
    if result.valid or result.glint_center is not None:
        mx = to_x(result.dx_norm)
        cv2.line(frame, (mx, bar_y - 6), (mx, bar_y + bar_h + 6),
                 LABEL_COLORS.get(result.label, (255, 255, 255)), 3)

    # Labels under the bar
    cv2.putText(frame, "LEFT", (bar_x, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    cv2.putText(frame, "CENTER", (bar_x + bar_w // 2 - 30, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
    cv2.putText(frame, "RIGHT", (bar_x + bar_w - 50, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 180), 1)


def _draw_dwell_bar(frame: np.ndarray, result: DetectionResult,
                    target_sec: float) -> None:
    """Progress bar showing dwell-time accumulating toward `target_sec`."""
    H, W = frame.shape[:2]
    bar_w = 280
    bar_h = 14
    bar_x = W // 2 - bar_w // 2
    bar_y = H - 130

    pct = max(0.0, min(1.0, result.dwell_seconds / target_sec))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * pct), bar_y + bar_h),
                  LABEL_COLORS.get(result.label, (200, 200, 200)), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
    cv2.putText(frame,
                f"DWELL  {result.dwell_seconds:0.2f} / {target_sec:0.2f} s",
                (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 220), 1)


def _draw_locked_banner(frame: np.ndarray, label: str) -> None:
    """Big central banner shown once dwell threshold is met."""
    H, W = frame.shape[:2]
    text = f"LOCKED: {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    pad = 20
    x0 = W // 2 - tw // 2 - pad
    y0 = H // 2 - th // 2 - pad
    x1 = W // 2 + tw // 2 + pad
    y1 = H // 2 + th // 2 + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 80, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(frame, text, (x0 + pad, y1 - pad - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (200, 255, 200), 4)


def draw_keys_help(frame: np.ndarray) -> None:
    h = frame.shape[0]
    lines = [
        "Calib: 1=LEFT  2=CENTER  3=RIGHT (hold ~1.2 s)   s save   o load   x clear",
        "t trackbars   d debug   r reselect ROI   q/ESC quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, h - 30 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Trackbars
# ---------------------------------------------------------------------------

WIN_MAIN = "Eye Direction"
WIN_TRACKBARS = "Eye Direction — Tuning"


def setup_trackbars(cfg: Config) -> None:
    """Create a separate small window holding the tuning trackbars."""
    cv2.namedWindow(WIN_TRACKBARS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TRACKBARS, 480, 280)

    def nop(_):
        pass
    cv2.createTrackbar("tophat_thr",     WIN_TRACKBARS, cfg.tophat_thr,           80, nop)
    cv2.createTrackbar("pupil_offset",   WIN_TRACKBARS, cfg.pupil_thr_offset,     80, nop)
    cv2.createTrackbar("search_r",       WIN_TRACKBARS, cfg.pupil_search_radius, 200, nop)
    cv2.createTrackbar("min_circ x100",  WIN_TRACKBARS,
                       int(cfg.pupil_min_circularity * 100), 80, nop)
    cv2.createTrackbar("deadband x100",  WIN_TRACKBARS,
                       int(cfg.deadband * 100), 100, nop)
    cv2.createTrackbar("dwell_x100",     WIN_TRACKBARS,
                       int(cfg.dwell_target_sec * 100), 500, nop)


def teardown_trackbars() -> None:
    try:
        cv2.destroyWindow(WIN_TRACKBARS)
    except cv2.error:
        pass


def read_trackbars(cfg: Config) -> None:
    """Read trackbar positions; silently no-op if the window isn't there."""
    try:
        cfg.tophat_thr            = max(1,    cv2.getTrackbarPos("tophat_thr", WIN_TRACKBARS))
        cfg.pupil_thr_offset      = max(5,    cv2.getTrackbarPos("pupil_offset", WIN_TRACKBARS))
        cfg.pupil_search_radius   = max(30,   cv2.getTrackbarPos("search_r", WIN_TRACKBARS))
        cfg.pupil_min_circularity = max(0.05, cv2.getTrackbarPos("min_circ x100", WIN_TRACKBARS) / 100.0)
        cfg.deadband              = max(0.05, cv2.getTrackbarPos("deadband x100", WIN_TRACKBARS) / 100.0)
        cfg.dwell_target_sec      = max(0.10, cv2.getTrackbarPos("dwell_x100", WIN_TRACKBARS) / 100.0)
    except cv2.error:
        pass


# ---------------------------------------------------------------------------
# ROI selection
# ---------------------------------------------------------------------------

def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    print()
    print("=" * 60)
    print("ROI selection — three steps:")
    print("  1. CLICK on the top-left corner of the eye")
    print("  2. DRAG to the bottom-right corner (rectangle appears)")
    print("  3. Release, then press SPACE or ENTER to confirm")
    print("Press 'c' to cancel and use a default centered ROI.")
    print("=" * 60)
    r = cv2.selectROI(WIN_MAIN, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = (int(v) for v in r)
    if w == 0 or h == 0:
        H, W = frame.shape[:2]
        print(f"No box drawn — using default centered ROI ({W//4},{H//4},{W//2},{H//2}).")
        return (W // 4, H // 4, W // 2, H // 2)
    print(f"ROI selected: ({x},{y},{w},{h})")
    return (x, y, w, h)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--select-on-start", action="store_true")
    parser.add_argument("--video", type=str, default=None,
                        help="use video file instead of camera (offline test)")
    parser.add_argument("--calib", type=str, default=None,
                        help="path to calibration json to load on startup")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-gui", action="store_true",
                        help="headless mode — no windows. Prints state changes "
                             "to stdout. Use for ROS 2 / Jetson deployment.")
    parser.add_argument("--width",  type=int, default=1280,
                        help="camera width  (default 1280; try 640 on Jetson)")
    parser.add_argument("--height", type=int, default=720,
                        help="camera height (default 720;  try 480 on Jetson)")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI as 'x,y,w,h' (skip selectROI; useful for "
                             "headless mode)")
    args = parser.parse_args()

    if args.list:
        cams = list_cameras()
        print("Available cameras:", cams)
        return

    cap = open_video_source(args)
    cfg = Config()
    detector = EyeDirectionDetector(cfg)
    if args.calib and os.path.isfile(args.calib):
        detector.load_calib(args.calib)
        print(f"Loaded calibration from {args.calib}: {detector.calib}")

    if not args.no_gui:
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_MAIN, 1280, 720)
        # Trackbars start hidden — toggle with 't'. This keeps the main
        # window clean and improves rendering speed.
        # setup_trackbars(cfg) is called when user presses 't'.

    # ROI
    roi_xywh = None
    # Pre-parsed ROI from CLI (so headless mode doesn't need a GUI)
    if args.roi:
        try:
            parts = [int(v) for v in args.roi.split(",")]
            if len(parts) == 4:
                roi_xywh = tuple(parts)
                print(f"ROI from --roi: {roi_xywh}")
        except ValueError:
            print(f"WARN: --roi='{args.roi}' is not in 'x,y,w,h' format; ignoring.")

    if args.select_on_start and roi_xywh is None:
        if args.no_gui:
            print("ERROR: --select-on-start needs a GUI; pass --roi 'x,y,w,h' "
                  "for headless mode.")
            return
        # Warm up the camera — many USB webcams return 1-10 black frames at
        # the very start. Read up to 30 frames over ~1 second until we get
        # one that's actually got pixel content.
        print("Warming up camera...")
        frame = None
        for attempt in range(30):
            ret, candidate = cap.read()
            if ret and candidate is not None and candidate.size > 0:
                if candidate.mean() > 5.0:
                    frame = candidate
                    print(f"Camera ready after {attempt + 1} frame(s).")
                    break
            time.sleep(0.05)
        if frame is None:
            print("ERROR: camera produced no usable frames.")
            print("  - Check the camera index with: python eye_direction_detector.py --list")
            print("  - If your laptop's built-in camera is 0 and USB is 1, try the other.")
            print("  - Make sure no other app (Zoom, Teams, browser) is holding the camera.")
            return
        if args.mirror:
            frame = cv2.flip(frame, 1)
        roi_xywh = select_roi(frame)
        print("ROI:", roi_xywh)

    # State
    last_t = time.time()
    fps = 0.0
    calib_state: Optional[str] = None
    calib_label: Optional[str] = None
    calib_samples: List[float] = []
    calib_t_start: float = 0.0
    CALIB_DURATION = 1.2  # seconds
    # Headless mode bookkeeping — only print on state CHANGES
    last_printed_label: str = ""
    last_printed_locked: bool = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        # ROI
        if roi_xywh is None:
            H, W = frame.shape[:2]
            roi_xywh = (W // 4, H // 4, W // 2, H // 2)
        x, y, w, h = roi_xywh
        x = max(0, min(x, frame.shape[1] - 2))
        y = max(0, min(y, frame.shape[0] - 2))
        w = max(2, min(w, frame.shape[1] - x))
        h = max(2, min(h, frame.shape[0] - y))
        roi_bgr = frame[y:y + h, x:x + w]

        # Detect
        result = detector.process(roi_bgr, (x, y))

        # Calibration sampling
        if calib_label is not None:
            if result.valid:
                calib_samples.append(result.dx_norm)
            elapsed = time.time() - calib_t_start
            calib_state = (f"{calib_label}: {len(calib_samples)} samples, "
                           f"{max(0.0, CALIB_DURATION - elapsed):.1f}s left")
            if elapsed >= CALIB_DURATION:
                ok = detector.finalize_calib(calib_label, calib_samples)
                if ok:
                    print(f"Calibrated {calib_label} = {detector.calib[calib_label]:+.3f}")
                    calib_state = f"{calib_label} done = {detector.calib[calib_label]:+.3f}"
                else:
                    print(f"Calib {calib_label} failed: only {len(calib_samples)} samples")
                    calib_state = f"{calib_label} FAILED (need more valid samples)"
                calib_label = None
                calib_samples = []
        else:
            if calib_state is not None and time.time() - calib_t_start > CALIB_DURATION + 1.5:
                calib_state = None

        # FPS
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        calibrated = all(v is not None for v in detector.calib.values())

        # Headless: just print state changes and locked events; skip rendering
        if args.no_gui:
            if result.label != last_printed_label or (result.locked and not last_printed_locked):
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"label={result.label:<7} dx={result.dx_norm:+.2f} "
                      f"conf={result.confidence:.2f} dwell={result.dwell_seconds:.2f}s "
                      f"locked={result.locked}")
                last_printed_label = result.label
                last_printed_locked = result.locked
            continue

        draw_overlay(frame, (x, y), (w, h), result, fps, calibrated,
                     calib_state, cfg=cfg, calib_data=detector.calib)
        draw_keys_help(frame)

        # Optional debug panel
        if cfg.show_debug:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gc, gm = detector._detect_glints(gray)
            _, dm = detector._detect_pupil(gray, gm, gc)
            small_h = 240
            scale = small_h / max(1, gm.shape[0])
            gm_s = cv2.resize(cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale)
            dm_s = cv2.resize(cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale)
            cv2.putText(gm_s, "glint_mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(dm_s, "dark_mask",  (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            xb = frame.shape[1] - gm_s.shape[1] - 10
            yb = 110
            try:
                frame[yb:yb + gm_s.shape[0], xb:xb + gm_s.shape[1]] = gm_s
                yb2 = yb + gm_s.shape[0] + 10
                frame[yb2:yb2 + dm_s.shape[0], xb:xb + dm_s.shape[1]] = dm_s
            except ValueError:
                pass

        cv2.imshow(WIN_MAIN, frame)
        # Read trackbars HERE (after imshow), so the trackbar window keeps
        # repainting even when we're slow.
        if cfg.show_trackbars:
            read_trackbars(cfg)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:                       # no key
            continue
        elif key in (27, ord('q')):          # ESC / q  → quit
            break
        elif key == ord('r'):                # reselect ROI
            roi_xywh = select_roi(frame)
        elif key == ord('d'):                # toggle debug masks
            cfg.show_debug = not cfg.show_debug
        elif key == ord('t'):                # toggle trackbars window
            cfg.show_trackbars = not cfg.show_trackbars
            if cfg.show_trackbars:
                setup_trackbars(cfg)
            else:
                teardown_trackbars()
        elif key == ord('1'):                # 1 = calibrate LEFT
            calib_label = "LEFT"
            calib_samples = []
            calib_t_start = time.time()
            print("Calibrating LEFT — keep looking hard left for 1.2 s ...")
        elif key == ord('2'):                # 2 = calibrate CENTER
            calib_label = "CENTER"
            calib_samples = []
            calib_t_start = time.time()
            print("Calibrating CENTER — keep looking straight for 1.2 s ...")
        elif key == ord('3'):                # 3 = calibrate RIGHT
            calib_label = "RIGHT"
            calib_samples = []
            calib_t_start = time.time()
            print("Calibrating RIGHT — keep looking hard right for 1.2 s ...")
        elif key == ord('x'):                # clear calibration
            detector.calib = {"LEFT": None, "CENTER": None, "RIGHT": None}
            print("Calibration cleared.")
        elif key == ord('s'):                # save calibration
            detector.save_calib("eye_calib.json")
            print("Saved calibration to eye_calib.json:", detector.calib)
        elif key == ord('o'):                # load calibration
            try:
                detector.load_calib("eye_calib.json")
                print("Loaded calibration:", detector.calib)
            except FileNotFoundError:
                print("No eye_calib.json found.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()