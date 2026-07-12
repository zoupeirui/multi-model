"""
eye_direction_node.py
=====================
ROS 2 node wrapping the EyeDirectionDetector with:
  * an ADAPTIVE ROI tracker that follows the pupil across the frame
  * a ROS-driven CALIBRATION flow (no GUI needed)

What changed vs the previous version
------------------------------------
The hard-coded `roi` parameter is replaced with TWO ROI concepts:

  * `bootstrap_roi` — a wide rectangle that's used until we first lock onto
    a pupil, and again whenever we've lost the pupil for too long.

  * `tracking_roi_wh` — the (width, height) of the *tight* ROI that follows
    the last detected pupil.

Calibration is driven entirely over a ROS topic now. Send a String to
`/eye/calibrate` and the node samples dx_norm for 2 s, then writes the
median into the detector's LEFT/CENTER/RIGHT slots. Once all three are
set, `_classify()` uses them instead of the fixed `deadband`.

Calibrate (look the named direction during each 2-second window):
  ros2 topic pub --once /eye/calibrate std_msgs/String "data: LEFT"
  ros2 topic pub --once /eye/calibrate std_msgs/String "data: CENTER"
  ros2 topic pub --once /eye/calibrate std_msgs/String "data: RIGHT"
  ros2 topic pub --once /eye/calibrate std_msgs/String "data: SAVE"
  # Optional:
  ros2 topic pub --once /eye/calibrate std_msgs/String "data: CLEAR"

After SAVE the file is at `calib_path` (default eye_calib.json in CWD).
Set `-p calib_path:=/some/path.json` to auto-load it on next startup.

Watch progress: `ros2 topic echo /eye/calibrate/status`

The `EyeDirectionDetector` itself now also has dx_norm EMA smoothing
(`cfg.dx_ema_alpha`) plus tighter glint thresholds — both reduce
classification jitter and don't need any node-side change.

Publishes
---------
  /eye/direction          std_msgs/String   "LEFT" | "CENTER" | "RIGHT" | "INVALID"
  /eye/locked             std_msgs/String   "" | "LEFT" | "RIGHT"
  /eye/calibrate/status   std_msgs/String   short progress messages

Parameters
----------
  camera             int    camera index                       (default 0)
  width              int    requested capture width            (default 1280)
  height             int    requested capture height           (default 720)
  mirror             bool   horizontal flip                    (default True)
  bootstrap_roi      int[4] [x,y,w,h] wide ROI for cold start  (default [150,200,500,320])
  tracking_roi_wh    int[2] [w,h] of tight tracking ROI        (default [360,260])
  miss_to_reset     int     consecutive misses → bootstrap     (default 30)
  calib_path        str     JSON calibration file              (default "eye_calib.json")
  publish_rate_hz   float   detection/publish rate             (default 30.0)
  dwell_target_sec  float   dwell threshold for `locked=True`  (default 1.5)
"""

import time
import threading
from typing import Optional, Tuple

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from dog_fusion_control.eye_direction_detector import (
    EyeDirectionDetector, Config, DetectionResult
)


# ---------------------------------------------------------------------------
# Calibrator — drives the LEFT / CENTER / RIGHT sampling over a ROS topic
# ---------------------------------------------------------------------------

class Calibrator:
    """Collects dx_norm samples for a fixed duration, then commits via
    `detector.finalize_calib()`. Feed every frame's DetectionResult into
    `feed()`; call `tick()` once per frame to check for completion.

    Designed for ROS message-driven control (no GUI). The node publishes
    progress messages from outside this class.
    """

    DURATION_SEC = 2.0   # 2 s so we still get >= 10 samples at 10 fps
    MIN_SAMPLES  = 5     # detector.finalize_calib needs >= 5

    def __init__(self, detector: EyeDirectionDetector):
        self.detector = detector
        self._active_label: Optional[str] = None
        self._samples: list = []
        self._start_time: float = 0.0

    @property
    def active(self) -> bool:
        return self._active_label is not None

    @property
    def active_label(self) -> Optional[str]:
        return self._active_label

    @property
    def time_remaining(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, self.DURATION_SEC - (time.time() - self._start_time))

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def is_complete(self) -> bool:
        """True when all three of LEFT / CENTER / RIGHT have been set."""
        return all(v is not None for v in self.detector.calib.values())

    def start(self, label: str) -> bool:
        label = label.upper().strip()
        if label not in ("LEFT", "CENTER", "RIGHT"):
            return False
        self._active_label = label
        self._samples = []
        self._start_time = time.time()
        return True

    def feed(self, result: DetectionResult) -> None:
        """Append this frame's dx_norm if calibration is running and the
        frame passed the detector's own quality gates."""
        if not self.active:
            return
        if result.valid and -5.0 < result.dx_norm < 5.0:
            self._samples.append(float(result.dx_norm))

    def tick(self) -> Optional[Tuple[str, bool, int, Optional[float]]]:
        """If the collection window has elapsed, commit and return
        (label, success, n_samples, calibrated_value). Else return None."""
        if not self.active:
            return None
        if time.time() - self._start_time < self.DURATION_SEC:
            return None
        label = self._active_label
        n = len(self._samples)
        ok = self.detector.finalize_calib(label, self._samples)
        value = self.detector.calib[label] if ok else None
        self._active_label = None
        self._samples = []
        return (label, ok, n, value)

    def clear(self) -> None:
        """Cancel any running collection and wipe stored calibration."""
        self._active_label = None
        self._samples = []
        self.detector.calib = {"LEFT": None, "CENTER": None, "RIGHT": None}


# ---------------------------------------------------------------------------
# Adaptive ROI tracker
# ---------------------------------------------------------------------------

class AdaptiveROI:
    """Keeps a tight ROI centred on the most recent valid pupil detection.

    Two modes:
      * BOOTSTRAP  — return the wide cold-start rectangle. Used until the
                     first valid pupil, and again after `miss_to_reset`
                     consecutive misses.
      * TRACKING   — return a `tracking_size` rectangle centred on the last
                     known pupil position (clamped to the frame).

    All positions are in FULL-frame pixel coordinates.
    """

    def __init__(self,
                 bootstrap_roi: Tuple[int, int, int, int],
                 tracking_size: Tuple[int, int] = (360, 260),
                 miss_to_reset: int = 30):
        self.bootstrap = tuple(int(v) for v in bootstrap_roi)
        self.tw, self.th = int(tracking_size[0]), int(tracking_size[1])
        self.miss_to_reset = int(miss_to_reset)

        self._last_pupil_xy: Optional[Tuple[float, float]] = None
        self._miss_count: int = 0

    @property
    def tracking(self) -> bool:
        """True iff we're currently following the pupil (not in bootstrap)."""
        return (self._last_pupil_xy is not None
                and self._miss_count < self.miss_to_reset)

    @property
    def last_pupil_xy(self) -> Optional[Tuple[float, float]]:
        return self._last_pupil_xy

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def next_roi(self, frame_wh: Tuple[int, int]) -> Tuple[int, int, int, int]:
        W, H = frame_wh
        if not self.tracking:
            return self._clamp(self.bootstrap, W, H)
        cx, cy = self._last_pupil_xy
        x = int(round(cx - self.tw / 2.0))
        y = int(round(cy - self.th / 2.0))
        return self._clamp((x, y, self.tw, self.th), W, H)

    def update(self, result: DetectionResult,
               roi_origin: Tuple[int, int]) -> None:
        """Feed back the detector's result. Call exactly once per processed frame."""
        # We update only on smoothed-valid AND raw pupil present this frame.
        # `result.valid` reflects the smoothed state machine (handles blinks);
        # `result.pupil_center` is None whenever the raw detection failed this
        # frame even if the smoother is still reporting valid. We need both.
        if result.valid and result.pupil_center is not None:
            px, py = result.pupil_center           # in ROI coords
            ox, oy = roi_origin
            self._last_pupil_xy = (px + ox, py + oy)
            self._miss_count = 0
        else:
            self._miss_count += 1
            # Note: we deliberately do NOT clear _last_pupil_xy on a single
            # miss. Bootstrap kicks in only when miss_count >= miss_to_reset.

    def force_reset(self) -> None:
        """Manually drop back to bootstrap (e.g., after a parameter change)."""
        self._last_pupil_xy = None
        self._miss_count = 0

    @staticmethod
    def _clamp(roi: Tuple[int, int, int, int],
               W: int, H: int) -> Tuple[int, int, int, int]:
        x, y, w, h = roi
        w = max(2, min(int(w), W))
        h = max(2, min(int(h), H))
        x = max(0, min(int(x), W - w))
        y = max(0, min(int(y), H - h))
        return (x, y, w, h)


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class EyeDirectionNode(Node):
    def __init__(self):
        super().__init__("eye_direction")

        # --- Parameters --------------------------------------------------
        self.declare_parameter("camera", 0)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("mirror", True)
        self.declare_parameter("bootstrap_roi", [150, 200, 500, 320])
        self.declare_parameter("tracking_roi_wh", [360, 260])
        self.declare_parameter("miss_to_reset", 30)
        self.declare_parameter("calib_path", "")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("dwell_target_sec", 1.5)

        cam_idx       = int(self.get_parameter("camera").value)
        req_w         = int(self.get_parameter("width").value)
        req_h         = int(self.get_parameter("height").value)
        self.mirror   = bool(self.get_parameter("mirror").value)
        bootstrap_roi = tuple(int(v) for v in self.get_parameter("bootstrap_roi").value)
        tracking_wh   = tuple(int(v) for v in self.get_parameter("tracking_roi_wh").value)
        miss_to_reset = int(self.get_parameter("miss_to_reset").value)
        calib_path    = str(self.get_parameter("calib_path").value)
        rate_hz       = float(self.get_parameter("publish_rate_hz").value)
        dwell_sec     = float(self.get_parameter("dwell_target_sec").value)

        # --- Detector ----------------------------------------------------
        cfg = Config()
        cfg.dwell_target_sec = dwell_sec
        self.detector = EyeDirectionDetector(cfg)
        self.calib_path = calib_path or "eye_calib.json"
        if calib_path:
            try:
                self.detector.load_calib(calib_path)
                self.get_logger().info(f"Loaded calibration: {self.detector.calib}")
            except FileNotFoundError:
                self.get_logger().info(
                    f"No calibration file at {calib_path} yet — using deadband. "
                    f"Send LEFT/CENTER/RIGHT/SAVE on /eye/calibrate to create one."
                )
            except Exception as e:
                self.get_logger().warn(f"Could not load {calib_path}: {e}")

        # --- Calibrator (ROS-driven) -----------------------------------
        self.calibrator = Calibrator(self.detector)

        # --- Adaptive ROI tracker ---------------------------------------
        self.roi_tracker = AdaptiveROI(
            bootstrap_roi=bootstrap_roi,
            tracking_size=tracking_wh,
            miss_to_reset=miss_to_reset,
        )
        # Only log on TRANSITIONS between bootstrap and tracking, not every tick
        self._was_tracking: bool = False

        # --- Camera (V4L2 on Linux/Jetson) ------------------------------
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {cam_idx}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency
        except cv2.error:
            pass

        # --- Publishers --------------------------------------------------
        self.pub_dir         = self.create_publisher(String, "/eye/direction",        10)
        self.pub_locked      = self.create_publisher(String, "/eye/locked",           10)
        self.pub_calib_state = self.create_publisher(String, "/eye/calibrate/status", 10)

        # --- Calibration command subscriber ----------------------------
        # Accepts String messages: "LEFT", "CENTER", "RIGHT", "SAVE", "CLEAR".
        # Usage:
        #   ros2 topic pub --once /eye/calibrate std_msgs/String "data: LEFT"
        #   (look hard left for 2 s)
        #   ros2 topic pub --once /eye/calibrate std_msgs/String "data: CENTER"
        #   ros2 topic pub --once /eye/calibrate std_msgs/String "data: RIGHT"
        #   ros2 topic pub --once /eye/calibrate std_msgs/String "data: SAVE"
        self.sub_calib = self.create_subscription(
            String, "/eye/calibrate", self._on_calib_cmd, 10
        )

        # --- Background reader thread (decouples capture from ROS timer) -
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # --- Detection timer --------------------------------------------
        self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(
            f"eye_direction up. cam={cam_idx} req={req_w}x{req_h} "
            f"bootstrap_roi={bootstrap_roi} tracking_wh={tracking_wh} "
            f"miss_to_reset={miss_to_reset} dwell={dwell_sec}s"
        )

    # -- background camera reader ---------------------------------------
    def _reader_loop(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._frame_lock:
                self._latest_frame = frame

    # -- per-tick processing --------------------------------------------
    def _on_timer(self):
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return

        H, W = frame.shape[:2]

        # Ask the tracker where to look this frame (clamped to actual frame size)
        x, y, w, h = self.roi_tracker.next_roi((W, H))
        roi_bgr = frame[y:y + h, x:x + w]

        # Run detection (detector is unchanged)
        result = self.detector.process(roi_bgr, (x, y))

        # Feed the result back into the tracker for next frame's ROI
        self.roi_tracker.update(result, (x, y))

        # Drive any running calibration collection. `tick()` returns a tuple
        # exactly on the frame the 2 s window elapses.
        self.calibrator.feed(result)
        done = self.calibrator.tick()
        if done is not None:
            label, ok, n, value = done
            if ok:
                self.get_logger().info(
                    f"Calibrated {label} = {value:+.3f} ({n} samples). "
                    f"calib so far: {self.detector.calib}"
                )
                self._publish_calib_status(
                    f"{label}_done={value:+.3f}  n={n}  "
                    f"complete={self.calibrator.is_complete()}"
                )
            else:
                self.get_logger().warn(
                    f"Calibration {label} FAILED — only {n} valid samples "
                    f"(need >= {self.calibrator.MIN_SAMPLES}). "
                    f"Make sure the pupil is being detected before retrying."
                )
                self._publish_calib_status(f"{label}_failed  n={n}")

        # Log only on bootstrap ↔ tracking transitions
        now_tracking = self.roi_tracker.tracking
        if now_tracking != self._was_tracking:
            if now_tracking:
                cx, cy = self.roi_tracker.last_pupil_xy
                self.get_logger().info(
                    f"ROI: pupil locked → tracking mode, centre=({cx:.0f},{cy:.0f})"
                )
            else:
                self.get_logger().warn(
                    f"ROI: pupil lost for {self.roi_tracker.miss_count} frames "
                    f"→ falling back to bootstrap_roi"
                )
            self._was_tracking = now_tracking

        # --- Publish ----------------------------------------------------
        msg = String()
        msg.data = result.label
        self.pub_dir.publish(msg)

        # /eye/locked carries the label only while truly locked, else empty
        locked_msg = String()
        locked_msg.data = result.label if result.locked else ""
        self.pub_locked.publish(locked_msg)

    # -- calibration command handler ------------------------------------
    def _on_calib_cmd(self, msg):
        cmd = (msg.data or "").upper().strip()
        if cmd in ("LEFT", "CENTER", "RIGHT"):
            if self.calibrator.start(cmd):
                self.get_logger().info(
                    f"Calibrating {cmd}: look that direction for "
                    f"{Calibrator.DURATION_SEC:.1f} s ..."
                )
                self._publish_calib_status(f"{cmd}_started")
            else:
                self.get_logger().warn(f"Invalid calibration label: {cmd}")
        elif cmd == "SAVE":
            if not self.calibrator.is_complete():
                self.get_logger().warn(
                    f"Calibration not complete yet. Have: {self.detector.calib}. "
                    f"Send LEFT / CENTER / RIGHT first."
                )
                self._publish_calib_status("save_incomplete")
                return
            try:
                self.detector.save_calib(self.calib_path)
                self.get_logger().info(
                    f"Saved calibration to {self.calib_path}: {self.detector.calib}"
                )
                self._publish_calib_status(f"saved {self.calib_path}")
            except Exception as e:
                self.get_logger().error(f"Save failed: {e}")
                self._publish_calib_status(f"save_failed: {e}")
        elif cmd == "CLEAR":
            self.calibrator.clear()
            self.get_logger().info("Calibration cleared.")
            self._publish_calib_status("cleared")
        else:
            self.get_logger().warn(
                f"Unknown /eye/calibrate command: '{cmd}'. "
                f"Valid: LEFT / CENTER / RIGHT / SAVE / CLEAR"
            )

    def _publish_calib_status(self, text: str) -> None:
        m = String()
        m.data = text
        self.pub_calib_state.publish(m)

    def destroy_node(self):
        self._stop.set()
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = EyeDirectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()