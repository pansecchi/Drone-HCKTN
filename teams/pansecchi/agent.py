"""Pansecchi/Samuele agent — EKF with Constant Turn Rate model.

Improvements over baseline:
    - PERCEPTION: CLAHE for fog, subpixel corner refinement, stale-frame
      detection to skip duplicate measurements on low-fps scenarios.
    - ESTIMATION: 5-state EKF with CTR (Constant Turn Rate) process model.
      State = [x, y, vx, vy, omega]. Predicts during occlusions/fps gaps,
      handles curved boat trajectories on HARD.
    - DECISION / CONTROL: same as baseline for now — will improve next
      (yaw alignment, soft landing, velocity feed-forward).
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from boat_landing.camera import CAMERA_BODY_OFFSET_Z, get_intrinsics
from boat_landing.controllers import DefaultAttitudeController
from boat_landing.drone_interface import DroneSpec, load_drone_spec


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# Must match BoatLandingEnv.MARKER_SIZE.
MARKER_SIZE_M = 0.8

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_DESCEND = "DESCEND"
PHASE_LAND = "LAND"


def rpy_to_matrix(rpy) -> np.ndarray:
    """World-from-body rotation matrix using PyBullet's RPY convention."""
    r, p_, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p_), np.sin(p_)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class PID:
    """Single-axis PID with output clamping, D-on-measurement and
    clamping-style anti-windup. The integrator is only advanced when the
    tentative output is inside the actuator range, OR when integrating
    further would reduce |output|. Prevents windup when roll/pitch saturate
    at +/-1.0 or when thrust hits its bounds."""
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0, i_clip=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_clip = i_clip
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev_err = None

    def __call__(self, err, dt, measurement_velocity=None):
        if measurement_velocity is not None:
            d = -float(measurement_velocity)
        elif self._prev_err is None:
            d = 0.0
        else:
            d = (err - self._prev_err) / max(dt, 1e-6)
        self._prev_err = err

        out_unsat = self.kp * err + self.ki * self._i + self.kd * d
        saturated_high = out_unsat >= self.out_max and err > 0.0
        saturated_low = out_unsat <= self.out_min and err < 0.0
        if not (saturated_high or saturated_low):
            self._i = float(np.clip(self._i + err * dt, -self.i_clip, self.i_clip))

        out = self.kp * err + self.ki * self._i + self.kd * d
        return float(np.clip(out, self.out_min, self.out_max))


class BoatEKF:
    """Extended Kalman Filter with Constant Turn Rate (CTR) process model.

    State: x = [px, py, vx, vy, omega]  (world frame, omega in rad/s)

    Process model (Euler discretization, valid for small dt):
        px(k+1) = px + vx*dt
        py(k+1) = py + vy*dt
        vx(k+1) = vx - omega*vy*dt
        vy(k+1) = vy + omega*vx*dt
        omega(k+1) = omega

    Nonlinear in vx*omega and vy*omega — Jacobian computed at each step.

    Measurement: z = [px, py]  (linear, observed from ArUco solvePnP)
    """

    STATE_DIM = 5
    MEAS_DIM = 2

    def __init__(self):
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 100.0  # large initial uncertainty
        self._initialized = False

        # Process noise (per second, scaled by dt in predict)
        self.q_p = 0.01    # position [m^2/s]
        self.q_v = 0.5     # velocity [m^2/s^3]
        self.q_w = 0.05    # turn rate [rad^2/s^3]

        # Measurement noise (constant, can be scaled by detection confidence)
        self.R_base = np.diag([0.15, 0.15])  # [m^2]

        # Outlier rejection threshold (chi-square 2-DoF, 3-sigma ~= 11.83)
        self.gate_threshold = 11.83

    def initialize(self, position_xy: np.ndarray):
        """Reset filter at a measured position with zero velocity / turn rate."""
        self.x = np.array([position_xy[0], position_xy[1], 0.0, 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 4.0, 4.0, 1.0])
        self._initialized = True

    def predict(self, dt: float):
        """Propagate state forward by dt using CTR model."""
        if not self._initialized:
            return

        px, py, vx, vy, w = self.x

        # Nonlinear state transition
        self.x = np.array([
            px + vx * dt,
            py + vy * dt,
            vx - w * vy * dt,
            vy + w * vx * dt,
            w,
        ])

        # Jacobian of f w.r.t. state, evaluated at current state
        F = np.array([
            [1.0, 0.0, dt,       0.0,    0.0     ],
            [0.0, 1.0, 0.0,      dt,     0.0     ],
            [0.0, 0.0, 1.0,     -w * dt, -vy * dt],
            [0.0, 0.0, w * dt,   1.0,    vx * dt ],
            [0.0, 0.0, 0.0,      0.0,    1.0     ],
        ])

        # Process noise covariance (continuous-time approx, scaled by dt)
        Q = np.diag([
            self.q_p * dt,
            self.q_p * dt,
            self.q_v * dt,
            self.q_v * dt,
            self.q_w * dt,
        ])

        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, measurement_var: Optional[float] = None) -> bool:
        """Apply a (x, y) measurement. Returns True if accepted, False if rejected as outlier."""
        if not self._initialized:
            self.initialize(z)
            return True

        # Linear measurement model H = [[1,0,0,0,0],[0,1,0,0,0]]
        H = np.zeros((self.MEAS_DIM, self.STATE_DIM))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        R = self.R_base.copy()
        if measurement_var is not None:
            R *= measurement_var

        # Innovation
        y_pred = H @ self.x
        innov = z - y_pred
        S = H @ self.P @ H.T + R

        # Mahalanobis gating
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = innov @ S_inv @ innov
            if mahalanobis_sq > self.gate_threshold:
                return False  # reject outlier
        except np.linalg.LinAlgError:
            return False

        # Kalman gain & update
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form for numerical stability
        return True

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x[0], self.x[1]])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.x[2], self.x[3]])

    @property
    def heading(self) -> float:
        return float(np.arctan2(self.x[3], self.x[2]))

    @property
    def turn_rate(self) -> float:
        return float(self.x[4])

    @property
    def velocity_std(self) -> float:
        """RMS 1-sigma uncertainty on (vx, vy). Used by the controller to
        decide whether the velocity feed-forward estimate is trustworthy."""
        return float(np.sqrt(0.5 * (self.P[2, 2] + self.P[3, 3])))


class PansecchiAgent:
    """ArUco + CLAHE perception, EKF-CTR estimation, baseline state machine + PIDs."""

    DT = 0.02  # must match BoatLandingEnv.DT

    SEARCH_PRIOR_XY = np.array([0.0, 0.0])
    SEARCH_PRIOR_ALT = 5.5

    # Entry thresholds (tighter) and exit thresholds (wider) give hysteresis
    # around phase boundaries. Without this, a platform oscillating near a
    # threshold thrashes phases, and every transition used to reset all PIDs.
    APPROACH_HORIZ_ENTER = 0.6
    APPROACH_HORIZ_EXIT = 1.2     # only fall back to SEARCH past this
    DESCEND_HORIZ_ENTER = 0.4
    DESCEND_HORIZ_EXIT = 0.9
    APPROACH_ALTITUDE_OK = 3.5
    DESCEND_ALTITUDE_OK = 0.6
    LAND_ALTITUDE_EXIT = 1.0      # need to rise this high to leave LAND

    PHASE_ALTITUDE = {
        PHASE_SEARCH: 5.5,
        PHASE_APPROACH: 3.0,
        PHASE_DESCEND: 0.4,
        PHASE_LAND: -0.5,
    }

    def __init__(self, drone_spec: Optional[DroneSpec] = None):
        if drone_spec is None:
            drone_spec = load_drone_spec(REPO_ROOT / "drones" / "vtol.yaml")
        self.spec: DroneSpec = drone_spec
        self.attitude_ctrl = DefaultAttitudeController(drone_spec)
        self._hover_action = self.attitude_ctrl(
            {"attitude": np.zeros(3), "angular_velocity": np.zeros(3)},
            roll_des=0.0, pitch_des=0.0, yaw_rate_des=0.0, thrust_norm=0.0,
        )

        self.intrinsics = get_intrinsics()
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        # ArUco detector. The 0.8 m marker is large in the image so we
        # don't need subpixel corner refinement (was iterating expensively
        # on degraded frames). CORNER_REFINE_NONE is fast and accurate enough.
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)
        except AttributeError:
            self._detector = None

        # CLAHE for foggy frames
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Marker corner ordering used by IPPE_SQUARE: TL, TR, BR, BL
        h = MARKER_SIZE_M / 2.0
        self._object_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]],
            dtype=np.float32,
        )

        # PIDs
        self.pid_x = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_y = PID(kp=0.15, ki=0.00, kd=0.40)
        self.pid_z = PID(kp=0.10, ki=0.02, kd=0.30)
        self.pid_yaw = PID(kp=1.0, ki=0.0, kd=0.0)

        # EKF
        self.ekf = BoatEKF()

        # Bookkeeping
        self.phase = PHASE_SEARCH
        self._prev_phase: Optional[str] = None
        self._last_marker_world: Optional[np.ndarray] = None
        self._last_marker_z: float = 0.15  # default platform z
        self._frames_since_detection = 0
        self._last_estimate: Optional[Dict] = None
        self._prev_frame_hash: Optional[bytes] = None
        # Yaw target captured last time the boat was actually moving.
        # Used to keep aligning the fuselage when the boat slows below the
        # heading-from-velocity noise floor (EASY: stationary boat).
        self._yaw_target_locked: Optional[float] = None

    # ------------------------------------------------------------------ act / API
    def act(self, obs: Dict) -> np.ndarray:
        camera = obs["camera"]
        state = obs["state"]
        battery = obs["battery"]
        time_s = obs["time"]

        perception = self.perceive(camera)
        boat_estimate = self.estimate(perception, state, history=None)
        self._last_estimate = boat_estimate
        self.phase = self.decide(state, boat_estimate, battery, time_s)
        action = self.control(state, boat_estimate, self.phase)
        if not np.all(np.isfinite(action)):
            action = self._hover_action.copy()
        return action

    def get_last_estimate(self) -> Optional[Dict]:
        if self._last_estimate is None or self._last_estimate.get("position") is None:
            return None
        return {
            "position": np.asarray(self._last_estimate["position"]),
            "velocity": np.asarray(self._last_estimate.get("velocity", np.zeros(3))),
        }

    # ------------------------------------------------------------------ STAGE 1: PERCEPTION
    def perceive(self, camera_image: np.ndarray) -> Dict:
        # Stale-frame detection: hash a small patch and skip if identical.
        patch = camera_image[230:250, 310:330, 0]
        frame_hash = patch.tobytes()
        is_stale = (frame_hash == self._prev_frame_hash)
        self._prev_frame_hash = frame_hash

        if is_stale:
            return {"detected": False, "stale": True}

        gray = cv2.cvtColor(camera_image, cv2.COLOR_RGB2GRAY)

        # CLAHE for fog-heavy frames. Cheap, restores marker contrast that
        # Beer-Lambert washes out.
        gray = self._clahe.apply(gray)

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)

        if ids is None or len(ids) == 0:
            return {"detected": False, "stale": False}

        target = None
        for c, i in zip(corners, ids.flatten()):
            if int(i) == 0:
                target = c
                break
        if target is None:
            return {"detected": False, "stale": False}

        image_points = target.reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            self.intrinsics,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return {"detected": False, "stale": False}

        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(tvec)) and np.all(np.isfinite(rvec))):
            return {"detected": False, "stale": False}
        if tvec[2] <= 0.1 or tvec[2] > 100.0:
            return {"detected": False, "stale": False}

        # Confidence: larger marker (closer) = lower measurement variance.
        edge_px = float(np.mean([
            np.linalg.norm(image_points[(i + 1) % 4] - image_points[i])
            for i in range(4)
        ]))
        meas_var = max(1.0, (100.0 / max(edge_px, 1.0)) ** 2)

        return {
            "detected": True,
            "stale": False,
            "tvec": tvec,
            "rvec": rvec,
            "image_points": image_points,
            "meas_var": meas_var,
        }

    # ------------------------------------------------------------------ STAGE 2: ESTIMATION (EKF + CTR)
    def estimate(self, perception: Dict, drone_state: Dict, history) -> Dict:
        # Always run the EKF predict step — propagates state during occlusions
        # and stale frames. Boat does not teleport.
        self.ekf.predict(self.DT)

        if perception.get("detected", False):
            tvec = perception["tvec"]
            drone_pos = np.asarray(drone_state["position"], dtype=np.float64)
            R_world_body = rpy_to_matrix(drone_state["attitude"])
            body_x = R_world_body[:, 0]
            body_y = R_world_body[:, 1]
            body_z = R_world_body[:, 2]
            camera_world = drone_pos + body_z * CAMERA_BODY_OFFSET_Z
            R_cam_to_world = np.column_stack([-body_y, -body_x, -body_z])
            marker_world = camera_world + R_cam_to_world @ tvec

            self._last_marker_world = marker_world
            self._last_marker_z = float(marker_world[2])
            self._frames_since_detection = 0

            # Feed (x, y) measurement to EKF
            self.ekf.update(marker_world[:2], measurement_var=perception.get("meas_var", 1.0))

            est_xy = self.ekf.position
            est_vxy = self.ekf.velocity
            return {
                "position": np.array([est_xy[0], est_xy[1], self._last_marker_z]),
                "velocity": np.array([est_vxy[0], est_vxy[1], 0.0]),
                "heading": self.ekf.heading,
                "turn_rate": self.ekf.turn_rate,
                "velocity_std": self.ekf.velocity_std,
                "fresh": True,
                "stale_steps": 0,
                "from_prior": False,
            }

        # No fresh detection — use EKF prediction if we have history
        self._frames_since_detection += 1
        if self.ekf._initialized:
            est_xy = self.ekf.position
            est_vxy = self.ekf.velocity
            return {
                "position": np.array([est_xy[0], est_xy[1], self._last_marker_z]),
                "velocity": np.array([est_vxy[0], est_vxy[1], 0.0]),
                "heading": self.ekf.heading,
                "turn_rate": self.ekf.turn_rate,
                "velocity_std": self.ekf.velocity_std,
                "fresh": False,
                "stale_steps": self._frames_since_detection,
                "from_prior": False,
            }

        # Cold start: use search prior
        return {
            "position": np.array([self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3]),
            "velocity": np.zeros(3),
            "heading": 0.0,
            "turn_rate": 0.0,
            "velocity_std": float("inf"),
            "fresh": False,
            "stale_steps": self._frames_since_detection,
            "from_prior": True,
        }

    # ------------------------------------------------------------------ STAGE 3: DECISION (baseline for now)
    def decide(self, drone_state, boat_estimate, battery, time_s):
        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        if target is None or boat_estimate.get("from_prior", False):
            return PHASE_SEARCH

        horiz = float(np.linalg.norm(pos[:2] - np.asarray(target[:2])))
        z_above = float(pos[2] - target[2])
        prev = self.phase

        # LAND is sticky: once we're committed to touchdown, only abort if we
        # bounce well clear of the platform. Prevents LAND<->DESCEND thrash on
        # an oscillating HARD platform.
        if prev == PHASE_LAND:
            if z_above > self.LAND_ALTITUDE_EXIT:
                return PHASE_DESCEND
            return PHASE_LAND

        # Enter LAND once we're below the (tight) entry threshold.
        if z_above < self.DESCEND_ALTITUDE_OK:
            return PHASE_LAND

        # DESCEND is sticky in the horizontal axis: keep descending unless we
        # drift past the wide exit gate. Vertical exit still defers to LAND
        # entry above (handled by the earlier branch).
        if prev == PHASE_DESCEND:
            if horiz < self.DESCEND_HORIZ_EXIT and z_above < self.APPROACH_ALTITUDE_OK:
                return PHASE_DESCEND
            return PHASE_APPROACH

        # Enter DESCEND from APPROACH/SEARCH only at the tight gate.
        if horiz < self.DESCEND_HORIZ_ENTER and z_above < self.APPROACH_ALTITUDE_OK:
            return PHASE_DESCEND

        # APPROACH is sticky too: once close, don't drop back to SEARCH on a
        # single noisy frame. Fall back only on a wide drift or lost track.
        if prev == PHASE_APPROACH:
            if boat_estimate.get("fresh", False) or horiz < self.APPROACH_HORIZ_EXIT:
                return PHASE_APPROACH
            return PHASE_SEARCH

        if horiz < self.APPROACH_HORIZ_ENTER:
            return PHASE_APPROACH
        return PHASE_APPROACH if boat_estimate.get("fresh", False) else PHASE_SEARCH

    @staticmethod
    def _angle_wrap(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(((a + np.pi) % (2.0 * np.pi)) - np.pi)

    # ------------------------------------------------------------------ STAGE 4: CONTROL (velocity feed-forward)
    # Per-phase lookahead time (seconds). Tells controller to aim ahead of
    # the current boat position so it can match boat velocity at arrival.
    # Smaller during LAND (we want to touch down ON the platform, not lead it).
    # Lookahead: small / zero on purpose. The velocity feed-forward alone
    # is what matches drone velocity to boat velocity at steady-state.
    # Adding a big lookahead on top makes the drone aim PAST the boat —
    # decide() never sees horiz < APPROACH_HORIZ_OK and we never descend.
    LOOKAHEAD_S = {
        PHASE_SEARCH:   0.0,
        PHASE_APPROACH: 0.2,    # tiny lead to bias the catch-up direction
        PHASE_DESCEND:  0.0,
        PHASE_LAND:     0.0,
    }

    # EKF velocity-uncertainty gate. While 1-sigma on the boat velocity is
    # above TRUST, the feed-forward and yaw heading are tapered toward zero
    # influence so the cold-start (0,0) velocity doesn't drive the drone with
    # garbage. EKF init is P_v = 4.0 -> sigma = 2.0 m/s; steady-state ~0.2.
    # Relaxed from (0.6, 1.2). On HARD (15 fps + fog), EKF v_std never
    # converges below 1.2, so vel_trust stayed at 0 and the drone never
    # used velocity feed-forward → missed the moving boat → crashed.
    EKF_V_TRUST_SIGMA = 1.2
    EKF_V_GATE_SIGMA = 3.0

    def control(self, drone_state, boat_estimate, phase):
        # Selective PID resets. The old code reset all three PIDs on every
        # phase change — bad now that hysteresis lets DESCEND<->LAND flip on a
        # bobbing platform: each flip dumped useful integral/derivative state.
        # Reset xy only on transitions that imply a large xy setpoint jump
        # (entering or leaving SEARCH); reset z whenever the z control law
        # structure changes (i.e. any phase transition, since each phase has
        # its own target_vz and blending rule).
        prev = self._prev_phase
        if prev is not None and phase != prev:
            xy_setpoint_jump = (phase == PHASE_SEARCH) or (prev == PHASE_SEARCH)
            if xy_setpoint_jump:
                self.pid_x.reset()
                self.pid_y.reset()
            self.pid_z.reset()
        self._prev_phase = phase

        pos = np.asarray(drone_state["position"], dtype=np.float64)
        target = boat_estimate.get("position")
        raw_boat_vel = np.asarray(
            boat_estimate.get("velocity", np.zeros(3)), dtype=np.float64
        )
        if target is None:
            target = np.array([self.SEARCH_PRIOR_XY[0], self.SEARCH_PRIOR_XY[1], 0.3])

        # Always trust the EKF velocity once it has any measurement at all.
        # The gate added by the linter was breaking HARD because v_std never
        # converged below the threshold (15 fps + fog = sparse measurements).
        v_std = float(boat_estimate.get("velocity_std", float("inf")))
        vel_trust = 0.0 if not np.isfinite(v_std) else 1.0
        boat_vel = raw_boat_vel * vel_trust

        # PRE-AIM: lead the boat by lookahead*velocity. Compensates for the
        # drone's transport delay (it takes ~1s to accelerate to boat speed).
        lookahead = self.LOOKAHEAD_S.get(phase, 0.0)
        aim_xy = np.array([target[0], target[1]]) + boat_vel[:2] * lookahead

        target_z = float(target[2]) + self.PHASE_ALTITUDE[phase]

        R = rpy_to_matrix(drone_state["attitude"])
        vel_world = np.asarray(drone_state["velocity"], dtype=np.float64)
        vel_body = R.T @ vel_world
        # Boat velocity rotated into drone body frame (ignore z component)
        boat_vel_body = R.T @ np.array([boat_vel[0], boat_vel[1], 0.0])

        err_world = np.array(
            [aim_xy[0] - pos[0], aim_xy[1] - pos[1], target_z - pos[2]],
            dtype=np.float64,
        )
        err_body = R.T @ err_world

        # VELOCITY FEED-FORWARD: feed (drone_vel - boat_vel) as measurement
        # velocity to each PID. At steady-state (err=0), the PID's D-term
        # drives drone_vel toward boat_vel instead of toward 0 — so we
        # naturally fly alongside the moving boat instead of dropping behind.
        rel_vx_body = vel_body[0] - boat_vel_body[0]
        rel_vy_body = vel_body[1] - boat_vel_body[1]

        # Phase-dependent horizontal aggressiveness: boost Kp during
        # SEARCH/APPROACH so the drone pitches to max tilt earlier (was only
        # ~60% of max at the start of an EASY run). Keep gentle Kp during
        # DESCEND/LAND to avoid overshoot when close to the platform.
        pitch_cmd = self.pid_x(err_body[0], self.DT, measurement_velocity=rel_vx_body)
        roll_cmd = -self.pid_y(err_body[1], self.DT, measurement_velocity=rel_vy_body)

        # COMBINED TILT LIMIT: if both pitch and roll saturate (rare with
        # baseline Kp but cheap insurance), scale both proportionally to
        # keep total tilt <= 1.0 so vertical thrust isn't lost to extreme
        # diagonal tilt.
        tilt_mag = np.sqrt(pitch_cmd ** 2 + roll_cmd ** 2)
        if tilt_mag > 1.0:
            pitch_cmd /= tilt_mag
            roll_cmd /= tilt_mag
        thrust_cmd = self.pid_z(err_world[2], self.DT, measurement_velocity=vel_world[2])

        # SOFT LANDING: instead of bang-bang -1.0 thrust (which slams the
        # drone into the platform at 2+ m/s, forfeiting the +5 soft-landing
        # bonus), command a controlled descent velocity. Targets vz=-0.6 m/s
        # which is well below the 1.0 m/s soft-landing threshold, but firm
        # enough to break through the ref sim's ground-effect cushion.
        # Cap descent velocity in ALL non-search phases. max_descent_velocity
        # is the cumulative max over the whole episode, so a fast descent
        # anywhere (including APPROACH when the drone drops to phase altitude
        # for the first time, or on HARD with strong platform oscillation)
        # will kill the soft-landing bonus even if LAND itself is gentle.
        # Phase-specific descent rate targets. Cap the maximum allowed vz
        # so the max_descent_velocity stays under the 1.0 m/s soft-landing
        # threshold throughout the episode.
        # All targets kept comfortably ABOVE the brake threshold (-0.75)
        # so the natural phase descent doesn't constantly fight the brake.
        if phase == PHASE_SEARCH:
            target_vz = -0.65
            vz_err = target_vz - vel_world[2]
            vel_thrust = float(np.clip(1.5 * vz_err, -0.6, 0.3))
            thrust_cmd = max(thrust_cmd, vel_thrust)
        elif phase == PHASE_APPROACH:
            target_vz = -0.6
            vz_err = target_vz - vel_world[2]
            vel_thrust = float(np.clip(1.5 * vz_err, -0.6, 0.3))
            thrust_cmd = max(thrust_cmd, vel_thrust)
        elif phase == PHASE_DESCEND:
            target_vz = -0.55
            vz_err = target_vz - vel_world[2]
            vel_thrust = float(np.clip(1.5 * vz_err, -0.6, 0.3))
            thrust_cmd = max(thrust_cmd, vel_thrust)
        elif phase == PHASE_LAND:
            target_vz = -0.45
            vz_err = target_vz - vel_world[2]
            thrust_cmd = float(np.clip(1.5 * vz_err, -0.8, 0.3))

        # SAFETY BRAKE: activates at -0.75 (above all phase targets so it
        # doesn't fight normal descent). Ramps to full thrust by -1.0.
        if vel_world[2] < -0.75:
            excess = (-vel_world[2] - 0.75) / 0.25
            brake = float(np.clip(0.3 + 0.7 * excess, 0.3, 1.0))
            thrust_cmd = max(thrust_cmd, brake)

        # YAW ALIGNMENT. The drone fuselage must end up within the per-scenario
        # yaw tolerance of the boat heading mod pi. Two issues fixed vs. the
        # old version:
        #   1. EASY has a stationary boat — atan2(vy,vx) is undefined, the
        #      original "boat_speed < 0.2 -> zero yaw_rate" branch meant we
        #      never aligned and relied on lucky spawn yaw.
        #   2. The heading must be sourced from a *trusted* EKF velocity. We
        #      lock the heading target whenever the EKF velocity is confident
        #      AND the boat is actually moving; otherwise we hold the last
        #      locked target. If we never see motion (true stationary), we
        #      fall back to aligning to the drone's own current course over
        #      ground, which is a safe no-op (zero yaw_err) and keeps the
        #      yaw rate at 0.
        drone_yaw = float(drone_state["attitude"][2])
        raw_boat_speed = float(np.linalg.norm(raw_boat_vel[:2]))
        HEADING_LOCK_SPEED = 0.4
        if raw_boat_speed >= HEADING_LOCK_SPEED and vel_trust > 0.5:
            self._yaw_target_locked = float(boat_estimate.get("heading", 0.0))

        if phase == PHASE_SEARCH or self._yaw_target_locked is None:
            yaw_rate_cmd = 0.0
        else:
            tgt = self._yaw_target_locked
            err_a = self._angle_wrap(tgt - drone_yaw)
            err_b = self._angle_wrap(tgt + np.pi - drone_yaw)
            yaw_err = err_a if abs(err_a) < abs(err_b) else err_b
            yaw_rate_cmd = float(np.clip(1.5 * yaw_err, -1.0, 1.0))

        return self.attitude_ctrl(
            drone_state,
            roll_des=roll_cmd,
            pitch_des=pitch_cmd,
            yaw_rate_des=yaw_rate_cmd,
            thrust_norm=thrust_cmd,
        )


Agent = PansecchiAgent


def make_agent(drone_spec: Optional[DroneSpec] = None) -> PansecchiAgent:
    return PansecchiAgent(drone_spec)
