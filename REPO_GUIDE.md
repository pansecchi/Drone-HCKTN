# gym-pybullet-drones — How It Works

A reference guide for the drone landing hackathon.

---

## Big Picture

The library is a **physics simulator wrapped as a Gym environment**. You write code that runs in a loop: ask the environment for the drone's current state, compute what RPMs to send to the 4 motors, send them, repeat 48 times per second.

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR CODE (agent)                      │
│                                                             │
│   perception → estimation → decision → control             │
│                                   ↓                         │
│                          4 motor RPMs                       │
└───────────────────────────┬─────────────────────────────────┘
                            │ env.step(action)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   gym-pybullet-drones                       │
│                                                             │
│   PyBullet physics (240 Hz)  →  observation (20 values)    │
│   + aerodynamics models      →  back to your code          │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Layout

```
gym-pybullet-drones/
│
├── gym_pybullet_drones/
│   ├── envs/          ← simulation environments (choose one to use)
│   ├── control/       ← ready-made controllers (PID, MPC, etc.)
│   ├── utils/         ← Logger, enums, helpers
│   └── assets/        ← drone .urdf files (physical parameters)
│
└── gym_pybullet_drones/examples/
    ├── pid.py          ← classical control demo (START HERE)
    ├── learn.py        ← RL training with Stable-Baselines3
    └── play.py         ← replay a trained RL policy
```

---

## Environments (`envs/`)

Every environment inherits from `BaseAviary → gym.Env`.

| Class | Use it when... | Action | Observation |
|-------|---------------|--------|-------------|
| **`CtrlAviary`** | You write your own controller (our case) | RPMs directly | 20-value state vector |
| **`HoverAviary`** | Training an RL agent to hover | RPMs or PID targets | kinematic or RGB |
| **`VelocityAviary`** | High-level waypoint navigation | velocity setpoints | kinematic |
| **`MultiHoverAviary`** | Multi-drone RL (leader-follower) | same as Hover | same as Hover |

**For the hackathon: use `CtrlAviary`.** You control RPMs directly and use `DSLPIDControl` to convert position targets to RPMs.

### Creating an environment

```python
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

env = CtrlAviary(
    drone_model=DroneModel.CF2X,   # Crazyflie 2.0 X-config, 27g
    num_drones=1,
    initial_xyzs=np.array([[0, 0, 0.5]]),  # start 50cm above origin
    physics=Physics.PYB,           # base PyBullet (fast)
    pyb_freq=240,                  # physics steps per second
    ctrl_freq=48,                  # your control loop Hz
    gui=True,                      # set False for headless/batch runs
)
obs, info = env.reset()
```

### Physics modes

| Mode | What it adds | Use for |
|------|-------------|---------|
| `PYB` | Basic rigid-body | Fast iteration, most testing |
| `PYB_DRAG` | + aerodynamic drag | Wind disturbance simulation |
| `PYB_GND` | + ground effect | Near-surface landing |
| `PYB_GND_DRAG_DW` | All effects | HARD scenario, spec sheet runs |

---

## The Observation Vector (20 values)

Every call to `env.step()` returns `obs` of shape `(num_drones, 20)`.

```
obs[0, :]  →  state of drone 0

Index   Symbol   Meaning                   Unit
─────────────────────────────────────────────────
0       x        position X                meters
1       y        position Y                meters
2       z        position Z (altitude)     meters
3-6     q        quaternion (w,x,y,z)      —
7       φ (roll) rotation around X        radians
8       θ (pitch)rotation around Y        radians
9       ψ (yaw)  rotation around Z        radians
10      vx       linear velocity X         m/s
11      vy       linear velocity Y         m/s
12      vz       linear velocity Z         m/s
13      wx       angular velocity X        rad/s
14      wy       angular velocity Y        rad/s
15      wz       angular velocity Z        rad/s
16-19   rpm0-3   last motor commands       RPM
```

**Quick access helpers:**
```python
pos      = obs[0, 0:3]
quat     = obs[0, 3:7]
rpy      = obs[0, 7:10]
vel      = obs[0, 10:13]
ang_vel  = obs[0, 13:16]
```

---

## The Action

`env.step(action)` takes `action` of shape `(num_drones, 4)` — the RPM for each of the 4 motors.

```
action[0] = [rpm_motor0, rpm_motor1, rpm_motor2, rpm_motor3]
```

Useful constants from the env:
```python
env.HOVER_RPM   # RPM needed to hover in place (~14,600)
env.MAX_RPM     # maximum motor RPM (~21,700)
env.CTRL_FREQ   # your loop runs this many times per second (48)
env.CTRL_TIMESTEP  # seconds per step (1/48 ≈ 0.021s)
```

Motor layout (top view, CF2X):
```
  M1 (CCW) --- M2 (CW)
      \       /
       drone body
      /       \
  M4 (CW)  --- M3 (CCW)
```

---

## The Controller (`control/DSLPIDControl`)

You rarely command RPMs by hand. Use the provided cascaded PID controller to go from "I want to be at position XYZ" to RPMs.

```python
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

# Inside your loop:
rpms, pos_error, yaw_error = ctrl.computeControlFromState(
    control_timestep = env.CTRL_TIMESTEP,
    state            = obs[0],           # full 20-value state
    target_pos       = np.array([x, y, z]),   # where you want to go
    target_rpy       = np.array([0, 0, 0]),   # desired orientation
)
action[0] = rpms
```

### How the cascaded PID works internally

```
target_pos
    ↓
[Position PID]  →  target_thrust + target_roll/pitch
    ↓
[Attitude PID]  →  per-motor torques
    ↓
[Mixer matrix]  →  4 motor RPMs
```

PID gains (in `DSLPIDControl.__init__`):
```python
P_COEFF_FOR = [0.4,  0.4,  1.25]   # position P gains (x, y, z)
I_COEFF_FOR = [0.05, 0.05, 0.05]   # position I gains
D_COEFF_FOR = [0.2,  0.2,  0.5]    # position D gains (uses current velocity)
```

---

## The Simulation Loop

This is the pattern every script uses:

```python
env = CtrlAviary(gui=True, ctrl_freq=48)
ctrl = DSLPIDControl(DroneModel.CF2X)
obs, _ = env.reset()

action = np.zeros((1, 4))

for step in range(int(duration * env.CTRL_FREQ)):

    # 1. Sense
    pos = obs[0, 0:3]
    vel = obs[0, 10:13]

    # 2. Decide where to go
    target = np.array([1.0, 0.0, 1.5])

    # 3. Compute RPMs
    action[0], _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=target,
    )

    # 4. Step physics
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()  # prints position to console

env.close()
```

---

## Logging and Plotting

```python
from gym_pybullet_drones.utils.Logger import Logger

logger = Logger(logging_freq_hz=48, num_drones=1)

# Inside loop:
logger.log(
    drone=0,
    timestamp=step / env.CTRL_FREQ,
    state=obs[0],
    control=np.hstack([target_pos, np.zeros(9)])  # 12-value target
)

# After loop:
logger.save()            # saves .npy files to results/
logger.save_as_csv("run_name")
logger.plot()            # shows matplotlib time-series plots
```

Logger stores 16 state values per timestep: `[x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, rpm0, rpm1, rpm2, rpm3]`.

---

## Drone Physical Parameters (CF2X)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Mass | 27 g | Very light — sensitive to thrust noise |
| Arm length | 39.7 mm | Distance center → motor |
| kf | 3.16e-10 | Thrust coefficient: `F = kf * rpm²` |
| km | 7.94e-12 | Torque coefficient: `τ = km * rpm²` |
| Max speed | 30 km/h | Translational limit |
| Thrust-to-weight | 2.25 | Can accelerate at ~1.25g upward |
| Hover RPM | ~14,600 | RPM needed to hold altitude |
| Max RPM | ~21,700 | Hard cap per motor |

---

## Camera (for Perception module)

The environment supports a downward-facing camera on the drone:

```python
env = CtrlAviary(..., vision_attributes=True)
# After env.step():
rgb_image = env.rgb[0]   # shape (48, 64, 4) — RGBA at 24 fps
depth_img = env.dep[0]   # shape (48, 64)    — depth in meters
```

Resolution: 64×48 pixels, 24 fps (captured every 10 physics steps).

For ArUco detection:
```python
import cv2
from cv2 import aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
gray = cv2.cvtColor(rgb_image[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
```

---

## What the Hackathon Adds on Top

The hackathon repo wraps `CtrlAviary` with:
1. A **moving boat** (PyBullet body with sinusoidal roll/pitch + drift velocity)
2. An **ArUco marker** on the boat deck
3. A **battery model** (decreases with time, mission fails if it hits 0)
4. **Scenario configs** (EASY / MEDIUM / HARD) that set wave amplitude, wind, boat speed
5. An **evaluation script** that runs N episodes and scores: landing success, position error, vertical velocity at touchdown, time, battery remaining

Your 4-module agent skeleton slots into the same simulation loop shown above.

---

## Cheat Sheet

```python
# Minimal working agent
env = CtrlAviary(gui=False, ctrl_freq=48, physics=Physics.PYB_DRAG)
ctrl = DSLPIDControl(DroneModel.CF2X)
obs, _ = env.reset()
action = np.zeros((1,4))

for _ in range(48 * 30):  # 30 seconds
    state = obs[0]
    pos   = state[0:3]
    
    target_pos = np.array([0.0, 0.0, 1.0])   # hover 1m up
    
    action[0], _, _ = ctrl.computeControlFromState(
        env.CTRL_TIMESTEP, state, target_pos
    )
    obs, *_ = env.step(action)

env.close()
```

```python
# Run headless at max speed (no GUI, no render)
env = CtrlAviary(gui=False)
# → runs at ~200x real time on a laptop
```

```python
# Add physics effects for HARD scenario testing
env = CtrlAviary(physics=Physics.PYB_GND_DRAG_DW)
```
