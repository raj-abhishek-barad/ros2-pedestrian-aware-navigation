# ROS 2 Pedestrian-Aware Navigation Simulator

A pure ROS 2 + RViz2 pedestrian-aware navigation simulator — no TurtleBot, no Gazebo, no hardware required.

The system tracks a synthetic pedestrian through a scripted scenario, estimates their intent using a Hidden Markov Model, computes an intent-conditioned collision risk score, and gates a robot velocity command in real time. Everything is visualised live in RViz2.

```
fake_scenario  →  pedestrian_pipeline  →  risk_filter  →  robot_simulator  →  RViz2
```

---

## Features

| Capability | Implementation |
|---|---|
| Pedestrian motion estimation | Interacting Multiple Model (IMM) tracker: CV · CTRV-EKF · Stop |
| Intent prediction | HMM forward pass (crossing · approaching · stopping · receding) |
| Risk assessment | Intent-conditioned clearance + TTC scoring |
| Velocity gating | Linear attenuation onset → full-stop |
| Visualisation | Colour-coded cylinders · velocity arrows · floating intent labels · TF |

---

## Architecture

### Nodes

| Node | Role |
|---|---|
| `fake_scenario` | Publishes moving pedestrian detections + constant drive command |
| `pedestrian_pipeline` | IMM tracker + HMM intent estimator → JSON tracks + RViz2 markers |
| `risk_filter` | 20 Hz safety loop: attenuates `/cmd_vel` based on risk score |
| `robot_simulator` | Integrates `/cmd_vel` → moves robot marker + broadcasts TF |
| `rviz_overlay` | Floats intent/risk text labels above each pedestrian cylinder |

### Topic graph

```
fake_scenario
 ├─ /pedestrian_detections  →  pedestrian_pipeline
 │                               ├─ /tracked_pedestrians_json  →  risk_filter
 │                               │                                  └─ /cmd_vel  →  robot_simulator
 │                               └─ /pedestrian_markers        →  RViz2
 └─ /cmd_vel_input          →  risk_filter
                                 └─ /risk_state_json  →  rviz_overlay  →  /ped_labels  →  RViz2

robot_simulator
 ├─ /odom          →  pedestrian_pipeline, risk_filter
 ├─ /robot_marker  →  RViz2
 └─ /robot_heading →  RViz2
```

### Message types

| Topic | Type | Direction |
|---|---|---|
| `/pedestrian_detections` | `std_msgs/String` (JSON) | fake_scenario → pipeline |
| `/tracked_pedestrians_json` | `std_msgs/String` (JSON) | pipeline → risk_filter, overlay |
| `/pedestrian_markers` | `visualization_msgs/MarkerArray` | pipeline → RViz2 |
| `/ped_labels` | `visualization_msgs/MarkerArray` | overlay → RViz2 |
| `/risk_state_json` | `std_msgs/String` (JSON) | risk_filter → overlay |
| `/cmd_vel_input` | `geometry_msgs/Twist` | fake_scenario → risk_filter |
| `/cmd_vel` | `geometry_msgs/Twist` | risk_filter → robot_simulator |
| `/odom` | `nav_msgs/Odometry` | robot_simulator → pipeline, risk_filter |
| `/robot_marker` | `visualization_msgs/Marker` | robot_simulator → RViz2 |
| `/robot_heading` | `visualization_msgs/Marker` | robot_simulator → RViz2 |

---

## Installation

**Requirements:** ROS 2 Jazzy, Python 3.12, numpy

```bash
# 1. Clone into your workspace
cd ~/ros2_ws/src
git clone https://github.com/raj-abhishek-barad/ros2-pedestrian-aware-navigation.git pedestrian_aware_tb4

# 2. If you have a Python venv in the workspace, prevent colcon from scanning it
touch ~/ros2_ws/venv/COLCON_IGNORE   # skip if you don't have a venv

# 3. Build
cd ~/ros2_ws
colcon build --packages-select pedestrian_aware_tb4

# 4. Source
source install/setup.bash
```

---

## Running the demo

```bash
ros2 launch pedestrian_aware_tb4 demo.launch.py
```

RViz2 opens automatically with a top-down view, pre-configured displays, and a 5-second countdown before the scenario begins.

### What you will see

The pedestrian (tall cylinder) moves through four phases that loop every ~32 seconds:

| Phase | Duration | Pedestrian behaviour | What changes in RViz2 |
|---|---|---|---|
| 1 | 8 s | Crosses robot's path left → right | Cylinder **green** (CV model), label: `crossing` |
| 2 | 9 s | Stops 1.2 m ahead of robot | Cylinder turns **red** (Stop model), `risk ≥ 8 — STOP`, `/cmd_vel` → 0 |
| 3 | 7 s | Approaches robot head-on | Label turns red-orange: `approaching`, risk climbs |
| 4 | 8 s | Recedes diagonally away | Label turns green: `receding`, risk drops, speed recovers |

**Cylinder colour** = dominant IMM model (green = CV, yellow = CTRV, red = Stop).  
**Label colour** = dominant intent (cyan = crossing, red = approaching, yellow = stopping, green = receding).  
**Blue cylinder + white arrow** = robot body and heading direction.

### Inspecting topics at runtime

```bash
# Risk score and hard-stop flag
ros2 topic echo /risk_state_json

# Full track state including IMM weights and intent probabilities
ros2 topic echo /tracked_pedestrians_json

# Filtered vs requested velocity
ros2 topic echo /cmd_vel          # after safety filter
ros2 topic echo /cmd_vel_input    # before safety filter (always 0.4 m/s)
```

---

## Configuration

All tunable parameters are in `config/params.yaml` and can be overridden at launch time:

```bash
ros2 launch pedestrian_aware_tb4 demo.launch.py   # uses defaults

# Example: tighter sensor noise, slower attenuation
ros2 launch pedestrian_aware_tb4 pedestrian_safety.launch.py \
    imm_sigma_r:=0.05 risk_attenuation_onset:=2.0
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `imm_max_distance` | 1.5 m | Max association distance (greedy NN) |
| `imm_sigma_r` | 0.15 m | Position measurement noise std |
| `imm_sigma_a` | 0.5 m/s² | Acceleration noise std (CV / Stop) |
| `intent_history` | 15 frames | HMM sliding-window length |
| `risk_attenuation_onset` | 3.0 | Risk score where speed starts reducing |
| `risk_attenuation_full` | 8.0 | Risk score where speed → 0 |
| `min_clearance` | 0.5 m | Hard-stop clearance threshold |
| `max_ttc` | 2.0 s | Hard-stop TTC threshold |

---

## Tests

```bash
cd ~/ros2_ws/src/pedestrian_aware_tb4
python3 -m pytest test/ -v
```

12 unit tests covering the IMM tracker (weight convergence, state dimensionality, backward compatibility) and HMM intent estimator (probability normalisation, intent dominance, history management).

---

## Package structure

```
pedestrian_aware_tb4/
├── pedestrian_aware_tb4/
│   ├── nodes/
│   │   ├── fake_scenario.py          scenario generator
│   │   ├── pedestrian_pipeline.py    IMM tracker + intent estimator
│   │   ├── risk_filter.py            velocity safety gating
│   │   ├── robot_simulator.py        pose integrator + TF broadcaster
│   │   └── rviz_overlay.py           floating text labels
│   └── utils/
│       ├── imm_tracker.py            CV / CTRV-EKF / Stop IMM
│       ├── intent_estimator.py       HMM intent (4 states, 5 features)
│       └── risk_model.py             clearance + TTC risk scoring
├── launch/
│   ├── demo.launch.py                full demo + RViz2
│   └── pedestrian_safety.launch.py   pipeline + filter only
├── config/
│   ├── params.yaml                   all ROS 2 parameters
│   └── demo.rviz                     pre-configured RViz2 layout
└── test/
    ├── test_imm_tracker.py           6 IMM unit tests
    └── test_intent_estimator.py      6 intent unit tests
```
