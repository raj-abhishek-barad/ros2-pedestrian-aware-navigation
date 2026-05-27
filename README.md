# ROS2 Pedestrian-Aware Navigation Simulator

A pure ROS2 + RViz2 pedestrian-aware navigation simulator implementing:

- Pedestrian tracking
- IMM-based motion estimation
- Intent prediction
- Risk-aware collision avoidance
- Velocity attenuation and hard-stop logic
- Real-time RViz2 visualization

This project is fully software-based and does not require TurtleBot, Gazebo, or physical hardware.

---

# Features

## Pedestrian Simulation
A synthetic pedestrian scenario generator publishes:
- crossing motion
- stopping behavior
- approaching motion
- receding motion

using ROS2 topics.

## IMM Tracking
The system uses an Interacting Multiple Model (IMM) tracker with:
- Constant Velocity (CV)
- Coordinated Turn (CTRV)
- Stopping models

to estimate pedestrian motion uncertainty and behavior.

## Intent Estimation
The pipeline estimates:
- crossing
- approaching
- stopping
- receding

using motion history and probabilistic inference.

## Risk Assessment
A risk filter computes:
- clearance
- time-to-collision (TTC)
- closing speed
- risk score

and attenuates robot velocity accordingly.

## RViz2 Visualization
The simulator visualizes:
- robot motion
- pedestrian trajectories
- velocity arrows
- intent labels
- IMM weights
- risk states

in RViz2.

---

# Architecture

```text
fake_scenario
      ↓
pedestrian_pipeline
      ↓
risk_filter
      ↓
robot_simulator
      ↓
RViz2
