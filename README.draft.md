# Multi-modal UAV for Natural Disaster Rescue

A hybrid locomotion UAV system capable of both flying and driving for natural disaster rescue scenarios, based on [RL-pybullets-cf](https://github.com/danielbinschmid/RL-pybullets-cf). (Flight functionality is implemented, and we additionally implemented driving functionality on top of it.)

## Overview

This project implements a system that automatically selects between driving and flying modes to maximize drone energy efficiency in u-shaped environment (A terrain with flat ground between two cliffs). The drone is equipped with both propellers and wheels, enabling efficient hybrid locomotion.

## Key Features

- **Hybrid Locomotion**: 
  - Wheeled driving for energy-efficient ground movement
  - Quadcopter flight
  - Automatic mode selection based on environment

- **Intelligent Control**:
  - Regression-based mode selection (Flight or drive). It selects a way to reach the target based on cliffs's height and distance between them.
  - Trajectory following with RL

- **Environment Analysis**:
  - U-shaped map structure analysis
  - Energy-efficient mode prediction (94.4% accuracy)

## System Architecture

1. **Hardware Design**
   - 4 propellers for flight control
   - 4 wheels for ground locomotion
   - Integrated control system for mode switching

2. **Control System**
   - Wheel-Propeller coordination
   - Trajectory following
   - Mode transition management
   - Energy optimization selection between flight and drive

3. **Decision Making**
   - Environment analysis
   - Mode selection by regression model

## Setup

Tested on ArchLinux, Ubuntu and macOS. Note that Eigen must be installed on the system.

On Ubuntu:

```bash
sudo apt-get install libeigen3-dev
```

On macOS:

```bash
brew install eigen
```

It is strongly recommended to use a python virtual environment such as conda or pyvenv.

### Installation Steps

1. Clone repository (recursively)

```bash
git clone https://github.com/rolsj/CS470_trajectory_following.git
git submodule update --init --recursive
```

2. Setup Python environment (tested with Python 3.10.13)

```bash
pyenv install 3.10.13
pyenv local 3.10.13
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
```

3. Install dependencies and build

```bash
pip3 install -e .  # Ubuntu users might need: sudo apt install build-essential
```

## Usage

Scripts for training, testing and visualization are provided.

### Training

```bash
cd runnables
./train_rl.sh
```

### Testing

```bash
cd runnables
./test_rl.sh
./test_pid.sh
```

### Visualization

```bash
cd runnables
./vis_rl.sh
```

### Run experiments
```bash
./runnables/exper_rl.sh [mode] [h1] [h2] [l]

mode = drive or flight // h1, h2, l are float.
Example 1: ./runnables/exper_rl.sh drive 1.5 1.5 8
A terrain is created with two cliffs, each 1.5 meters high, and a 7-meter gap between them.
Currently, h1 and h2 must be the same.
```

## Implementation Details

The system is implemented using PyBullet physics engine with custom URDF models for the hybrid drone:

### 1. Hybrid Locomotion Control

#### Ground Movement
- Velocity-based differential wheel control
- Dynamic speed adjustment based on future waypoints
- Automatic direction control using cross-product for optimal rotation
- Logarithmic speed scaling for smooth acceleration/deceleration
- Integrated ground contact detection for mode switching

#### Flight Control
- 4-motor RPM control system
- Attitude-based control
- Smooth transition between ground and flight modes

### 2. Path Planning & Navigation

#### Trajectory Generation
- Parabolic trajectory generation for both modes
- Flight mode: Curved path
- Ground mode: Curved paths from cliff to ground. (Different parabolic path is generated based on the cliff.) A direct path connecting takeoff and landing.
- Dynamic waypoint generation based on terrain

#### Mode Selection
- Energy consumption prediction for each mode
- Automatic mode switching based on:
  - Distance to target
  - cliff's heights

### 3. Environment Generation

#### Map Generation
- Parametric U-shaped terrain generation
- Configurable cliff heights (h1, h2)
- Adjustable cliff distance (l)
- Dynamic cliff URDF model generation

#### Energy Model
- Mode-specific energy consumption calculation
- Ground mode: Energy is calculated by multiplying the torque acting on the wheel by its angular velocity.
- Flight mode: It is calculated using the formula for energy used by the propeller.
- Real-time energy monitoring and logging

## Results

1. **Mode Selection Efficiency**
   - Drive mode optimal for longer distances
   - Flight mode preferred for short distances
   - 94.4% accuracy in mode prediction

2. **Energy Optimization**
   - Significant energy savings through mode switching
   - Optimal path planning for minimal energy consumption

## Team
- Doheun Kim
- Minsol Park
- Doun Lee
- Seongjun Lee

KAIST
