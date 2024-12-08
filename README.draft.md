# Multi-modal UAV for Natural Disaster Rescue

A hybrid locomotion UAV system capable of both flying and driving for natural disaster rescue scenarios, based on [RL-pybullets-cf](https://github.com/danielbinschmid/RL-pybullets-cf).

## Overview

This project implements a system that automatically selects between driving and flying modes to maximize drone energy efficiency in complex environments like natural disaster rescue scenes. The drone is equipped with both propellers and wheels, enabling efficient hybrid locomotion.

## Key Features

- **Hybrid Locomotion**: 
  - Wheeled driving for energy-efficient ground movement
  - Quadcopter flight for obstacle avoidance and vertical movement
  - Automatic mode selection based on environment

- **Intelligent Control**:
  - Regression-based mode selection
  - Trajectory following with RL
  - Energy optimization

- **Environment Analysis**:
  - U-shaped map structure analysis
  - Energy consumption prediction (94.4% accuracy)

## System Architecture

1. **Hardware Design**
   - 4 propellers for flight control
   - 4 wheels for ground locomotion
   - Integrated control system for mode switching

2. **Control System**
   - Wheel-Propeller coordination
   - Trajectory following
   - Mode transition management
   - Energy optimization

3. **Decision Making**
   - Environment analysis
   - Mode selection

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
- Flight mode: Curved path over obstacles
- Ground mode: Direct path with height consideration
- Dynamic waypoint generation based on terrain

#### Mode Selection
- Energy consumption prediction for each mode
- Automatic mode switching based on:
  - Distance to target
  - Obstacle height
  - Energy efficiency

### 3. Environment Generation

#### Map Generation
- Parametric U-shaped terrain generation
- Configurable obstacle heights (h1, h2)
- Adjustable obstacle distance (l)
- Dynamic URDF model generation

#### Energy Model
- Mode-specific energy consumption calculation
- Ground mode: Linear with distance
- Flight mode: Considers vertical movement cost
- Real-time energy monitoring and logging

## Results

1. **Mode Selection Efficiency**
   - Drive mode optimal for longer distances
   - Flight mode preferred for short distances
   - 94.4% accuracy in energy prediction

2. **Energy Optimization**
   - Significant energy savings through mode switching
   - Optimal path planning for minimal energy consumption

## Team
- Doheun Kim
- Minsol Park
- Doun Lee
- Seongjun Lee

KAIST
