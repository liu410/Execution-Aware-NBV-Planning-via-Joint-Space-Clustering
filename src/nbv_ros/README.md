# Execution-Aware NBV Planning via Joint-Space Clustering

ROS implementation of the **execution-aware Next-Best-View (NBV) planning framework** for apple perception under occlusion.

This repository provides the implementation used in our research on **execution-aware active perception for eye-in-hand manipulators**.

## Overview

In orchard environments, apples are frequently occluded by leaves and branches, which makes reliable pose estimation challenging. Active perception using **Next-Best-View (NBV)** planning allows a robot to adaptively select observation viewpoints that improve perception quality.

However, task-space optimal viewpoints are not always executable due to:

- joint limits
- kinematic singularities
- inverse kinematics multiplicity
- motion planning feasibility

To address this issue, we propose a **dual-layer NBV framework**:

**Layer-1:**  
Task-space candidate viewpoint generation and perceptual evaluation.

**Layer-2:**  
Joint-space Action-Mode clustering and execution-aware selection.

The framework improves planning success rate and execution robustness by incorporating joint-space structure into NBV decision making.



## System Architecture

Example system setup:

- **Robot:** 6-DoF collaborative manipulator  
- **Camera:** Intel RealSense D405 (eye-in-hand)  
- **Perception:** YOLOv8 instance segmentation  
- **Planning:** MoveIt motion planning  

### Example pipeline
```text
RGB-D perception
  → Apple segmentation
  → 3D localization
  → Layer-1 NBV candidate generation
  → IK mapping to joint space
  → Layer-2 Action-Mode clustering
  → Execution-aware viewpoint selection
  → Robot motion execution
```

## System Requirements

Recommended environment:

- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- MoveIt
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## Dependencies
This repository depends on the following ROS packages.

#### RealSense ROS Driver
https://github.com/IntelRealSense/realsense-ros

Please install Intel RealSense SDK version **2.53.1**.

#### Ultralytics ROS
https://github.com/Alpaca-zip/ultralytics_ros


Clone them into your workspace:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# clone realsense-ros:
git clone https://github.com/realsenseai/realsense-ros.git

#clone yolov8:
git clone -b noetic-devel https://github.com/Alpaca-zip/ultralytics_ros.git
python3 -m pip install -r ultralytics_ros/requirements.txt
```

**Note:** After cloning the repositories, please refer to the official documentation of each package to install the required dependencies.

## Installation
Clone the repository:
```bash
cd ~/catkin_ws/src
git clone https://github.com/liu410/Execution-Aware-NBV-Planning-via-Joint-Space-Clustering.git
```

Make sure your ROS workspace has the following structure:

```text
catkin_ws
├── src
│   ├── nbv_ros
│   ├── realsense-ros
│   ├── frcobot_ros-master
│   └── ultralytics_ros
```

Before building the workspace, install the required dependencies:
```bash
sudo apt install -y \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  build-essential \
  ros-noetic-ros-control \
  ros-noetic-ros-controllers \
  ros-noetic-joint-state-publisher \
  ros-noetic-joint-state-publisher-gui \
  ros-noetic-robot-state-publisher \
  ros-noetic-xacro \
  ros-noetic-rviz \
  ros-noetic-rviz-visual-tools \
  ros-noetic-moveit \
  ros-noetic-moveit-ros-planning \
  ros-noetic-moveit-ros-planning-interface \
  ros-noetic-moveit-ros-move-group \
  ros-noetic-moveit-simple-controller-manager \
  ros-noetic-ompl \
  ros-noetic-gazebo-ros-pkgs \
  ros-noetic-gazebo-ros-control \
  python3-catkin-tools \
  python3-pip \
  python3-pykdl \
  ros-noetic-kdl-parser-py \
  ros-noetic-octomap-server
```
**Note:** Additional dependencies may be required depending on your system configuration.
If missing packages are reported during compilation, please install them according to the terminal output.
Build & Source:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Running the System
Go to the workspace directory:
```bash
cd ~/catkin_ws
```
Start ROS master:
```bash
roscore
```
### 1. Robot simulation
```bash
# launch robot simulation:
roslaunch fr3_moveit_config demo_simulation.launch

# or launch real robot:
roslaunch fr3_moveit_config demo_hardware.launch
```

### 2. Launch eye-in-hand camera
```bash
roslaunch nbv_ros realsense_in_hand.launch
```

### 3. Apple detection and mask segmentation
```bash
rosrun nbv_ros apple_segmentation_detector.py
```

### 4. Apple reconstruction
```bash
rosrun nbv_ros apple_reconstruction.py
```

### 5. Run point cloud filtering
```bash
rosrun nbv_ros point_filter.py
```

### 6. Run OctoMap
This OctoMap is used for NBV raycasting evaluation.
```bash
roslaunch nbv_ros octomap_server.launch
```
**Note:** Steps 5 and 6 set up two independent OctoMaps. After both OctoMaps appear in RViz, you may stop `point_filter.py` with `Ctrl+C`. The OctoMaps will remain active.

### 7. Silhouette sampling
```bash
rosrun nbv_ros silhouette_detector.py
```
### 8. NBV generation
```bash
rosrun nbv_ros silhouette_nbv_analyzer.py
```
### 9. Try our proposed method
```bash
# Select NBV using joint-space Action-Mode clustering:
rosrun nbv_ros nbv_selector_action_mode.py

# Execute the NBV configuration:
rosrun nbv_ros nbv_executor_action_mode.py
```
### Additional Metrics
The following scripts can be used to compute additional metrics for evaluating the current observation state and kinematic safety.

#### Apple Mask Quantifier
Evaluates the quality of the apple segmentation mask under occlusion.

```bash
rosrun nbv_ros mask_quality_quantifier.py
```
#### Singularity Monitoring
Monitors the manipulator’s proximity to kinematic singularities.

```bash
rosrun nbv_ros singularity_monitor.py
```

## Experiments

Experimental result data are stored in the `experiment` folder.  
You can generate the summarized results by running:
```bash
cd ~/catkin_ws/src/nbv_ros/experiment

python3 merge_experiments.py
python3 make_three_line_table.py
python3 paired_trial_analysis.py
```

#### Running New Experiments
If you want to run new experiments, create numbered folders in the corresponding directory and move the generated log files into those folders for statistical analysis.

Each experiment run generates .CSV log files used for the statistical analysis reported in the paper.
#### Baseline Method (Layer-1 + IK)
Select NBV using IK feasibility and PSC scoring:
```bash
cd ~/catkin_ws/src/nbv_ros/experiment/L1+IK_only
python3 nbv_selector_ik_only.py
```
Execute the selected NBV configuration:
```bash
cd ~/catkin_ws/src/nbv_ros/experiment/L1+IK_only
python3 nbv_executor_IK_only.py
```
#### Proposed Method (Layer-1 + Layer-2)

Select NBV using joint-space Action-Mode clustering:
```bash
cd ~/catkin_ws/src/nbv_ros/experiment/L1+L2
python3 nbv_selector_action_mode.py
```
Execute the selected NBV configuration:
```bash
cd ~/catkin_ws/src/nbv_ros/experiment/L1+L2
python3 nbv_executor_action_mode.py
```
**Note:** After all experiments are completed, please check the generated .CSV files. Some files may contain multiple rows of data because MoveIt may attempt motion planning more than once (e.g., the first plan fails and a second attempt is generated). For statistical consistency, keep only the first row of each file and delete the remaining rows. Otherwise, `merge_experiments.py` may produce incorrect statistics.

---
