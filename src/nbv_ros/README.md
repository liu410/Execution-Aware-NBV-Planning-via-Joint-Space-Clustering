# nbv_ros

A ROS package for active vision guided apple observation and next-best-view (NBV) planning with a 6-DoF manipulator.

This package contains the perception, reconstruction, scene representation, candidate-view generation, and execution modules used in our apple observation experiments. It supports both simulation and hardware-based execution.

## Main Functions

This package includes:

- in-hand RealSense camera launch
- apple instance segmentation and mask extraction
- mask quality analysis
- apple point cloud extraction and geometric reconstruction
- scene point filtering
- OctoMap-based scene representation
- silhouette-based candidate view generation
- NBV analysis in task space
- singularity monitoring
- two comparative NBV execution pipelines:
  - Layer-1 + IK-only
  - Layer-1 + Layer-2 Action-Mode

## Dependencies

This package depends on the following external ROS packages:

- `frcobot_ros-master`
- `realsense-ros`
- `ultralytics_ros`

Please make sure these packages are correctly installed in the same catkin workspace and can be built successfully.

## Workspace

Example workspace:

```bash
~/catkin_ws/src/
├── frcobot_ros-master
├── realsense-ros
├── ultralytics_ros
└── nbv_ros
