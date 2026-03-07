#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
singularity_monitor.py  (ROS1 + MoveIt)

Real-time kinematic singularity monitor based on Jacobian SVD.

Overview
--------
This node evaluates the current manipulator configuration and measures
how close the robot is to a kinematic singularity. The computation follows
the same core idea used in the NBV selector module:

    Jacobian -> SVD -> sigma_min / condition number -> m_sing score

Inputs
------
/joint_states (sensor_msgs/JointState)

Method
------
1. Obtain the Jacobian matrix from MoveIt:

        J = get_jacobian_matrix(q)      (6×N twist Jacobian)

2. Perform singular value decomposition:

        J = U * diag(S) * V^T

   where S contains the singular values.

3. Extract singularity indicators:

        sigma_min = min(S)
        sigma_max = max(S)
        cond = sigma_max / sigma_min

4. Convert sigma_min into a normalized singularity score:

        sigma_min >= sigma_safe  -> m_sing = 1
        sigma_min <= sigma_hard  -> m_sing = 0
        otherwise linear interpolation

5. The weakest controllable task-space direction can be approximated
   using the left singular vector corresponding to sigma_min.

Run
---
rosrun <your_pkg> singularity_monitor.py _group:=fr3_arm _rate:=10

Parameters
----------
~group (str)
    MoveIt planning group name (default: "fr3_arm")

~rate (float)
    Monitor loop frequency in Hz (default: 10)

~ee_link (str)
    End-effector link (if empty, the group's default EE is used)

~sigma_safe (float)
    Threshold for safe manipulability (default: 0.05)

~sigma_hard (float)
    Hard singularity threshold (default: 0.005)

~print_every (int)
    Logging frequency in loop iterations (default: 1)

~use_current_state (bool)
    True  -> use MoveIt current state
    False -> map from /joint_states

Outputs (topics)
----------------
~sigma_min (std_msgs/Float32)
    Minimum singular value

~cond (std_msgs/Float32)
    Jacobian condition number

~m_sing (std_msgs/Float32)
    Normalized singularity score

Typical interpretation
----------------------
State            sigma_min       cond
---------------------------------------
Non-singular     0.05 ~ 0.3      5 ~ 40
Near singular    < 0.02          > 100
Extreme singular → 0             → ∞
"""

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import moveit_commander


class SingularityMonitor:
    """
    Real-time Jacobian-based singularity monitor.
    """

    def __init__(self):

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.group_name = rospy.get_param("~group", "fr3_arm")
        self.rate_hz = float(rospy.get_param("~rate", 10.0))

        self.sigma_safe = float(rospy.get_param("~sigma_safe", 0.05))
        self.sigma_hard = float(rospy.get_param("~sigma_hard", 0.005))

        self.print_every = int(rospy.get_param("~print_every", 1))

        if self.sigma_hard >= self.sigma_safe:
            self.sigma_hard = self.sigma_safe * 0.1

        # --------------------------------------------------
        # Joint state subscriber
        # --------------------------------------------------
        self.last_js = None
        rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)

        # --------------------------------------------------
        # MoveIt initialization
        # --------------------------------------------------
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        self.joint_names = self.group.get_active_joints()
        self.ee_link = self.group.get_end_effector_link()

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------
        self.pub_sigma_min = rospy.Publisher("~sigma_min", Float32, queue_size=10)
        self.pub_cond = rospy.Publisher("~cond", Float32, queue_size=10)
        self.pub_msing = rospy.Publisher("~m_sing", Float32, queue_size=10)

        rospy.loginfo("SingularityMonitor started")
        rospy.loginfo("Group: %s", self.group_name)
        rospy.loginfo("EE link: %s", self.ee_link)

        self.loop_count = 0

    # --------------------------------------------------
    # Joint state callback
    # --------------------------------------------------
    def _js_cb(self, msg):
        """
        Cache the latest JointState message.
        """
        self.last_js = msg

    # --------------------------------------------------
    # Retrieve joint vector
    # --------------------------------------------------
    def _get_q(self):
        """
        Extract the active joint configuration vector from /joint_states.
        """
        if self.last_js is None:
            return None

        name_to_pos = dict(zip(self.last_js.name, self.last_js.position))

        q = []
        for j in self.joint_names:
            if j not in name_to_pos:
                return None
            q.append(name_to_pos[j])

        return q

    # --------------------------------------------------
    # Compute normalized singularity score
    # --------------------------------------------------
    def _compute_msing(self, sigma_min):
        """
        Map sigma_min to a normalized singularity score in [0,1].
        """

        if sigma_min >= self.sigma_safe:
            return 1.0

        if sigma_min <= self.sigma_hard:
            return 0.0

        return (sigma_min - self.sigma_hard) / (self.sigma_safe - self.sigma_hard)

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------
    def spin(self):

        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():

            q = self._get_q()

            if q is None:
                rate.sleep()
                continue

            # --------------------------------------------------
            # Compute Jacobian
            # --------------------------------------------------
            try:
                J = np.array(self.group.get_jacobian_matrix(q))
            except Exception as e:
                rospy.logwarn_throttle(2.0, "Jacobian failed: %s", str(e))
                rate.sleep()
                continue

            if J.size == 0:
                rate.sleep()
                continue

            # --------------------------------------------------
            # SVD decomposition
            # --------------------------------------------------
            try:
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
            except np.linalg.LinAlgError:
                rate.sleep()
                continue

            sigma_min = float(np.min(S))
            sigma_max = float(np.max(S))

            # --------------------------------------------------
            # Condition number
            # --------------------------------------------------
            if sigma_min < 1e-12:
                cond = float("inf")
            else:
                cond = sigma_max / sigma_min

            # --------------------------------------------------
            # Singularity score
            # --------------------------------------------------
            m_sing = self._compute_msing(sigma_min)

            # --------------------------------------------------
            # Publish results
            # --------------------------------------------------
            self.pub_sigma_min.publish(Float32(sigma_min))
            self.pub_cond.publish(Float32(cond))
            self.pub_msing.publish(Float32(m_sing))

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------
            self.loop_count += 1

            if self.loop_count % self.print_every == 0:
                rospy.loginfo(
                    "sigma_min=%.6f  cond=%.2f  m_sing=%.3f",
                    sigma_min,
                    cond,
                    m_sing
                )

            rate.sleep()


if __name__ == "__main__":

    rospy.init_node("singularity_monitor")

    monitor = SingularityMonitor()
    monitor.spin()