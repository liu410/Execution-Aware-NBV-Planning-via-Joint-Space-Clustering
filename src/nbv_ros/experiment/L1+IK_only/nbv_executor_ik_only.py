#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nbv_executor_ik_only.py

One-shot IK-only NBV executor aligned with the outputs of the IK-only selector.

Overview
--------
This node executes the single best joint-space target selected by the IK-only
Layer-2 baseline. It is designed to stay aligned with the logging format used
by the Action-Mode executor so that experimental CSV files can be compared
directly across methods.

Inputs
------
Required:
- /nbv/ik_psc_best_q
    sensor_msgs/JointState
    Best joint-space solution selected by the IK-only selector.

Optional but recommended:
- /nbv/ik_psc_best_view_pose
    geometry_msgs/PoseArray
    Best selected camera pose, typically of length 1.

- /nbv/ik_psc_selected_index
    std_msgs/Int32
    Selected candidate index in the original valid-view set.

Aligned selector metrics (Float32, latched):
- /nbv/best_mode_score
- /nbv/best_mode_dq_exec
- /nbv/best_mode_m_limit
- /nbv/best_mode_m_sing
- /nbv/best_mode_m_exec

IK statistics from selector (Int32, latched):
- /nbv/ik_success_count
- /nbv/ik_fail_count

Fallback inputs for candidate count and PSC lookup:
- /nbv/valid_view_poses
    geometry_msgs/PoseArray
- /nbv/valid_view_meta
    std_msgs/Float32MultiArray

Outputs
-------
This node does not publish execution commands directly as ROS messages. Instead,
it sends the selected joint target to MoveIt, plans once, executes once, and
writes one CSV file containing one execution record.

Design Notes
------------
- The node is intentionally one-shot by default.
- It waits for the best joint target and optionally waits for aligned metrics
  and selected index before executing.
- If some aligned selector metrics are unavailable after timeout, execution can
  still proceed and missing values are logged as NaN.
- CSV fields are kept aligned with the Action-Mode executor format whenever possible.
"""

import os
import csv
import time
import random
import threading
import numpy as np

import rospy
import moveit_commander

from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray, Int32


def _now_stamp():
    """Return the current ROS time as a float in seconds."""
    return float(rospy.Time.now().to_sec())


def _safe_float(x, default=np.nan):
    """Convert a value to finite float; otherwise return the provided default."""
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _home_path(p):
    """Expand '~' in a filesystem path."""
    return os.path.expanduser(p)


class NBVExecutorIKOnlyAligned(object):
    def __init__(self):
        rospy.init_node("nbv_executor_ik_only", anonymous=False)

        # ---------------- Parameters ----------------
        self.move_group_name = rospy.get_param("~move_group", "fr3_arm")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.one_shot = bool(rospy.get_param("~one_shot", True))

        # Waiting and synchronization gates
        self.wait_best_timeout = float(rospy.get_param("~wait_best_timeout", 8.0))
        self.require_best_mode_metrics = bool(rospy.get_param("~require_best_mode_metrics", True))
        self.require_selected_index = bool(rospy.get_param("~require_selected_index", True))

        # CSV output folder
        default_out_dir = "~/catkin_ws/src/ultralytics_ros/script/experiment/L1+IK_only"
        self.out_dir = _home_path(rospy.get_param("~out_dir", default_out_dir))
        os.makedirs(self.out_dir, exist_ok=True)

        self.trial_tag = str(rospy.get_param("~trial_tag", "ik_only"))
        self.use_comp = int(rospy.get_param("~use_comp", 1))
        self.used_mode0_members = int(rospy.get_param("~used_mode0_members", 0))

        # Meta format for PSC fallback lookup
        self.meta_stride = int(rospy.get_param("~meta_stride", 3))
        self.meta_idx_psc = int(rospy.get_param("~meta_idx_psc", 0))

        # ---------------- MoveIt ----------------
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander(self.move_group_name)

        # ---------------- State buffers ----------------
        self._lock = threading.Lock()

        self._best_q_msg = None
        self._best_pose_msg = None
        self._valid_poses_msg = None
        self._valid_meta_msg = None

        self._best_index = None
        self._best_index_time = 0.0

        # Aligned selector metrics
        self._best_mode_score = np.nan
        self._best_mode_dq_exec = np.nan
        self._best_mode_m_limit = np.nan
        self._best_mode_m_sing = np.nan
        self._best_mode_m_exec = np.nan

        # IK statistics from selector
        self._ik_success_count = np.nan
        self._ik_fail_count = np.nan

        # One-shot completion flag
        self._done = False

        # Waiting start time
        self._wait_start_time = time.time()

        # ---------------- Subscribers ----------------
        rospy.Subscriber("/nbv/ik_psc_best_q", JointState, self._cb_best_q, queue_size=1)
        rospy.Subscriber("/nbv/ik_psc_best_view_pose", PoseArray, self._cb_best_pose, queue_size=1)

        rospy.Subscriber("/nbv/valid_view_poses", PoseArray, self._cb_valid_poses, queue_size=1)
        rospy.Subscriber("/nbv/valid_view_meta", Float32MultiArray, self._cb_valid_meta, queue_size=1)

        # The selected index should be published as Int32
        rospy.Subscriber("/nbv/ik_psc_selected_index", Int32, self._cb_best_index, queue_size=1)

        rospy.Subscriber("/nbv/best_mode_score", Float32, self._cb_best_mode_score, queue_size=1)
        rospy.Subscriber("/nbv/best_mode_dq_exec", Float32, self._cb_best_mode_dq_exec, queue_size=1)
        rospy.Subscriber("/nbv/best_mode_m_limit", Float32, self._cb_best_mode_m_limit, queue_size=1)
        rospy.Subscriber("/nbv/best_mode_m_sing", Float32, self._cb_best_mode_m_sing, queue_size=1)
        rospy.Subscriber("/nbv/best_mode_m_exec", Float32, self._cb_best_mode_m_exec, queue_size=1)

        # IK statistics, typically latched by selector
        rospy.Subscriber("/nbv/ik_success_count", Int32, self._cb_ik_ok, queue_size=1)
        rospy.Subscriber("/nbv/ik_fail_count", Int32, self._cb_ik_fail, queue_size=1)

        rospy.loginfo("[IKOnlyExecutor] Ready. Waiting for /nbv/ik_psc_best_q ...")

        if self.one_shot:
            self._timer = rospy.Timer(rospy.Duration(0.05), self._tick)

    # ---------------- Callbacks ----------------
    def _cb_best_q(self, msg):
        with self._lock:
            self._best_q_msg = msg

    def _cb_best_pose(self, msg):
        with self._lock:
            self._best_pose_msg = msg

    def _cb_valid_poses(self, msg):
        with self._lock:
            self._valid_poses_msg = msg

    def _cb_valid_meta(self, msg):
        with self._lock:
            self._valid_meta_msg = msg

    def _cb_best_index(self, msg):
        with self._lock:
            self._best_index = int(msg.data)
            self._best_index_time = time.time()

    def _cb_best_mode_score(self, msg):
        with self._lock:
            self._best_mode_score = _safe_float(msg.data)

    def _cb_best_mode_dq_exec(self, msg):
        with self._lock:
            self._best_mode_dq_exec = _safe_float(msg.data)

    def _cb_best_mode_m_limit(self, msg):
        with self._lock:
            self._best_mode_m_limit = _safe_float(msg.data)

    def _cb_best_mode_m_sing(self, msg):
        with self._lock:
            self._best_mode_m_sing = _safe_float(msg.data)

    def _cb_best_mode_m_exec(self, msg):
        with self._lock:
            self._best_mode_m_exec = _safe_float(msg.data)

    def _cb_ik_ok(self, msg):
        with self._lock:
            self._ik_success_count = float(int(msg.data))

    def _cb_ik_fail(self, msg):
        with self._lock:
            self._ik_fail_count = float(int(msg.data))

    # ---------------- Main timer tick ----------------
    def _tick(self, _evt):
        if self._done:
            return

        # Take a thread-safe snapshot of all buffered inputs
        with self._lock:
            best_q_msg = self._best_q_msg
            best_pose_msg = self._best_pose_msg
            valid_poses_msg = self._valid_poses_msg
            valid_meta_msg = self._valid_meta_msg

            best_index = self._best_index

            best_mode_score = float(self._best_mode_score)
            best_mode_dq_exec = float(self._best_mode_dq_exec)
            best_mode_m_limit = float(self._best_mode_m_limit)
            best_mode_m_sing = float(self._best_mode_m_sing)
            best_mode_m_exec = float(self._best_mode_m_exec)

            ik_ok = float(self._ik_success_count)
            ik_fail = float(self._ik_fail_count)

        # ---- Waiting gates ----
        if best_q_msg is None:
            # Quiet wait until the best joint-space target becomes available
            return

        if self.require_best_mode_metrics:
            if (not np.isfinite(best_mode_score)) or (not np.isfinite(best_mode_dq_exec)):
                if (time.time() - self._wait_start_time) > self.wait_best_timeout:
                    rospy.logwarn_throttle(1.0, "[IKOnlyExecutor] timeout waiting for best_mode_* metrics")
                    # Continue running so that at least a CSV can still be written
                else:
                    return

        if self.require_selected_index:
            if best_index is None:
                if (time.time() - self._wait_start_time) > self.wait_best_timeout:
                    rospy.logwarn_throttle(1.0, "[IKOnlyExecutor] timeout waiting for /nbv/ik_psc_selected_index")
                    # Continue running so that at least a CSV can still be written
                else:
                    return

        # Execute only once
        self._done = True
        try:
            self._run_once(
                best_q_msg=best_q_msg,
                best_pose_msg=best_pose_msg,
                valid_poses_msg=valid_poses_msg,
                valid_meta_msg=valid_meta_msg,
                best_index=best_index,
                best_mode_score=best_mode_score,
                best_mode_dq_exec=best_mode_dq_exec,
                best_mode_m_limit=best_mode_m_limit,
                best_mode_m_sing=best_mode_m_sing,
                best_mode_m_exec=best_mode_m_exec,
                ik_ok=ik_ok,
                ik_fail=ik_fail,
            )
        except Exception as e:
            rospy.logerr("[IKOnlyExecutor] run_once exception: %s", str(e))
            rospy.signal_shutdown("IK-only executor exception")

    # ---------------- Core execution ----------------
    def _run_once(self,
                  best_q_msg,
                  best_pose_msg,
                  valid_poses_msg,
                  valid_meta_msg,
                  best_index,
                  best_mode_score,
                  best_mode_dq_exec,
                  best_mode_m_limit,
                  best_mode_m_sing,
                  best_mode_m_exec,
                  ik_ok,
                  ik_fail):

        stamp = _now_stamp()

        reps_size = 1
        candidates_size = int(len(valid_poses_msg.poses)) if (valid_poses_msg is not None) else -1
        selected_idx_in_candidates = int(best_index) if best_index is not None else -1

        # Recover selected PSC from Layer-1 meta if possible
        selected_psc = np.nan
        if (selected_idx_in_candidates >= 0) and (valid_meta_msg is not None):
            need = self.meta_stride * (selected_idx_in_candidates + 1)
            if len(valid_meta_msg.data) >= need:
                selected_psc = _safe_float(
                    valid_meta_msg.data[self.meta_stride * selected_idx_in_candidates + self.meta_idx_psc],
                    default=np.nan
                )

        # Align executor-side selected metrics with selector-side summary topics
        selected_dq_exec = best_mode_dq_exec
        score = best_mode_score

        # ---- Planning ----
        q_best = list(best_q_msg.position)
        joint_names = list(best_q_msg.name)

        group_joints = list(self.group.get_active_joints())
        name_to_pos = dict(zip(joint_names, q_best))

        try:
            q_target = [float(name_to_pos[n]) for n in group_joints]
        except KeyError:
            rospy.logwarn("[IKOnlyExecutor] best_q names do not match group joints. Using raw joint order.")
            q_target = [float(x) for x in q_best]

        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        try:
            self.group.set_joint_value_target(q_target)
        except Exception as e:
            rospy.logerr("[IKOnlyExecutor] set_joint_value_target rejected: %s", str(e))
            self._write_csv_row(
                stamp, reps_size, candidates_size, selected_idx_in_candidates,
                selected_psc, selected_dq_exec, score,
                best_mode_score, best_mode_dq_exec, best_mode_m_limit, best_mode_m_sing, best_mode_m_exec,
                ik_ok, ik_fail,
                planning_ok=0, exec_ok=0, traj_points=0, planning_time_sec=np.nan,
                ee_xyz=self._get_ee_xyz(), cam_xyz=self._get_cam_xyz(best_pose_msg)
            )
            rospy.signal_shutdown("IK-only planning rejected")
            return

        t_plan0 = time.time()
        plan = self.group.plan()
        planning_time_sec = float(time.time() - t_plan0)

        if isinstance(plan, tuple):
            planning_ok = 1 if bool(plan[0]) else 0
            traj = plan[1]
        else:
            traj = plan
            planning_ok = 1 if (hasattr(traj, "joint_trajectory") and len(traj.joint_trajectory.points) > 0) else 0

        traj_points = int(len(traj.joint_trajectory.points)) if (planning_ok and hasattr(traj, "joint_trajectory")) else 0

        if planning_ok == 0 or traj_points == 0:
            rospy.logwarn("[IKOnlyExecutor] Planning failed. points=%d time=%.3f", traj_points, planning_time_sec)
            self._write_csv_row(
                stamp, reps_size, candidates_size, selected_idx_in_candidates,
                selected_psc, selected_dq_exec, score,
                best_mode_score, best_mode_dq_exec, best_mode_m_limit, best_mode_m_sing, best_mode_m_exec,
                ik_ok, ik_fail,
                planning_ok=0, exec_ok=0, traj_points=traj_points, planning_time_sec=planning_time_sec,
                ee_xyz=self._get_ee_xyz(), cam_xyz=self._get_cam_xyz(best_pose_msg)
            )
            rospy.signal_shutdown("IK-only planning failed")
            return

        # ---- Execution ----
        rospy.loginfo(
            "[IKOnlyExecutor] Plan OK (%.3fs, %d pts). Executing... | score=%.3f dq_exec=%.3f m_exec=%.3f idx=%d psc=%.4f",
            planning_time_sec, traj_points,
            _safe_float(best_mode_score, np.nan),
            _safe_float(best_mode_dq_exec, np.nan),
            _safe_float(best_mode_m_exec, np.nan),
            int(selected_idx_in_candidates),
            _safe_float(selected_psc, np.nan),
        )

        exec_ok = 0
        try:
            ok = self.group.execute(traj, wait=True)
            self.group.stop()
            exec_ok = 1 if bool(ok) else 0
        except Exception as e:
            rospy.logerr("[IKOnlyExecutor] execute exception: %s", str(e))
            exec_ok = 0

        self._write_csv_row(
            stamp, reps_size, candidates_size, selected_idx_in_candidates,
            selected_psc, selected_dq_exec, score,
            best_mode_score, best_mode_dq_exec, best_mode_m_limit, best_mode_m_sing, best_mode_m_exec,
            ik_ok, ik_fail,
            planning_ok=1, exec_ok=exec_ok, traj_points=traj_points, planning_time_sec=planning_time_sec,
            ee_xyz=self._get_ee_xyz(), cam_xyz=self._get_cam_xyz(best_pose_msg)
        )

        rospy.loginfo("[IKOnlyExecutor] Done. exec_ok=%d. CSV written under %s", exec_ok, self.out_dir)
        rospy.signal_shutdown("IK-only done")

    # ---------------- Pose logging helpers ----------------
    def _get_ee_xyz(self):
        """Return the current end-effector Cartesian position."""
        try:
            ps = self.group.get_current_pose().pose
            return (float(ps.position.x), float(ps.position.y), float(ps.position.z))
        except Exception:
            return (np.nan, np.nan, np.nan)

    @staticmethod
    def _get_cam_xyz(best_pose_msg):
        """Return the selected camera pose position if available."""
        if best_pose_msg is None or (not hasattr(best_pose_msg, "poses")) or len(best_pose_msg.poses) == 0:
            return (np.nan, np.nan, np.nan)
        p = best_pose_msg.poses[0].position
        return (float(p.x), float(p.y), float(p.z))

    # ---------------- CSV writer ----------------
    def _csv_path(self):
        """Create a unique CSV filename for the current execution."""
        stamp = int(time.time())
        rid = random.randint(100000, 999999)
        return os.path.join(self.out_dir, f"nbv_executor_IK_only_{stamp}_{rid}.csv")

    def _write_csv_row(self,
                       stamp, reps_size, candidates_size, selected_idx_in_candidates,
                       selected_psc, selected_dq_exec, score,
                       best_mode_score, best_mode_dq_exec, best_mode_m_limit, best_mode_m_sing, best_mode_m_exec,
                       ik_ok, ik_fail,
                       planning_ok, exec_ok, traj_points, planning_time_sec,
                       ee_xyz, cam_xyz):

        path = self._csv_path()

        header = [
            "stamp", "trial_tag", "reps_size", "used_mode0_members", "candidates_size", "use_comp",
            "selected_idx_in_candidates", "selected_psc", "selected_dq_exec", "score",
            "sel_best_mode_score", "sel_best_mode_dq_exec", "sel_best_mode_m_limit", "sel_best_mode_m_sing", "sel_best_mode_m_exec",
            "ik_success_count", "ik_fail_count",
            "planning_ok", "exec_ok", "traj_points", "planning_time_sec",
            "ee_x", "ee_y", "ee_z", "cam_x", "cam_y", "cam_z",
        ]

        row = [
            _safe_float(stamp, np.nan),
            str(self.trial_tag),
            int(reps_size),
            int(self.used_mode0_members),
            int(candidates_size),
            int(self.use_comp),
            int(selected_idx_in_candidates),
            _safe_float(selected_psc, np.nan),
            _safe_float(selected_dq_exec, np.nan),
            _safe_float(score, np.nan),
            _safe_float(best_mode_score, np.nan),
            _safe_float(best_mode_dq_exec, np.nan),
            _safe_float(best_mode_m_limit, np.nan),
            _safe_float(best_mode_m_sing, np.nan),
            _safe_float(best_mode_m_exec, np.nan),
            _safe_float(ik_ok, np.nan),
            _safe_float(ik_fail, np.nan),
            int(planning_ok),
            int(exec_ok),
            int(traj_points),
            _safe_float(planning_time_sec, np.nan),
            _safe_float(ee_xyz[0], np.nan),
            _safe_float(ee_xyz[1], np.nan),
            _safe_float(ee_xyz[2], np.nan),
            _safe_float(cam_xyz[0], np.nan),
            _safe_float(cam_xyz[1], np.nan),
            _safe_float(cam_xyz[2], np.nan),
        ]

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(row)


if __name__ == "__main__":
    NBVExecutorIKOnlyAligned()
    rospy.spin()