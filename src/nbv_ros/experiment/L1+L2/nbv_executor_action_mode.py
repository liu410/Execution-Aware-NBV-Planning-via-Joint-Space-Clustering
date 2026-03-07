#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nbv_executor_action_mode_csv.py

Action-Mode NBV executor with per-trial CSV logging.

Overview
--------
This node is the execution stage of the Action-Mode-based NBV pipeline.

It uses the ranked Action-Mode outputs from the selector and executes one
chosen viewpoint with MoveIt. The main trigger is /nbv/action_modes_reps.
Whenever a new Action-Mode representative set arrives, this node:

1. Prefers the members of the best Action-Mode from /nbv/action_mode0_members
2. Optionally uses aligned PSC metadata from /nbv/action_mode0_member_meta
3. Converts each candidate camera pose to an end-effector pose
4. Solves IK for all candidates
5. Selects the best candidate by minimizing:

       w_dq * Δq_exec - w_psc * PSC

   If PSC metadata is unavailable, the selection falls back to Δq_exec only.

6. Plans in joint space with MoveIt
7. Executes the plan once (unless plan_only is enabled)
8. Writes one CSV row for the trial

Optional selector summaries
---------------------------
This executor can also subscribe to selector-published best-mode summaries:

    /nbv/best_mode_score
    /nbv/best_mode_dq_exec
    /nbv/best_mode_m_limit
    /nbv/best_mode_m_sing
    /nbv/best_mode_m_exec

If these topics are available, their latest values are logged into the
executor CSV. If they are absent, the corresponding CSV fields are written
as NaN.

Design notes
------------
- This executor does not read any selector CSV files.
- The executor is decoupled from RViz-specific visualization logic.
- By default, it behaves as a one-shot executor and shuts down after one run.
"""

import csv
import os
import time

import moveit_commander
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped
from moveit_commander import RobotCommander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray


# ============================================================================
# SE(3) helpers
# ============================================================================
def pose_to_T(pose: Pose):
    """Convert a Pose into a 4x4 homogeneous transform."""
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    T = tft.quaternion_matrix(q)
    T[0, 3] = pose.position.x
    T[1, 3] = pose.position.y
    T[2, 3] = pose.position.z
    return T


def T_to_pose(T):
    """Convert a 4x4 homogeneous transform into a Pose."""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    q = tft.quaternion_from_matrix(T)
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def tf_to_T(tfmsg):
    """Convert a geometry_msgs/TransformStamped into a 4x4 transform."""
    t = tfmsg.transform.translation
    r = tfmsg.transform.rotation

    T = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def wrap_diff(a, b):
    """Return the wrapped smallest signed angular difference per joint."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


# ============================================================================
# Executor
# ============================================================================
class NBVExecutorActionModeCSV(object):
    """Action-Mode executor with MoveIt planning and CSV logging."""

    def __init__(self):
        rospy.init_node("nbv_executor_action_mode_csv")

        # --------------------------------------------------------------------
        # Basic execution parameters
        # --------------------------------------------------------------------
        self.one_shot = bool(rospy.get_param("~one_shot", True))
        self._done = False

        self.move_group_name = rospy.get_param("~move_group", "fr3_arm")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_link = rospy.get_param("~ee_link", "hand_base_visual_link")
        self.camera_link = rospy.get_param("~camera_link", "in_hand_camera_link")
        self.wrist_link = rospy.get_param("~wrist_link", "wrist3_Link")

        self.min_replan_dt = float(rospy.get_param("~min_replan_dt", 0.8))
        self._last_plan_wall = rospy.Time(0)

        # --------------------------------------------------------------------
        # Candidate selection weights
        # --------------------------------------------------------------------
        self.w_dq = float(rospy.get_param("~w_dq", 1.0))
        self.w_psc = float(rospy.get_param("~w_psc", 0.4))
        self.comp_thresh = float(rospy.get_param("~comp_thresh", 0.0))

        joint_weights = rospy.get_param("~joint_weights", [1.5, 2.0, 2.0, 0.8, 0.6, 0.5])
        self.joint_weights = np.asarray(joint_weights, dtype=np.float64)

        # --------------------------------------------------------------------
        # IK and cache parameters
        # --------------------------------------------------------------------
        self.ik_timeout = float(rospy.get_param("~ik_timeout", 0.15))
        self.cache_pos_res = float(rospy.get_param("~cache_pos_res", 0.003))
        self.cache_ang_res_deg = float(rospy.get_param("~cache_ang_res_deg", 2.0))
        self.cache_ang_res = float(max(1e-3, np.deg2rad(self.cache_ang_res_deg)))
        self.cache_enabled = bool(rospy.get_param("~cache_enabled", True))
        self.max_cache_size = int(rospy.get_param("~max_cache_size", 6000))

        # --------------------------------------------------------------------
        # Input topics
        # --------------------------------------------------------------------
        self.topic_reps = rospy.get_param("~reps_topic", "/nbv/action_modes_reps")
        self.topic_members = rospy.get_param("~members_topic", "/nbv/action_mode0_members")
        self.topic_members_meta = rospy.get_param("~members_meta_topic", "/nbv/action_mode0_member_meta")

        # --------------------------------------------------------------------
        # Optional selector summary topics
        # --------------------------------------------------------------------
        self.topic_best_score = rospy.get_param("~best_score_topic", "/nbv/best_mode_score")
        self.topic_best_dqexec = rospy.get_param("~best_dq_exec_topic", "/nbv/best_mode_dq_exec")
        self.topic_best_m_limit = rospy.get_param("~best_m_limit_topic", "/nbv/best_mode_m_limit")
        self.topic_best_m_sing = rospy.get_param("~best_m_sing_topic", "/nbv/best_mode_m_sing")
        self.topic_best_m_exec = rospy.get_param("~best_m_exec_topic", "/nbv/best_mode_m_exec")

        # --------------------------------------------------------------------
        # Planning and execution switches
        # --------------------------------------------------------------------
        self.execute_plan = bool(rospy.get_param("~execute_plan", True))
        self.plan_only = bool(rospy.get_param("~plan_only", False))

        # --------------------------------------------------------------------
        # CSV logging
        # --------------------------------------------------------------------
        self.save_csv = bool(rospy.get_param("~save_csv", True))

        # By default, write the CSV next to this script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = rospy.get_param("~out_dir", script_dir)
        self.trial_tag = rospy.get_param("~trial_tag", "action_mode")

        os.makedirs(self.out_dir, exist_ok=True)
        ts = int(rospy.Time.now().to_sec())
        pid = os.getpid()
        self.csv_path = os.path.join(
            self.out_dir,
            f"nbv_executor_{self.trial_tag}_{ts}_{pid}.csv"
        )

        self._csv_file = None
        self._csv_writer = None

        if self.save_csv:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                # Basic executor information
                "stamp",
                "trial_tag",
                "reps_size",
                "used_mode0_members",
                "candidates_size",
                "use_comp",
                "selected_idx_in_candidates",
                "selected_psc",
                "selected_dq_exec",
                "score",

                # Optional selector summaries
                "sel_best_mode_score",
                "sel_best_mode_dq_exec",
                "sel_best_mode_m_limit",
                "sel_best_mode_m_sing",
                "sel_best_mode_m_exec",

                # IK statistics
                "ik_success_count",
                "ik_fail_count",

                # Planning and execution status
                "planning_ok",
                "exec_ok",
                "traj_points",
                "planning_time_sec",

                # Selected EE pose
                "ee_x", "ee_y", "ee_z",

                # Selected camera pose
                "cam_x", "cam_y", "cam_z",
            ])
            self._csv_file.flush()
            rospy.loginfo("[Executor] Saving CSV to %s", self.csv_path)

        rospy.on_shutdown(self._on_shutdown)

        # --------------------------------------------------------------------
        # MoveIt initialization
        # --------------------------------------------------------------------
        moveit_commander.roscpp_initialize([])

        self.group = moveit_commander.MoveGroupCommander(self.move_group_name)
        self.robot = RobotCommander()

        self.group.set_planner_id(rospy.get_param("~planner_id", "RRTConnectkConfigDefault"))
        self.group.set_planning_time(float(rospy.get_param("~planning_time", 8.0)))
        self.group.set_num_planning_attempts(int(rospy.get_param("~num_planning_attempts", 80)))
        self.group.allow_replanning(True)

        self.group.set_max_velocity_scaling_factor(float(rospy.get_param("~vel_scale", 0.4)))
        self.group.set_max_acceleration_scaling_factor(float(rospy.get_param("~acc_scale", 0.4)))

        self.group.set_goal_position_tolerance(float(rospy.get_param("~pos_tol", 0.005)))
        self.group.set_goal_orientation_tolerance(float(rospy.get_param("~ori_tol", 0.3)))

        # --------------------------------------------------------------------
        # TF
        # --------------------------------------------------------------------
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        self._T_ee_cam = None
        self._T_ee_cam_stamp = rospy.Time(0)

        # --------------------------------------------------------------------
        # IK service
        # --------------------------------------------------------------------
        rospy.loginfo("[Executor] Waiting for /compute_ik ...")
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("[Executor] /compute_ik connected.")

        self._last_js = None
        rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)
        self._ik_cache = {}

        # --------------------------------------------------------------------
        # Candidate buffers
        # --------------------------------------------------------------------
        self._mode0_members = None
        self._mode0_members_meta = None

        rospy.Subscriber(self.topic_members, PoseArray, self._cb_members, queue_size=1)
        rospy.Subscriber(self.topic_members_meta, Float32MultiArray, self._cb_members_meta, queue_size=1)

        # Optional target center subscription
        self.target_center = None
        rospy.Subscriber("/apple/center", PointStamped, self._cb_target, queue_size=1)

        # --------------------------------------------------------------------
        # Selector summary cache
        # --------------------------------------------------------------------
        self.sel_best_mode_score = np.nan
        self.sel_best_mode_dq_exec = np.nan
        self.sel_best_mode_m_limit = np.nan
        self.sel_best_mode_m_sing = np.nan
        self.sel_best_mode_m_exec = np.nan
        self._sel_last_stamp = rospy.Time(0)

        rospy.Subscriber(self.topic_best_score, Float32, self._cb_sel_best_score, queue_size=1)
        rospy.Subscriber(self.topic_best_dqexec, Float32, self._cb_sel_best_dqexec, queue_size=1)
        rospy.Subscriber(self.topic_best_m_limit, Float32, self._cb_sel_best_m_limit, queue_size=1)
        rospy.Subscriber(self.topic_best_m_sing, Float32, self._cb_sel_best_m_sing, queue_size=1)
        rospy.Subscriber(self.topic_best_m_exec, Float32, self._cb_sel_best_m_exec, queue_size=1)

        # Main trigger
        rospy.Subscriber(self.topic_reps, PoseArray, self._cb_reps, queue_size=1)

        rospy.loginfo(
            "\n[Executor] Ready (Action-Mode CSV)\n"
            "  move_group   = %s\n"
            "  ee_link      = %s\n"
            "  camera_link  = %s\n"
            "  reps_topic   = %s\n"
            "  members_topic= %s (optional)\n"
            "  members_meta = %s (optional PSC)\n"
            "  selection    = w_dq*Δq_exec - w_psc*PSC (PSC optional)\n"
            "  w_dq=%.3f w_psc=%.3f comp_thresh=%.3f\n"
            "  selector summaries (optional):\n"
            "    %s, %s, %s, %s, %s\n"
            "  execute_plan=%s plan_only=%s one_shot=%s\n",
            self.move_group_name,
            self.ee_link,
            self.camera_link,
            self.topic_reps,
            self.topic_members,
            self.topic_members_meta,
            self.w_dq,
            self.w_psc,
            self.comp_thresh,
            self.topic_best_score,
            self.topic_best_dqexec,
            self.topic_best_m_limit,
            self.topic_best_m_sing,
            self.topic_best_m_exec,
            str(self.execute_plan),
            str(self.plan_only),
            str(self.one_shot),
        )

    # ------------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------------
    def _on_shutdown(self):
        """Flush and close the CSV file on shutdown."""
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Basic callbacks
    # ------------------------------------------------------------------------
    def _cb_target(self, msg: PointStamped):
        """Cache the latest target center."""
        self.target_center = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def _js_cb(self, msg: JointState):
        """Cache the latest joint state."""
        self._last_js = msg

    def _cb_members(self, msg: PoseArray):
        """Cache the members of the best Action-Mode."""
        self._mode0_members = msg

    def _cb_members_meta(self, msg: Float32MultiArray):
        """Cache metadata aligned with mode-0 members."""
        self._mode0_members_meta = list(msg.data)

    # ------------------------------------------------------------------------
    # Selector summary callbacks
    # ------------------------------------------------------------------------
    def _touch_sel_stamp(self):
        """Update the timestamp for selector summary reception."""
        self._sel_last_stamp = rospy.Time.now()

    def _cb_sel_best_score(self, msg: Float32):
        self.sel_best_mode_score = float(msg.data)
        self._touch_sel_stamp()

    def _cb_sel_best_dqexec(self, msg: Float32):
        self.sel_best_mode_dq_exec = float(msg.data)
        self._touch_sel_stamp()

    def _cb_sel_best_m_limit(self, msg: Float32):
        self.sel_best_mode_m_limit = float(msg.data)
        self._touch_sel_stamp()

    def _cb_sel_best_m_sing(self, msg: Float32):
        self.sel_best_mode_m_sing = float(msg.data)
        self._touch_sel_stamp()

    def _cb_sel_best_m_exec(self, msg: Float32):
        self.sel_best_mode_m_exec = float(msg.data)
        self._touch_sel_stamp()

    # ------------------------------------------------------------------------
    # Joint-state helper
    # ------------------------------------------------------------------------
    def _get_current_q(self):
        """Extract the current active-joint vector from the latest JointState."""
        js = self._last_js
        if js is None:
            return None

        name_to_pos = dict(zip(js.name, js.position))
        joint_names = self.group.get_active_joints()
        try:
            return np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
        except KeyError:
            return None

    # ------------------------------------------------------------------------
    # TF utilities
    # ------------------------------------------------------------------------
    def _lookup_T(self, target_frame, source_frame, timeout=0.3):
        """Look up a TF transform and return it as a 4x4 matrix."""
        t = rospy.Time(0)
        if not self.tfbuf.can_transform(target_frame, source_frame, t, rospy.Duration(timeout)):
            return None
        tfm = self.tfbuf.lookup_transform(target_frame, source_frame, t, rospy.Duration(timeout))
        return tf_to_T(tfm)

    def _get_T_ee_cam(self):
        """
        Get the cached EE-to-camera transform.

        Priority:
        1. Direct ee_link -> camera_link
        2. Reconstruct through wrist_link if needed
        """
        now = rospy.Time.now()

        if self._T_ee_cam is not None and (now - self._T_ee_cam_stamp).to_sec() < 5.0:
            return self._T_ee_cam

        # Direct ee -> camera lookup
        T_ec = self._lookup_T(self.ee_link, self.camera_link, timeout=0.5)
        if T_ec is not None:
            self._T_ee_cam = T_ec
            self._T_ee_cam_stamp = now
            return self._T_ee_cam

        # Fallback through the wrist frame
        T_we = self._lookup_T(self.wrist_link, self.ee_link, timeout=0.5)
        T_wc = self._lookup_T(self.wrist_link, self.camera_link, timeout=0.5)
        if (T_we is not None) and (T_wc is not None):
            self._T_ee_cam = np.linalg.inv(T_we) @ T_wc
            self._T_ee_cam_stamp = now
            return self._T_ee_cam

        rospy.logwarn_throttle(
            1.0,
            "[Executor] TF not available for T_ee_cam: %s <- %s (direct or via %s)",
            self.ee_link,
            self.camera_link,
            self.wrist_link,
        )
        return None

    def make_ee_pose_from_viewpoint(self, cam_pose: Pose):
        """
        Convert a camera viewpoint pose into the corresponding EE pose.
        """
        T_ec = self._get_T_ee_cam()
        if T_ec is None:
            return None

        T_bc = pose_to_T(cam_pose)
        T_be = T_bc @ np.linalg.inv(T_ec)
        return T_to_pose(T_be)

    # ------------------------------------------------------------------------
    # IK cache key
    # ------------------------------------------------------------------------
    def _quat_to_rpy(self, quat_xyzw):
        """Convert quaternion to roll-pitch-yaw."""
        return tft.euler_from_quaternion(quat_xyzw)

    def _ee_pose_key(self, ee_pose: Pose):
        """
        Build a discretized cache key for one EE pose.
        """
        px, py, pz = ee_pose.position.x, ee_pose.position.y, ee_pose.position.z
        qx, qy, qz, qw = (
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        )
        r, p, y = self._quat_to_rpy([qx, qy, qz, qw])

        ipx = int(np.round(px / self.cache_pos_res))
        ipy = int(np.round(py / self.cache_pos_res))
        ipz = int(np.round(pz / self.cache_pos_res))

        ir = int(np.round(r / self.cache_ang_res))
        ip = int(np.round(p / self.cache_ang_res))
        iy = int(np.round(y / self.cache_ang_res))

        return ipx, ipy, ipz, ir, ip, iy

    def _solve_ik_ee(self, ee_pose: Pose):
        """
        Solve IK for an EE pose, with optional caching.
        """
        if ee_pose is None:
            return None

        key = None
        if self.cache_enabled:
            try:
                key = self._ee_pose_key(ee_pose)
                if key in self._ik_cache:
                    return self._ik_cache[key].copy()
            except Exception:
                key = None

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group.get_name()
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        req.ik_request.pose_stamped.pose = ee_pose
        req.ik_request.robot_state = self.robot.get_current_state()
        req.ik_request.timeout = rospy.Duration(self.ik_timeout)
        req.ik_request.avoid_collisions = True

        try:
            res = self.ik_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, "[Executor] IK call failed: %s", str(e))
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            return None

        js = res.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        joint_names = self.group.get_active_joints()

        try:
            q = np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
        except KeyError:
            return None

        if self.cache_enabled and key is not None:
            if len(self._ik_cache) >= self.max_cache_size:
                self._ik_cache.clear()
            self._ik_cache[key] = q.copy()

        return q

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------
    def _dq_exec(self, q_target, q_current):
        """
        Compute weighted wrap-aware joint displacement:

            dq = wrap(q_target - q_current)
            || W * dq ||_2

        where W is a diagonal weight vector.
        """
        if q_target is None or q_current is None:
            return None

        qt = np.asarray(q_target, dtype=np.float64).reshape(-1)
        qc = np.asarray(q_current, dtype=np.float64).reshape(-1)

        if qt.shape != qc.shape or qt.size == 0:
            return None

        dq = wrap_diff(qt, qc)
        if self.joint_weights.shape[0] == dq.shape[0]:
            w = self.joint_weights
            return float(np.sqrt(np.sum((w * dq) ** 2)))

        return float(np.linalg.norm(dq))

    def _set_joint_target(self, q_target: np.ndarray):
        """
        Set a MoveIt joint target directly from a joint vector.
        """
        if q_target is None:
            return False

        q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
        joint_names = self.group.get_active_joints()

        if q_target.shape[0] != len(joint_names):
            rospy.logwarn(
                "[Executor] q_target size mismatch: got %d, expected %d",
                int(q_target.shape[0]),
                int(len(joint_names)),
            )
            return False

        try:
            self.group.set_joint_value_target([float(x) for x in q_target.tolist()])
            return True
        except Exception as e:
            rospy.logwarn("[Executor] set_joint_value_target failed: %s", str(e))
            return False

    # ------------------------------------------------------------------------
    # Selection inside the best Action-Mode
    # ------------------------------------------------------------------------
    def _select_best_viewpoint(self, cam_poses, comp_list):
        """
        Select the best camera viewpoint among candidate poses.

        Selection rule:
            minimize w_dq * Δq_exec - w_psc * PSC

        If PSC metadata is unavailable, only Δq_exec is used.

        Returns
        -------
        best_idx : int or None
            Index of selected candidate in cam_poses
        ee_poses : list
            EE poses converted from camera poses
        qs : list
            IK solutions corresponding to candidates
        dbg : dict
            Debug information
        """
        q_cur = self._get_current_q()

        ee_poses = []
        qs = []
        scores = []
        dq_execs = []

        use_comp = (comp_list is not None) and (len(comp_list) == len(cam_poses))

        ik_ok = 0
        ik_fail = 0

        for i, cam_pose in enumerate(cam_poses):
            ee_pose = self.make_ee_pose_from_viewpoint(cam_pose)
            ee_poses.append(ee_pose)

            q = self._solve_ik_ee(ee_pose) if ee_pose is not None else None
            qs.append(q)

            if q is None:
                ik_fail += 1
                dq_execs.append(None)
                scores.append(np.inf)
                continue

            ik_ok += 1

            dq = self._dq_exec(q, q_cur)
            dq_execs.append(dq)

            comp = float(comp_list[i]) if use_comp else None

            # Optional hard gate on the completion / PSC-like term
            if use_comp and (comp is not None) and np.isfinite(comp) and (comp < self.comp_thresh):
                scores.append(np.inf)
                continue

            score = 0.0

            if dq is not None and np.isfinite(dq):
                score += self.w_dq * float(dq)

            if use_comp and (comp is not None) and np.isfinite(comp):
                score -= self.w_psc * float(comp)

            scores.append(float(score))

        best_idx = None
        if len(scores) > 0:
            best_idx = int(np.argmin(np.asarray(scores)))
            if not np.isfinite(scores[best_idx]):
                best_idx = None

        dbg = {
            "use_comp": use_comp,
            "scores": scores,
            "dq_execs": dq_execs,
            "has_q_current": (q_cur is not None),
            "ik_ok": ik_ok,
            "ik_fail": ik_fail,
        }
        return best_idx, ee_poses, qs, dbg

    # ------------------------------------------------------------------------
    # Main trigger callback
    # ------------------------------------------------------------------------
    def _cb_reps(self, msg: PoseArray):
        """
        Main execution callback triggered by Action-Mode representatives.
        """
        now = rospy.Time.now()

        if (now - self._last_plan_wall).to_sec() < self.min_replan_dt:
            return
        self._last_plan_wall = now

        if not msg.poses:
            return

        if self.one_shot and self._done:
            return

        reps_size = len(msg.poses)

        # Prefer best-mode members if available.
        used_mode0 = False
        if self._mode0_members is not None and self._mode0_members.poses:
            candidates = list(self._mode0_members.poses)
            comp_list = self._mode0_members_meta
            used_mode0 = True
        else:
            candidates = [msg.poses[0]]
            comp_list = None
            used_mode0 = False

        candidates_size = len(candidates)

        # --------------------------------------------------------------------
        # Candidate selection
        # --------------------------------------------------------------------
        best_idx, ee_poses, qs, dbg = self._select_best_viewpoint(candidates, comp_list)

        if best_idx is None:
            rospy.logwarn("[Executor] No valid candidate after selection.")
            self._write_csv_row(
                stamp=now.to_sec(),
                reps_size=reps_size,
                used_mode0_members=used_mode0,
                candidates_size=candidates_size,
                use_comp=bool(dbg.get("use_comp", False)),
                selected_idx=-1,
                selected_psc=np.nan,
                selected_dq=np.nan,
                score=np.nan,
                ik_ok=int(dbg.get("ik_ok", 0)),
                ik_fail=int(dbg.get("ik_fail", 0)),
                planning_ok=False,
                exec_ok=False,
                traj_points=0,
                planning_time=np.nan,
                ee_pose=None,
                cam_pose=None,
            )
            return

        cam_pose = candidates[best_idx]
        ee_pose = ee_poses[best_idx]
        q_target = qs[best_idx] if qs is not None else None

        if ee_pose is None or q_target is None:
            rospy.logwarn("[Executor] Selected candidate missing ee_pose/q_target.")
            self._write_csv_row(
                stamp=now.to_sec(),
                reps_size=reps_size,
                used_mode0_members=used_mode0,
                candidates_size=candidates_size,
                use_comp=bool(dbg.get("use_comp", False)),
                selected_idx=int(best_idx),
                selected_psc=np.nan,
                selected_dq=np.nan,
                score=np.nan,
                ik_ok=int(dbg.get("ik_ok", 0)),
                ik_fail=int(dbg.get("ik_fail", 0)),
                planning_ok=False,
                exec_ok=False,
                traj_points=0,
                planning_time=np.nan,
                ee_pose=ee_pose,
                cam_pose=cam_pose,
            )
            return

        # --------------------------------------------------------------------
        # Selected metadata
        # --------------------------------------------------------------------
        use_comp = bool(dbg.get("use_comp", False))

        selected_psc = np.nan
        if use_comp and comp_list is not None and len(comp_list) == len(candidates):
            try:
                selected_psc = float(comp_list[best_idx])
            except Exception:
                selected_psc = np.nan

        q_cur = self._get_current_q()
        selected_dq = self._dq_exec(q_target, q_cur)
        selected_dq = float(selected_dq) if (selected_dq is not None and np.isfinite(selected_dq)) else np.nan

        score = dbg.get("scores", [np.nan])[best_idx]
        score = float(score) if (score is not None and np.isfinite(score)) else np.nan

        rospy.loginfo(
            "[Executor] Selected idx=%d/%d | used_mode0=%s | use_comp=%s | PSC=%s | dq_exec=%s | score=%s",
            int(best_idx),
            int(candidates_size),
            str(used_mode0),
            str(use_comp),
            ("%.4f" % selected_psc) if np.isfinite(selected_psc) else "nan",
            ("%.4f" % selected_dq) if np.isfinite(selected_dq) else "nan",
            ("%.4f" % score) if np.isfinite(score) else "nan",
        )

        # --------------------------------------------------------------------
        # Planning and execution
        # --------------------------------------------------------------------
        self.group.stop()
        self.group.clear_pose_targets()
        self.group.set_start_state_to_current_state()

        ok_target = self._set_joint_target(q_target)

        # Fallback to pose target if direct joint target is rejected.
        if not ok_target:
            rospy.logwarn("[Executor] Joint target rejected by MoveIt. Falling back to pose target.")
            ps = PoseStamped()
            ps.header.frame_id = self.base_frame
            ps.header.stamp = rospy.Time.now()
            ps.pose = ee_pose
            self.group.set_pose_target(ps, self.ee_link)

        t0 = time.time()
        plan = self.group.plan()
        planning_time = float(time.time() - t0)

        # MoveIt Python APIs may return either:
        #   - a tuple (ok, traj, time, error_code)
        #   - a RobotTrajectory directly
        traj_msg = None
        ok_plan = False

        if isinstance(plan, tuple):
            ok_plan = bool(plan[0])
            traj_msg = plan[1]
        else:
            traj_msg = plan
            ok_plan = True

        if (
            traj_msg is None
            or (not hasattr(traj_msg, "joint_trajectory"))
            or (len(traj_msg.joint_trajectory.points) == 0)
        ):
            ok_plan = False

        traj_points = int(len(traj_msg.joint_trajectory.points)) if ok_plan else 0

        exec_ok = False
        do_exec = bool(self.execute_plan) and (not bool(self.plan_only))
        if ok_plan and do_exec:
            exec_ok = bool(self.group.execute(traj_msg, wait=True))
            self.group.stop()

        # --------------------------------------------------------------------
        # CSV logging
        # --------------------------------------------------------------------
        self._write_csv_row(
            stamp=now.to_sec(),
            reps_size=reps_size,
            used_mode0_members=used_mode0,
            candidates_size=candidates_size,
            use_comp=use_comp,
            selected_idx=int(best_idx),
            selected_psc=selected_psc,
            selected_dq=selected_dq,
            score=score,
            ik_ok=int(dbg.get("ik_ok", 0)),
            ik_fail=int(dbg.get("ik_fail", 0)),
            planning_ok=bool(ok_plan),
            exec_ok=bool(exec_ok) if do_exec else False,
            traj_points=traj_points,
            planning_time=planning_time,
            ee_pose=ee_pose,
            cam_pose=cam_pose,
        )

        self._done = True
        self.group.clear_pose_targets()

        if self.one_shot:
            rospy.loginfo("[Executor] One-shot finished. Shutting down.")
            rospy.signal_shutdown("one-shot done")

    # ------------------------------------------------------------------------
    # CSV writer
    # ------------------------------------------------------------------------
    def _write_csv_row(
        self,
        stamp,
        reps_size,
        used_mode0_members,
        candidates_size,
        use_comp,
        selected_idx,
        selected_psc,
        selected_dq,
        score,
        ik_ok,
        ik_fail,
        planning_ok,
        exec_ok,
        traj_points,
        planning_time,
        ee_pose,
        cam_pose,
    ):
        """
        Write one CSV row for one execution trial.
        """
        if not self.save_csv or self._csv_writer is None:
            return

        # Selected EE pose
        if ee_pose is not None:
            ee_x, ee_y, ee_z = ee_pose.position.x, ee_pose.position.y, ee_pose.position.z
        else:
            ee_x = ee_y = ee_z = np.nan

        # Selected camera pose
        if cam_pose is not None:
            cam_x, cam_y, cam_z = cam_pose.position.x, cam_pose.position.y, cam_pose.position.z
        else:
            cam_x = cam_y = cam_z = np.nan

        # Latest selector summaries
        sel_score = self.sel_best_mode_score
        sel_dqexec = self.sel_best_mode_dq_exec
        sel_mlim = self.sel_best_mode_m_limit
        sel_msing = self.sel_best_mode_m_sing
        sel_mexec = self.sel_best_mode_m_exec

        row = [
            float(stamp),
            str(self.trial_tag),
            int(reps_size),
            int(bool(used_mode0_members)),
            int(candidates_size),
            int(bool(use_comp)),
            int(selected_idx),

            float(selected_psc) if np.isfinite(selected_psc) else np.nan,
            float(selected_dq) if np.isfinite(selected_dq) else np.nan,
            float(score) if np.isfinite(score) else np.nan,

            float(sel_score) if np.isfinite(sel_score) else np.nan,
            float(sel_dqexec) if np.isfinite(sel_dqexec) else np.nan,
            float(sel_mlim) if np.isfinite(sel_mlim) else np.nan,
            float(sel_msing) if np.isfinite(sel_msing) else np.nan,
            float(sel_mexec) if np.isfinite(sel_mexec) else np.nan,

            int(ik_ok),
            int(ik_fail),
            int(bool(planning_ok)),
            int(bool(exec_ok)),
            int(traj_points),
            float(planning_time) if np.isfinite(planning_time) else np.nan,

            float(ee_x),
            float(ee_y),
            float(ee_z),

            float(cam_x),
            float(cam_y),
            float(cam_z),
        ]

        try:
            self._csv_writer.writerow(row)
            self._csv_file.flush()
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[Executor] CSV write failed: %s", str(e))


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    try:
        NBVExecutorActionModeCSV()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass