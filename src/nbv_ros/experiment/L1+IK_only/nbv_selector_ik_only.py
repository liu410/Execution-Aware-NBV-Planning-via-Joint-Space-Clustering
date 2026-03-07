#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nbv_selector_ik_only.py

IK-only baseline selector aligned with the Action-Mode selector topic interface.

Overview
--------
This node implements the IK-only baseline used for comparison against the
Action-Mode Layer-2 selector. It does not perform task-space reduction,
joint-space clustering, or mode-level reasoning. Instead, it evaluates all
IK-feasible Layer-1 viewpoints directly and selects the best one using
strict PSC-only ranking.

Selection Rule
--------------
Among all IK-feasible candidates:

    best = argmax PSC

Tie breaking:
- If multiple candidates have the same PSC value, the one with the lower
  original candidate index is selected.

Key Properties
--------------
- No task-space pre-clustering
- No joint-space clustering
- No Action-Mode construction
- No safe-first gating
- Selection criterion is strictly perception-only (PSC)

To stay compatible with the rest of the aligned execution pipeline, this node
still computes and publishes the same best_mode_* summary topics as the
Action-Mode selector. These values are intended mainly for executor-side CSV
alignment rather than for implying that IK-only uses the same scoring logic
internally.

Published Topics
----------------
Executor support:
- /nbv/ik_psc_best_q
    sensor_msgs/JointState
    Best joint-space solution for execution.

- /nbv/ik_psc_best_view_pose
    geometry_msgs/PoseArray
    Best selected camera viewpoint, published as a PoseArray of length 1.

- /nbv/ik_psc_selected_index
    std_msgs/Int32
    Selected candidate index in the original Layer-1 viewpoint set.

- /nbv/ik_psc_selected_psc
    std_msgs/Float32
    Raw PSC value of the selected viewpoint.

- /nbv/ik_success_count
    std_msgs/Int32
    Number of IK-feasible candidates.

- /nbv/ik_fail_count
    std_msgs/Int32
    Number of IK-infeasible candidates.

Action-Mode-aligned summary topics:
- /nbv/best_mode_score
    std_msgs/Float32
    Here this is the normalized PSC score of the selected viewpoint.

- /nbv/best_mode_dq_exec
    std_msgs/Float32
    Execution-oriented joint displacement cost of the selected viewpoint,
    computed with the same definition used by the Action-Mode selector.

- /nbv/best_mode_m_limit
    std_msgs/Float32
    Joint-limit safety margin of the selected viewpoint.

- /nbv/best_mode_m_sing
    std_msgs/Float32
    Singularity safety margin of the selected viewpoint.

- /nbv/best_mode_m_exec
    std_msgs/Float32
    Execution safety score, defined as min(m_limit, m_sing).

Notes
-----
- No safe-first execution threshold is applied. This is intentional so that the
  IK-only baseline remains purely viewpoint-level and differs clearly from the
  Action-Mode selector.
- best_mode_score does not have the same semantic meaning as the fused score in
  the Action-Mode method. Here it is simply the normalized PSC of the selected
  viewpoint.
"""

import rospy
import numpy as np
import moveit_commander
import tf2_ros
import tf.transformations as tft

from geometry_msgs.msg import PoseArray, Pose, PointStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander

from std_msgs.msg import Header, Float32MultiArray, Float32, Int32

import threading
import hashlib


# ============================================================
# IK solver aligned with the Action-Mode selector
# ============================================================
class IKSolverAligned(object):
    """
    IK solver wrapper aligned with the Action-Mode selector implementation.

    Features
    --------
    - MoveIt IK service interface
    - Seed reuse from the last successful solution
    - Pose-based positive cache
    - Negative cache for repeated IK failures
    - Manual joint-limit normalization
    - Camera-pose to end-effector-pose conversion using TF extrinsics
    """

    def __init__(self,
                 move_group="fr3_arm",
                 base_frame="base_link",
                 wrist_link="hand_base_visual_link",
                 camera_link="in_hand_camera_link",
                 ik_timeout=0.15,
                 use_action_seed=True,
                 seed_reset_every_n=80,
                 cache_enabled=True,
                 cache_pos_res=0.003,
                 cache_ang_res_deg=2.0,
                 neg_cache_ttl_sec=3.0,
                 max_cache_size=6000,
                 max_neg_cache_size=6000):

        moveit_commander.roscpp_initialize([])
        self.arm = moveit_commander.MoveGroupCommander(move_group)
        self.robot = RobotCommander()

        self.base_frame = base_frame
        self.wrist_link = wrist_link
        self.camera_link = camera_link
        self.ee_link = wrist_link  # Keep the same convention as the Action-Mode selector

        self.ik_timeout = float(max(0.01, ik_timeout))
        self.use_action_seed = bool(use_action_seed)
        self.seed_reset_every_n = int(max(1, seed_reset_every_n))

        self.cache_enabled = bool(cache_enabled)
        self.cache_pos_res = float(max(1e-4, cache_pos_res))
        self.cache_ang_res = float(max(1e-3, np.deg2rad(cache_ang_res_deg)))
        self.neg_cache_ttl_sec = float(max(0.1, neg_cache_ttl_sec))
        self.max_cache_size = int(max(100, max_cache_size))
        self.max_neg_cache_size = int(max(100, max_neg_cache_size))

        self._solve_calls = 0
        self._last_js = None
        self._last_good_q = None

        self._ik_cache = {}
        self._neg_cache = {}

        rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)

        rospy.loginfo("[IK-ONLY-PSC] Waiting for /compute_ik ...")
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("[IK-ONLY-PSC] /compute_ik connected.")

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        self._T_wrist_cam = None
        self._T_wrist_cam_stamp = rospy.Time(0)

        # Manual joint limits must match the MoveIt active joint names
        self.MANUAL_POS_LIMITS = {
            "j1": (-3.0543261,  3.0543261),
            "j2": (-4.6251,     1.4835),
            "j3": (-2.8274,     2.8274),
            "j4": (-4.6251,     1.4835),
            "j5": (-3.0543,     3.0543),
            "j6": (-3.0543,     3.0543),
        }

        self._joint_limits = {}
        active = list(self.arm.get_active_joints())
        missing = []
        for jn in active:
            if jn in self.MANUAL_POS_LIMITS:
                self._joint_limits[jn] = tuple(self.MANUAL_POS_LIMITS[jn])
            else:
                missing.append(jn)

        rospy.loginfo("[IK-ONLY-PSC] Active joints: %s", ", ".join(active))
        rospy.loginfo("[IK-ONLY-PSC] Manual limits loaded for: %s", ", ".join(sorted(self._joint_limits.keys())))
        if missing:
            rospy.logwarn("[IK-ONLY-PSC] Manual limits missing joints: %s", ", ".join(missing))
            rospy.logwarn("[IK-ONLY-PSC] If MoveIt joint names are not j1~j6, rename MANUAL_POS_LIMITS keys accordingly.")

    def _js_cb(self, msg):
        """Cache the latest joint state."""
        self._last_js = msg

    @staticmethod
    def pose_to_T(pose):
        """Convert a geometry_msgs/Pose into a 4x4 homogeneous transform."""
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        T = tft.quaternion_matrix(q)
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T

    @staticmethod
    def T_to_pose(T):
        """Convert a 4x4 homogeneous transform into a geometry_msgs/Pose."""
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        q = tft.quaternion_from_matrix(T)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return pose

    def lookup_T(self, parent, child, timeout=0.5):
        """Look up a TF transform and convert it into a 4x4 homogeneous matrix."""
        tfm = self.tfbuf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(timeout))
        t = tfm.transform.translation
        r = tfm.transform.rotation
        T = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
        T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
        return T

    def get_T_wrist_cam(self, force_refresh=False):
        """
        Return the wrist-to-camera extrinsic transform.

        A short-lived cache is used because this transform is static in the
        current setup and does not need to be recomputed every call.
        """
        now = rospy.Time.now()
        if (self._T_wrist_cam is not None) and (not force_refresh):
            if (now - self._T_wrist_cam_stamp).to_sec() < 5.0:
                return self._T_wrist_cam
        try:
            T_wc = self.lookup_T(self.wrist_link, self.camera_link, timeout=0.5)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[IK-ONLY-PSC] TF extrinsic %s->%s failed: %s",
                                   self.wrist_link, self.camera_link, str(e))
            return None
        self._T_wrist_cam = T_wc
        self._T_wrist_cam_stamp = now
        return self._T_wrist_cam

    def camera_pose_to_ee_pose(self, cam_pose):
        """
        Convert a camera pose in the base frame into the corresponding
        end-effector pose used for IK solving.
        """
        T_ec = self.get_T_wrist_cam()
        if T_ec is None:
            return None
        T_bc = self.pose_to_T(cam_pose)
        T_be = np.matmul(T_bc, np.linalg.inv(T_ec))
        return self.T_to_pose(T_be)

    def _quat_to_rpy(self, quat_xyzw):
        """Convert quaternion to roll-pitch-yaw."""
        roll, pitch, yaw = tft.euler_from_quaternion(quat_xyzw)
        return float(roll), float(pitch), float(yaw)

    def _wrist_pose_key(self, ee_pose):
        """
        Quantize an end-effector pose into a hashable cache key.

        This reduces repeated IK calls for nearly identical poses.
        """
        px, py, pz = ee_pose.position.x, ee_pose.position.y, ee_pose.position.z
        qx, qy, qz, qw = ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w
        r, p, y = self._quat_to_rpy([qx, qy, qz, qw])

        ipx = int(np.round(px / self.cache_pos_res))
        ipy = int(np.round(py / self.cache_pos_res))
        ipz = int(np.round(pz / self.cache_pos_res))

        ir = int(np.round(r / self.cache_ang_res))
        ip = int(np.round(p / self.cache_ang_res))
        iy = int(np.round(y / self.cache_ang_res))

        return (ipx, ipy, ipz, ir, ip, iy)

    def _neg_cache_hit(self, key, now_sec):
        """Check whether a recent negative-cache entry is still valid."""
        ts = self._neg_cache.get(key, None)
        if ts is None:
            return False
        return (now_sec - ts) < self.neg_cache_ttl_sec

    def _neg_cache_put(self, key, now_sec):
        """Insert an IK failure into the negative cache."""
        if len(self._neg_cache) >= self.max_neg_cache_size:
            self._neg_cache.clear()
        self._neg_cache[key] = now_sec

    def _ik_cache_put(self, key, q):
        """Insert a successful IK solution into the positive cache."""
        if len(self._ik_cache) >= self.max_cache_size:
            self._ik_cache.clear()
        self._ik_cache[key] = q.copy()

    def _make_robot_state_seed(self):
        """
        Build the robot state seed for IK.

        If enabled, the last successful IK result is reused as a seed to
        improve consistency and speed.
        """
        rs = self.robot.get_current_state()
        if (not self.use_action_seed) or (self._last_good_q is None):
            return rs
        if (self._solve_calls % self.seed_reset_every_n) == 0:
            return rs

        joint_names = self.arm.get_active_joints()
        pos_list = list(rs.joint_state.position)
        name_to_idx = {n: i for i, n in enumerate(rs.joint_state.name)}

        for j, name in enumerate(joint_names):
            idx = name_to_idx.get(name, None)
            if idx is not None and j < len(self._last_good_q):
                pos_list[idx] = float(self._last_good_q[j])

        rs.joint_state.position = tuple(pos_list)
        return rs

    def _normalize_q_into_limits(self, q, joint_names, eps=1e-6):
        """
        Normalize joint values into the configured manual limits.

        If a value can be wrapped by ±2π into the valid interval, it is accepted.
        Otherwise the configuration is rejected.
        """
        jl = getattr(self, "_joint_limits", None)
        if not isinstance(jl, dict) or len(jl) == 0:
            return q.copy(), True

        q2 = q.copy()
        for i, name in enumerate(joint_names):
            if name not in jl:
                continue
            lo, hi = jl[name]
            v = float(q2[i])

            if (lo - eps) <= v <= (hi + eps):
                continue

            found = False
            for k in (-2, -1, 0, 1, 2):
                vv = v + k * 2.0 * np.pi
                if (lo - eps) <= vv <= (hi + eps):
                    q2[i] = vv
                    found = True
                    break
            if not found:
                return q2, False

        return q2, True

    def solve_ik_from_camera_pose(self, cam_pose):
        """
        Solve IK for a camera viewpoint by first converting it to the
        corresponding end-effector pose.
        """
        self._solve_calls += 1
        ee_pose = self.camera_pose_to_ee_pose(cam_pose)
        if ee_pose is None:
            return None

        key = None
        now_sec = rospy.Time.now().to_sec()

        if self.cache_enabled:
            try:
                key = self._wrist_pose_key(ee_pose)
            except Exception:
                key = None
            if key is not None:
                q_cached = self._ik_cache.get(key, None)
                if q_cached is not None:
                    return q_cached.copy()
                if self._neg_cache_hit(key, now_sec):
                    return None

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.arm.get_name()
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped.header.frame_id = self.base_frame
        req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        req.ik_request.pose_stamped.pose = ee_pose
        req.ik_request.robot_state = self._make_robot_state_seed()
        req.ik_request.timeout = rospy.Duration(self.ik_timeout)
        req.ik_request.avoid_collisions = False

        try:
            res = self.ik_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, "[IK-ONLY-PSC] IK service call failed: %s", str(e))
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        js = res.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        joint_names = self.arm.get_active_joints()
        try:
            q = np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
        except KeyError:
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        q, ok_norm = self._normalize_q_into_limits(q, joint_names)
        if not ok_norm:
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        self._last_good_q = q.copy()
        if key is not None:
            self._ik_cache_put(key, q)

        return q


# ============================================================
# IK-only selector node (strict PSC-only)
# ============================================================
class IKOnlyAlignedSelectorNode(object):
    """
    IK-only selector node with strict PSC-only ranking.

    Processing pipeline
    -------------------
    1. Receive Layer-1 valid viewpoints.
    2. Read aligned Layer-1 meta information.
    3. Solve IK for all candidates.
    4. Keep only IK-feasible candidates.
    5. Select the best candidate using raw PSC only.
    6. Publish the selected pose, joint target, and aligned summary metrics.
    """

    def __init__(self):
        rospy.init_node("nbv_selector_ik_only_aligned")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.nbv_topic = rospy.get_param("~nbv_topic", "/nbv/valid_view_poses")

        # Layer-1 meta layout
        self.meta_stride = int(rospy.get_param("~meta_stride", 3))
        self.idx_psc = int(rospy.get_param("~meta_idx_psc", 0))
        self.idx_view_dist = int(rospy.get_param("~meta_idx_view_dist", 1))
        self.idx_used_dynamic = int(rospy.get_param("~meta_idx_used_dynamic", 2))

        # Same execution penalty weight used by the Action-Mode selector
        self.pen_w = float(rospy.get_param("~mexec_pen_w", 2.0))

        # Same weighted joint metric used by the Action-Mode selector
        self.joint_weights = rospy.get_param("~joint_weights", [1.5, 2.0, 2.0, 0.8, 0.6, 0.5])
        self.joint_weights = np.asarray(self.joint_weights, dtype=np.float64)

        # Scheduling and buffering
        self._lock = threading.Lock()
        self._latest_msg = None
        self._busy = False

        self._min_period = float(rospy.get_param("~min_process_period", 0.6))
        self._drop_if_busy = bool(rospy.get_param("~drop_if_busy", True))

        self._hash_enable = bool(rospy.get_param("~enable_hash_skip", True))
        self._hash_pos_res = float(rospy.get_param("~hash_pos_res", 0.003))
        self._hash_ang_res_deg = float(rospy.get_param("~hash_ang_res_deg", 2.0))
        self._last_proc_stamp = 0.0
        self._last_hash = None

        # Layer-1 meta buffer
        self._last_meta = None
        self._last_meta_time = rospy.Time(0)
        self._meta_max_age = float(rospy.get_param("~meta_max_age", 1.0))
        rospy.Subscriber("/nbv/valid_view_meta", Float32MultiArray, self._meta_cb, queue_size=1)

        # Target center is used as a processing gate, consistent with the Action-Mode node
        self.target_center = None
        rospy.Subscriber("/apple/center", PointStamped, self._target_cb, queue_size=1)
        rospy.loginfo("[IK-ONLY-PSC] Waiting for /apple/center to enable processing...")

        # IK solver
        self.ik = IKSolverAligned(
            move_group=rospy.get_param("~move_group", "fr3_arm"),
            base_frame=self.base_frame,
            wrist_link=rospy.get_param("~wrist_link", "hand_base_visual_link"),
            camera_link=rospy.get_param("~camera_link", "in_hand_camera_link"),
            ik_timeout=rospy.get_param("~ik_timeout", 0.15),
            use_action_seed=rospy.get_param("~use_action_seed", True),
            seed_reset_every_n=rospy.get_param("~seed_reset_every_n", 80),
            cache_enabled=rospy.get_param("~cache_enabled", True),
            cache_pos_res=rospy.get_param("~cache_pos_res", 0.003),
            cache_ang_res_deg=rospy.get_param("~cache_ang_res_deg", 2.0),
            neg_cache_ttl_sec=rospy.get_param("~neg_cache_ttl_sec", 3.0),
            max_cache_size=rospy.get_param("~max_cache_size", 6000),
            max_neg_cache_size=rospy.get_param("~max_neg_cache_size", 6000),
        )

        # Executor support outputs
        self.best_pose_pub = rospy.Publisher("/nbv/ik_psc_best_view_pose", PoseArray, queue_size=1)
        self.best_q_pub = rospy.Publisher("/nbv/ik_psc_best_q", JointState, queue_size=1)

        # Alignment outputs
        self.selected_index_pub = rospy.Publisher("/nbv/ik_psc_selected_index", Int32, queue_size=1, latch=True)
        self.selected_psc_pub = rospy.Publisher("/nbv/ik_psc_selected_psc", Float32, queue_size=1, latch=True)
        self.ik_success_pub = rospy.Publisher("/nbv/ik_success_count", Int32, queue_size=1, latch=True)
        self.ik_fail_pub = rospy.Publisher("/nbv/ik_fail_count", Int32, queue_size=1, latch=True)

        # Action-Mode-aligned best-mode summary outputs
        self.best_mode_score_pub = rospy.Publisher("/nbv/best_mode_score", Float32, queue_size=1, latch=True)
        self.best_mode_dq_exec_pub = rospy.Publisher("/nbv/best_mode_dq_exec", Float32, queue_size=1, latch=True)
        self.best_mode_m_limit_pub = rospy.Publisher("/nbv/best_mode_m_limit", Float32, queue_size=1, latch=True)
        self.best_mode_m_sing_pub = rospy.Publisher("/nbv/best_mode_m_sing", Float32, queue_size=1, latch=True)
        self.best_mode_m_exec_pub = rospy.Publisher("/nbv/best_mode_m_exec", Float32, queue_size=1, latch=True)

        # Delay subscription to NBVs until target center is available
        self.nbv_sub = None

        # Timer-driven processing
        tick_hz = float(rospy.get_param("~tick_hz", 10.0))
        self._timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, tick_hz)), self._tick)

        rospy.loginfo("[IK-ONLY-PSC] Ready. (STRICT PSC-ONLY selection)")

    # ---------------- Input callbacks ----------------
    def _target_cb(self, msg):
        """
        Receive the target center.

        The NBV subscription is enabled only after the first valid target center
        arrives, which keeps the processing logic consistent with the
        Action-Mode selector pipeline.
        """
        if self.target_center is None:
            rospy.loginfo("[IK-ONLY-PSC] target_center received, enabling NBV subscription: %s", self.nbv_topic)
            self.nbv_sub = rospy.Subscriber(self.nbv_topic, PoseArray, self._nbv_cb, queue_size=1)
        self.target_center = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def _meta_cb(self, msg):
        """Cache the latest Layer-1 meta array."""
        self._last_meta = msg.data[:]
        self._last_meta_time = rospy.Time.now()

    def _nbv_cb(self, msg):
        """Cache the latest valid-view PoseArray."""
        if self.target_center is None:
            return
        with self._lock:
            self._latest_msg = msg

    # ---------------- Scheduling helpers ----------------
    def _hash_pose_array(self, pose_array):
        """
        Compute a quantized hash of the pose array.

        This is used to skip repeated processing of effectively unchanged
        candidate sets.
        """
        ang_res = np.deg2rad(self._hash_ang_res_deg)
        buf = []
        for p in pose_array.poses:
            px = int(np.round(p.position.x / self._hash_pos_res))
            py = int(np.round(p.position.y / self._hash_pos_res))
            pz = int(np.round(p.position.z / self._hash_pos_res))
            q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            r, pitch, yaw = tft.euler_from_quaternion(q)
            ir = int(np.round(r / ang_res))
            ip = int(np.round(pitch / ang_res))
            iy = int(np.round(yaw / ang_res))
            buf.append((px, py, pz, ir, ip, iy))
        m = hashlib.md5()
        m.update(np.asarray(buf, dtype=np.int32).tobytes())
        return m.hexdigest()

    def _tick(self, _evt):
        """Periodic processing entry point."""
        if self.target_center is None:
            return

        now = rospy.Time.now().to_sec()
        if (now - self._last_proc_stamp) < self._min_period:
            return

        with self._lock:
            msg = self._latest_msg
            if msg is None or len(msg.poses) == 0:
                return
            if self._busy and self._drop_if_busy:
                return

            # Hash-based skip
            if self._hash_enable:
                try:
                    h = self._hash_pose_array(msg)
                    if self._last_hash is not None and h == self._last_hash:
                        return
                except Exception:
                    h = None
            else:
                h = None

            # Meta snapshot
            meta = None
            if self._last_meta is not None:
                expected = self.meta_stride * len(msg.poses)
                if len(self._last_meta) >= expected:
                    meta = self._last_meta[:]
                    age = (rospy.Time.now() - self._last_meta_time).to_sec()
                    if age > self._meta_max_age:
                        rospy.logwarn_throttle(1.0, "[IK-ONLY-PSC] meta old (age=%.3fs) but size matches; using.", age)

            self._busy = True
            self._last_proc_stamp = now
            if h is not None:
                self._last_hash = h

        try:
            self._process(msg, meta)
        finally:
            with self._lock:
                self._busy = False

    # ---------------- Math helpers aligned with AM ----------------
    @staticmethod
    def _wrap_diff(a, b):
        """Compute wrap-aware angular difference."""
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _weighted_dq(self, q_a, q_b):
        """
        Compute weighted wrap-aware joint displacement.

        This matches the style used in the Action-Mode selector.
        """
        qa = np.asarray(q_a, dtype=np.float64)
        qb = np.asarray(q_b, dtype=np.float64)
        if qa.shape != qb.shape:
            return 1e18
        if (not np.all(np.isfinite(qa))) or (not np.all(np.isfinite(qb))):
            return 1e18
        dq = self._wrap_diff(qa, qb)
        w = self.joint_weights
        if w.size != dq.size or (not np.all(np.isfinite(w))):
            v = float(np.linalg.norm(dq))
            return v if np.isfinite(v) else 1e18
        v = float(np.sqrt(np.sum((w * dq) ** 2)))
        return v if np.isfinite(v) else 1e18

    def _compute_m_limit(self, q, joint_names):
        """
        Compute the joint-limit safety margin.

        The score is normalized to [0, 1], where 1 means comfortably far from
        limits and 0 means at or beyond a limit.
        """
        jl = getattr(self.ik, "_joint_limits", None)
        if not isinstance(jl, dict) or len(jl) == 0:
            return 0.0

        dist_safe = float(rospy.get_param("~limit_safe_rad", 0.60))
        worst_margin = 1e9
        used_any = False

        for name, v in zip(joint_names, q):
            if name not in jl:
                continue
            used_any = True
            lo, hi = jl[name]
            dist_lo = float(v - lo)
            dist_hi = float(hi - v)
            if dist_lo < -1e-6 or dist_hi < -1e-6:
                return 0.0
            dist = dist_lo if dist_lo < dist_hi else dist_hi
            margin = float(np.clip(dist / max(dist_safe, 1e-6), 0.0, 1.0))
            worst_margin = min(worst_margin, margin)

        if not used_any:
            return 0.0
        return float(worst_margin)

    def _compute_m_sing(self, q):
        """
        Compute the singularity safety margin based on Jacobian condition number.

        This follows the same monotonic mapping strategy as the Action-Mode node.
        """
        try:
            q_list = [float(x) for x in q]
            J = self.ik.arm.get_jacobian_matrix(q_list)
            J = np.asarray(J, dtype=np.float64)
            if J.size == 0:
                return 0.0
            S = np.linalg.svd(J, compute_uv=False)
            if S.size == 0 or np.min(S) < 1e-8:
                cond = float("inf")
            else:
                cond = float(np.max(S) / np.min(S))
        except Exception:
            cond = float("inf")

        cond_good = float(rospy.get_param("~cond_good", 80.0))
        cond_bad = float(rospy.get_param("~cond_bad", 400.0))

        if (not np.isfinite(cond)) or cond <= 0:
            return 0.0
        if cond <= cond_good:
            return 1.0
        if cond >= cond_bad:
            return 0.0
        return float(1.0 - (cond - cond_good) / (cond_bad - cond_good))

    def _compute_exec_terms(self, q_candidate, q_current, joint_names):
        """
        Compute execution-oriented terms for a candidate joint solution.

        Although the IK-only selector does not use these terms to rank
        viewpoints, they are still published for executor-side CSV alignment.
        """
        dq_cost = self._weighted_dq(q_current, q_candidate)
        m_limit = self._compute_m_limit(q_candidate, joint_names)
        m_sing = self._compute_m_sing(q_candidate)
        m_exec = float(min(m_limit, m_sing))
        dq_exec = float(dq_cost + self.pen_w * (1.0 - m_exec))
        return dq_cost, dq_exec, m_limit, m_sing, m_exec

    @staticmethod
    def _robust_minmax(arr, lo_q=10.0, hi_q=90.0, fallback=(0.0, 1.0)):
        """Compute a robust min-max range using percentiles."""
        xs = [float(x) for x in arr if x is not None and np.isfinite(x)]
        if len(xs) == 0:
            return float(fallback[0]), float(fallback[1])
        lo = float(np.percentile(xs, lo_q))
        hi = float(np.percentile(xs, hi_q))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        return lo, hi

    @staticmethod
    def _norm01(x, lo, hi):
        """Normalize a scalar value into [0, 1]."""
        try:
            xv = float(x)
        except Exception:
            return 0.0
        if not np.isfinite(xv):
            return 0.0
        v = (xv - float(lo)) / (float(hi) - float(lo) + 1e-12)
        return float(np.clip(v, 0.0, 1.0))

    # ---------------- Publishing helpers ----------------
    def _publish_best(self, pose_best, q_best):
        """Publish the selected best pose and joint solution."""
        pa = PoseArray()
        pa.header = Header(stamp=rospy.Time.now(), frame_id=self.base_frame)
        pa.poses = [pose_best]
        self.best_pose_pub.publish(pa)

        joint_names = self.ik.arm.get_active_joints()
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.header.frame_id = self.base_frame
        js.name = list(joint_names)
        js.position = [float(x) for x in q_best]
        js.velocity = [0.0] * len(joint_names)
        js.effort = [0.0] * len(joint_names)
        self.best_q_pub.publish(js)

    def _publish_best_mode_metrics(self, score, dq_exec, m_limit, m_sing, m_exec):
        """Publish Action-Mode-aligned best_mode_* summary topics."""
        self.best_mode_score_pub.publish(Float32(data=float(score)))
        self.best_mode_dq_exec_pub.publish(Float32(data=float(dq_exec)))
        self.best_mode_m_limit_pub.publish(Float32(data=float(m_limit)))
        self.best_mode_m_sing_pub.publish(Float32(data=float(m_sing)))
        self.best_mode_m_exec_pub.publish(Float32(data=float(m_exec)))

    # ---------------- Main processing ----------------
    def _process(self, msg, meta_snapshot):
        poses = list(msg.poses)
        N = len(poses)

        # Get current robot joint state
        joint_names = self.ik.arm.get_active_joints()
        q_current = None
        js = self.ik._last_js
        if js is not None:
            name_to_pos = dict(zip(js.name, js.position))
            try:
                q_current = np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
            except KeyError:
                q_current = None

        if q_current is None or (not np.all(np.isfinite(q_current))):
            q_current = (
                self.ik._last_good_q.copy()
                if (self.ik._last_good_q is not None and np.all(np.isfinite(self.ik._last_good_q)))
                else np.zeros((len(joint_names),), dtype=np.float64)
            )

        # Decode Layer-1 PSC values
        psc_list_raw = [None] * N
        if meta_snapshot is not None and len(meta_snapshot) >= self.meta_stride * N:
            for i in range(N):
                base = self.meta_stride * i
                try:
                    psc_list_raw[i] = float(meta_snapshot[base + self.idx_psc])
                except Exception:
                    psc_list_raw[i] = None

        feasible = []
        ok = 0
        fail = 0

        # Evaluate all candidates independently
        for i, pose in enumerate(poses):
            q = self.ik.solve_ik_from_camera_pose(pose)
            if q is None:
                fail += 1
                continue
            ok += 1

            psc = psc_list_raw[i]
            psc = float(psc) if (psc is not None and np.isfinite(psc) and psc >= 0.0) else -1e18

            dq_cost, dq_exec, m_limit, m_sing, m_exec = self._compute_exec_terms(q, q_current, joint_names)

            feasible.append({
                "idx": int(i),
                "pose": pose,
                "q": q,
                "psc": float(psc),
                "dq_cost": float(dq_cost),
                "dq_exec": float(dq_exec),
                "m_limit": float(m_limit),
                "m_sing": float(m_sing),
                "m_exec": float(m_exec),
            })

        # Publish IK feasibility statistics
        self.ik_success_pub.publish(Int32(data=int(ok)))
        self.ik_fail_pub.publish(Int32(data=int(fail)))
        rospy.loginfo("[IK-ONLY-PSC] IK stats: total=%d ok=%d fail=%d", N, ok, fail)

        if not feasible:
            rospy.logwarn_throttle(1.0, "[IK-ONLY-PSC] No IK-feasible NBVs.")
            self._publish_best_mode_metrics(np.nan, np.nan, np.nan, np.nan, np.nan)
            return

        # Normalize PSC for aligned best_mode_score publication
        psc_vals = [it["psc"] for it in feasible]
        psc_lo, psc_hi = self._robust_minmax(psc_vals, fallback=(0.0, 1.0))
        for it in feasible:
            it["psc_norm"] = float(self._norm01(it["psc"], psc_lo, psc_hi))

        # -------- Strict PSC-only selection --------
        feasible.sort(key=lambda it: (-it["psc"], it["idx"]))
        best = feasible[0]

        # Publish selected pose and joint target
        self._publish_best(best["pose"], best["q"])

        # Publish aligned summary metrics
        self._publish_best_mode_metrics(
            score=best["psc_norm"],
            dq_exec=best["dq_exec"],
            m_limit=best["m_limit"],
            m_sing=best["m_sing"],
            m_exec=best["m_exec"],
        )

        # Publish selected candidate index and raw PSC
        self.selected_index_pub.publish(Int32(data=int(best["idx"])))
        if np.isfinite(best["psc"]) and best["psc"] > -1e17:
            self.selected_psc_pub.publish(Float32(data=float(best["psc"])))
        else:
            self.selected_psc_pub.publish(Float32(data=float("nan")))

        rospy.loginfo(
            "[IK-ONLY-PSC] idx=%d | psc=%.6f | psc_norm=%.4f | dq_exec=%.4f | m_limit=%.3f m_sing=%.3f m_exec=%.3f",
            int(best["idx"]), float(best["psc"]), float(best["psc_norm"]), float(best["dq_exec"]),
            float(best["m_limit"]), float(best["m_sing"]), float(best["m_exec"])
        )


if __name__ == "__main__":
    try:
        IKOnlyAlignedSelectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass