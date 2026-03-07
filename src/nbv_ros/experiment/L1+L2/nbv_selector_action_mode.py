#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nbv_selector_action_mode.py

Layer-2 execution-aware NBV selection via joint-space Action-Modes.

Overview
--------
This module implements the execution-aware selection layer of the NBV pipeline.

Its input is the set of geometrically valid NBV candidates produced by Layer-1.
Its responsibility is to map those task-space NBV candidates into joint space,
organize feasible IK solutions into joint-space Action-Modes, and rank the modes
using execution-oriented criteria.

Compared with Layer-1, this module explicitly considers:
    - IK feasibility
    - joint-space clustering
    - execution motion cost
    - joint-limit safety margin
    - singularity safety margin
    - execution-oriented representative selection

Main pipeline
-------------
1. Receive Layer-1 NBV candidates from /nbv/valid_view_poses
2. Optionally reduce redundant task-space candidates
3. Solve IK for candidate camera poses
4. Cluster feasible IK solutions in joint space using DBSCAN
5. Build Action-Modes from joint-space clusters
6. Select a mode-internal execution representative
7. Rank Action-Modes using:
       - PSC
       - mode compactness
       - execution cost
       - execution safety margin
8. Publish the best mode summary and representative poses

Published outputs
-----------------
/nbv/action_modes_reps            : geometry_msgs/PoseArray
    Representative poses of ranked Action-Modes

/nbv/action_mode0_members         : geometry_msgs/PoseArray
    Members of the highest-ranked Action-Mode

/nbv/action_mode0_member_meta     : std_msgs/Float32MultiArray
    Optional PSC values aligned with /nbv/action_mode0_members

/nbv/action_modes_markers         : visualization_msgs/MarkerArray
    RViz markers for ranked Action-Modes

/nbv/best_mode_score              : std_msgs/Float32
/nbv/best_mode_dq_exec            : std_msgs/Float32
/nbv/best_mode_m_limit            : std_msgs/Float32
/nbv/best_mode_m_sing             : std_msgs/Float32
/nbv/best_mode_m_exec             : std_msgs/Float32

Design note
-----------
Layer-1 remains purely geometric and perception-driven.
Layer-2 is the first stage that explicitly reasons in joint space.

This separation keeps the full NBV framework modular, interpretable,
and easier to analyze experimentally.
"""

import hashlib
import threading

import moveit_commander
import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from moveit_commander import RobotCommander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from sklearn.cluster import DBSCAN
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================================
# Basic data structures
# ============================================================================
class NBV(object):
    """Container for one NBV candidate and its Layer-1 metadata."""

    def __init__(self, index, pose):
        self.index = index
        self.pose = pose

        # Layer-1 metadata
        self.psc = None
        self.view_dist = None
        self.used_dynamic = None


class NBVCluster(object):
    """Joint-space Action-Mode container."""

    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.nbvs = []
        self.q_members = []

        self.q_center = None
        self.rep_nbv = None
        self.rep_q = None

        # Intra-cluster spread statistics
        self.dq_max = None
        self.dq_mean = None
        self.dq_std = None

        # Execution-aware metrics
        self.dq_exec = None
        self.m_limit = None
        self.m_sing = None
        self.m_exec = None

        # Final mode-internal execution choice
        self.q_best = None
        self.best_nbv = None

        # Ranking-related normalized terms
        self.psc_norm = None
        self.dq_mean_norm = None
        self.dq_exec_norm = None
        self.m_exec_norm = None
        self.score = None

        # Assigned after sorting
        self.mode_id = None

        self.is_singleton = False


# ============================================================================
# Task-space pre-clusterer
# ============================================================================
class TaskSpaceClusterer(object):
    """
    Lightweight task-space pre-clusterer.

    This stage reduces redundant Layer-1 NBVs before expensive IK solving.
    The clustering is based on:
        - viewpoint position
        - viewing direction toward target center

    Representatives are chosen using Layer-1 metadata when available.
    """

    def __init__(
        self,
        eps_pos=0.06,
        eps_dir=0.35,
        min_samples=3,
        max_rep_per_cluster=2,
        noise_rep=8,
        w_psc=1.0,
        w_vdist=0.2,
        vdist_pref=0.30,
        vdist_sigma=0.10,
    ):
        self.eps_pos = float(eps_pos)
        self.eps_dir = float(eps_dir)
        self.min_samples = int(min_samples)
        self.max_rep = int(max_rep_per_cluster)
        self.noise_rep = int(noise_rep)

        self.w_psc = float(w_psc)
        self.w_vdist = float(w_vdist)
        self.vdist_pref = float(vdist_pref)
        self.vdist_sigma = float(max(1e-6, vdist_sigma))

    def _pose_to_feat(self, pose, target_center):
        """Convert a pose into position and normalized target-facing direction."""
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        view_dir = target_center - pos
        view_dir /= (np.linalg.norm(view_dir) + 1e-9)
        return pos, view_dir

    @staticmethod
    def _safe_float(x, default=None):
        """Convert to finite float or return default."""
        if x is None:
            return default
        try:
            val = float(x)
        except Exception:
            return default
        if not np.isfinite(val):
            return default
        return val

    def _nbv_meta_score(self, nbv, stats):
        """Build a metadata-based score for representative preference."""
        p_lo, p_hi, vd_lo, vd_hi = stats
        psc = self._safe_float(getattr(nbv, "psc", None), None)
        vd = self._safe_float(getattr(nbv, "view_dist", None), None)

        if psc is None and vd is None:
            return -1e9

        if psc is not None and psc >= 0.0:
            p_n = (psc - p_lo) / (p_hi - p_lo + 1e-12)
            p_n = float(np.clip(p_n, 0.0, 1.0))
        else:
            p_n = 0.0

        if vd is not None and vd >= 0.0:
            z = (vd - self.vdist_pref) / self.vdist_sigma
            vd_n = float(np.exp(-0.5 * z * z))
        else:
            vd_n = 0.0

        return float(self.w_psc * p_n + self.w_vdist * vd_n)

    def _build_norm_stats(self, nbvs):
        """Build robust normalization ranges for metadata-based scoring."""
        psc_vals = []
        vd_vals = []

        for nbv in nbvs:
            p = self._safe_float(getattr(nbv, "psc", None), None)
            v = self._safe_float(getattr(nbv, "view_dist", None), None)
            if p is not None and p >= 0.0:
                psc_vals.append(p)
            if v is not None and v >= 0.0:
                vd_vals.append(v)

        def _robust(arr, fallback):
            if len(arr) == 0:
                return fallback
            lo = float(np.percentile(arr, 10.0))
            hi = float(np.percentile(arr, 90.0))
            if hi - lo < 1e-9:
                hi = lo + 1e-9
            return lo, hi

        p_lo, p_hi = _robust(psc_vals, (0.0, 1.0))
        vd_lo, vd_hi = _robust(vd_vals, (0.0, 1.0))
        return p_lo, p_hi, vd_lo, vd_hi

    def cluster(self, nbvs, target_center):
        """
        Cluster NBVs in task space and return a reduced representative list.
        """
        if not nbvs:
            return []

        dir_feats = []
        for nbv in nbvs:
            _, view_dir = self._pose_to_feat(nbv.pose, target_center)
            dir_feats.append(view_dir)
        dir_feats = np.vstack(dir_feats)
        dir_feats /= np.linalg.norm(dir_feats, axis=1, keepdims=True)

        num_nbvs = len(nbvs)
        dist_mat = np.zeros((num_nbvs, num_nbvs), dtype=np.float64)

        positions = np.zeros((num_nbvs, 3), dtype=np.float64)
        for i, nbv in enumerate(nbvs):
            pos, _ = self._pose_to_feat(nbv.pose, target_center)
            positions[i] = pos

        for i in range(num_nbvs):
            cos_vals = np.clip(np.dot(dir_feats[i], dir_feats.T), -1.0, 1.0)
            d_dir = np.arccos(cos_vals) / max(self.eps_dir, 1e-9)
            d_pos = np.linalg.norm(positions[i] - positions, axis=1) / max(self.eps_pos, 1e-9)
            dist_mat[i] = np.sqrt(d_dir * d_dir + d_pos * d_pos)

        clustering = DBSCAN(
            eps=1.0,
            min_samples=self.min_samples,
            metric="precomputed"
        ).fit(dist_mat)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(nbvs[i])

        stats = self._build_norm_stats(nbvs)
        reps = []

        for label, members in clusters.items():
            if label == -1:
                members_sorted = sorted(
                    members,
                    key=lambda n: self._nbv_meta_score(n, stats),
                    reverse=True,
                )
                reps.extend(members_sorted[:max(self.max_rep, self.noise_rep)])
                continue

            members_sorted = sorted(
                members,
                key=lambda n: (self._nbv_meta_score(n, stats), -int(getattr(n, "index", 0))),
                reverse=True,
            )

            if len(members_sorted) > 0 and self._nbv_meta_score(members_sorted[0], stats) <= -1e8:
                members_sorted = members

            chosen = []
            for cand in members_sorted:
                if len(chosen) >= self.max_rep:
                    break
                _, vc = self._pose_to_feat(cand.pose, target_center)

                keep = True
                for prev in chosen:
                    _, vp = self._pose_to_feat(prev.pose, target_center)
                    ang = np.arccos(np.clip(np.dot(vc, vp), -1.0, 1.0))
                    if ang < 0.08:
                        keep = False
                        break

                if keep:
                    chosen.append(cand)

            if len(chosen) < self.max_rep:
                for cand in members_sorted:
                    if cand in chosen:
                        continue
                    chosen.append(cand)
                    if len(chosen) >= self.max_rep:
                        break

            reps.extend(chosen)

        rospy.loginfo(
            "[NBVSelector] Task-space reduced %d NBVs -> %d representatives (%d clusters)",
            len(nbvs), len(reps), len(clusters)
        )
        return reps


# ============================================================================
# Joint-space clusterer
# ============================================================================
class JointSpaceClusterer(object):
    """
    Joint-space clusterer with IK seed reuse and caching.

    This stage converts task-space NBV poses into feasible joint-space solutions,
    then clusters those solutions into Action-Modes.
    """

    def __init__(
        self,
        move_group="fr3_arm",
        base_frame="base_link",
        wrist_link="wrist3_Link",
        camera_link="in_hand_camera_link",
        eps_joint=0.10,
        min_samples=1,
        use_action_seed=True,
        seed_reset_every_n=80,
        ik_timeout=0.15,
        cache_enabled=True,
        cache_pos_res=0.003,
        cache_ang_res_deg=2.0,
        neg_cache_ttl_sec=3.0,
        max_cache_size=6000,
        max_neg_cache_size=6000,
        log_every_sec=1.0,
    ):
        rospy.loginfo("[NBVSelector] Initializing JointSpaceClusterer (seed+cache)")

        moveit_commander.roscpp_initialize([])
        self.arm = moveit_commander.MoveGroupCommander(move_group)
        self.robot = RobotCommander()

        self.base_frame = base_frame
        self.wrist_link = wrist_link
        self.camera_link = camera_link

        self.eps_joint = float(eps_joint)
        self.min_samples = int(min_samples)

        self.use_action_seed = bool(use_action_seed)
        self.seed_reset_every_n = int(max(1, seed_reset_every_n))
        self.ik_timeout = float(max(0.01, ik_timeout))

        self.cache_enabled = bool(cache_enabled)
        self.cache_pos_res = float(max(1e-4, cache_pos_res))
        self.cache_ang_res = float(max(1e-3, np.deg2rad(cache_ang_res_deg)))
        self.neg_cache_ttl_sec = float(max(0.1, neg_cache_ttl_sec))

        self.max_cache_size = int(max(100, max_cache_size))
        self.max_neg_cache_size = int(max(100, max_neg_cache_size))
        self.log_every_sec = float(max(0.1, log_every_sec))

        self._last_js = None
        self._last_good_q = None
        self._solve_calls = 0

        self._ik_cache = {}
        self._neg_cache = {}

        rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)

        rospy.loginfo("[NBVSelector] Waiting for /compute_ik service...")
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("[NBVSelector] IK service connected")

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        self._T_wrist_cam = None
        self._T_wrist_cam_stamp = rospy.Time(0)

        # The wrist link is the IK end-effector link.
        self.ee_link = wrist_link

        # Manual joint limits must match MoveIt active joint names.
        manual_pos_limits = {
            "j1": (-3.0543261, 3.0543261),
            "j2": (-4.6251, 1.4835),
            "j3": (-2.8274, 2.8274),
            "j4": (-4.6251, 1.4835),
            "j5": (-3.0543, 3.0543),
            "j6": (-3.0543, 3.0543),
        }

        self._joint_limits = {}
        active = list(self.arm.get_active_joints())
        missing = []

        for joint_name in active:
            if joint_name in manual_pos_limits:
                self._joint_limits[joint_name] = tuple(manual_pos_limits[joint_name])
            else:
                missing.append(joint_name)

        rospy.loginfo("[NBVSelector] Active joints: %s", ", ".join(active))
        rospy.loginfo(
            "[NBVSelector] Manual limits loaded for: %s",
            ", ".join(sorted(self._joint_limits.keys()))
        )
        if missing:
            rospy.logwarn(
                "[NBVSelector] Manual limits missing joints: %s",
                ", ".join(missing)
            )
            rospy.logwarn(
                "[NBVSelector] If MoveIt joint names are not j1~j6, rename manual_pos_limits keys accordingly."
            )

        self.sanity_check()

    def _js_cb(self, msg):
        """Cache the latest joint state."""
        self._last_js = msg

    @staticmethod
    def pose_to_T(pose):
        """Convert Pose to homogeneous transform."""
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans = tft.quaternion_matrix(quat)
        trans[0, 3] = pose.position.x
        trans[1, 3] = pose.position.y
        trans[2, 3] = pose.position.z
        return trans

    @staticmethod
    def T_to_pose(trans):
        """Convert homogeneous transform to Pose."""
        pose = Pose()
        pose.position.x = float(trans[0, 3])
        pose.position.y = float(trans[1, 3])
        pose.position.z = float(trans[2, 3])
        quat = tft.quaternion_from_matrix(trans)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        return pose

    @staticmethod
    def _wrap_diff(a, b):
        """Wrapped angular difference."""
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _normalize_q_into_limits(self, q, joint_names, eps=1e-6):
        """
        Try to wrap a joint vector into the configured limits.
        """
        joint_limits = getattr(self, "_joint_limits", None)
        if not isinstance(joint_limits, dict) or len(joint_limits) == 0:
            return q.copy(), True

        q2 = q.copy()
        for i, name in enumerate(joint_names):
            if name not in joint_limits:
                continue

            lo, hi = joint_limits[name]
            val = float(q2[i])

            if (lo - eps) <= val <= (hi + eps):
                continue

            found = False
            for k in (-2, -1, 0, 1, 2):
                vv = val + k * 2.0 * np.pi
                if (lo - eps) <= vv <= (hi + eps):
                    q2[i] = vv
                    found = True
                    break

            if not found:
                return q2, False

        return q2, True

    def _joint_distance(self, q_a, q_b):
        """
        Compute normalized wrapped joint-space distance.
        """
        dq = self._wrap_diff(q_a, q_b)

        norm = np.ones_like(dq)
        for i, name in enumerate(self.arm.get_active_joints()):
            lim = self._joint_limits.get(name, None)
            if lim is None:
                continue
            lo, hi = lim
            span = hi - lo
            if span > 1e-6:
                norm[i] = span

        dq_norm = dq / norm
        return float(np.linalg.norm(dq_norm))

    def lookup_T(self, parent, child, timeout=0.2):
        """Lookup a TF transform as a homogeneous matrix."""
        tfm = self.tfbuf.lookup_transform(
            parent, child, rospy.Time(0), rospy.Duration(timeout)
        )
        t = tfm.transform.translation
        r = tfm.transform.rotation
        trans = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
        trans[0, 3] = t.x
        trans[1, 3] = t.y
        trans[2, 3] = t.z
        return trans

    def get_T_wrist_cam(self, force_refresh=False):
        """Get the wrist-to-camera extrinsic transform."""
        now = rospy.Time.now()
        if (self._T_wrist_cam is not None) and (not force_refresh):
            if (now - self._T_wrist_cam_stamp).to_sec() < 5.0:
                return self._T_wrist_cam

        try:
            T_wc = self.lookup_T(self.wrist_link, self.camera_link, timeout=0.5)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[NBVSelector] TF lookup failed for extrinsic %s->%s : %s",
                self.wrist_link,
                self.camera_link,
                str(exc),
            )
            return None

        self._T_wrist_cam = T_wc
        self._T_wrist_cam_stamp = now
        return self._T_wrist_cam

    def camera_pose_to_ee_pose(self, cam_pose):
        """Convert a camera pose into an end-effector pose."""
        T_ec = self.get_T_wrist_cam()
        if T_ec is None:
            return None
        T_bc = self.pose_to_T(cam_pose)
        T_be = np.matmul(T_bc, np.linalg.inv(T_ec))
        return self.T_to_pose(T_be)

    def _quat_to_rpy(self, quat_xyzw):
        """Quaternion to roll-pitch-yaw."""
        roll, pitch, yaw = tft.euler_from_quaternion(quat_xyzw)
        return float(roll), float(pitch), float(yaw)

    def _wrist_pose_key(self, ee_pose):
        """
        Build a discretized cache key from wrist pose.
        """
        px, py, pz = ee_pose.position.x, ee_pose.position.y, ee_pose.position.z
        qx, qy, qz, qw = (
            ee_pose.orientation.x,
            ee_pose.orientation.y,
            ee_pose.orientation.z,
            ee_pose.orientation.w,
        )
        roll, pitch, yaw = self._quat_to_rpy([qx, qy, qz, qw])

        ipx = int(np.round(px / self.cache_pos_res))
        ipy = int(np.round(py / self.cache_pos_res))
        ipz = int(np.round(pz / self.cache_pos_res))

        ir = int(np.round(roll / self.cache_ang_res))
        ip = int(np.round(pitch / self.cache_ang_res))
        iy = int(np.round(yaw / self.cache_ang_res))

        return ipx, ipy, ipz, ir, ip, iy

    def _neg_cache_hit(self, key, now_sec):
        """Check whether a negative cache entry is still valid."""
        ts = self._neg_cache.get(key, None)
        if ts is None:
            return False
        return (now_sec - ts) < self.neg_cache_ttl_sec

    def _neg_cache_put(self, key, now_sec):
        """Store a negative cache entry."""
        if len(self._neg_cache) >= self.max_neg_cache_size:
            self._neg_cache.clear()
        self._neg_cache[key] = now_sec

    def _ik_cache_put(self, key, q):
        """Store a successful IK solution in cache."""
        if len(self._ik_cache) >= self.max_cache_size:
            self._ik_cache.clear()
        self._ik_cache[key] = q.copy()

    def _make_robot_state_seed(self):
        """
        Build an IK seed robot state.

        If action-seed reuse is enabled, the last successful solution is used
        as seed except for periodic resets.
        """
        rs = self.robot.get_current_state()

        if (not self.use_action_seed) or (self._last_good_q is None):
            return rs
        if (self._solve_calls % self.seed_reset_every_n) == 0:
            return rs

        joint_names = self.arm.get_active_joints()
        pos_list = list(rs.joint_state.position)
        name_to_idx = {name: i for i, name in enumerate(rs.joint_state.name)}

        for j, name in enumerate(joint_names):
            idx = name_to_idx.get(name, None)
            if idx is not None and j < len(self._last_good_q):
                pos_list[idx] = float(self._last_good_q[j])

        rs.joint_state.position = tuple(pos_list)
        return rs

    def solve_ik_ee(self, ee_pose):
        """
        Solve IK for an end-effector pose.
        """
        self._solve_calls += 1

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
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(
                1.0,
                "[NBVSelector] IK service call failed: %s",
                str(exc),
            )
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
            q = np.array([name_to_pos[name] for name in joint_names], dtype=np.float64)
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

    def solve_ik_from_camera_pose(self, cam_pose):
        """Solve IK from a camera pose by converting it to EE pose first."""
        ee_pose = self.camera_pose_to_ee_pose(cam_pose)
        if ee_pose is None:
            return None
        return self.solve_ik_ee(ee_pose)

    def compute_jacobian_condition(self, q):
        """Compute Jacobian condition number as a singularity indicator."""
        try:
            q_list = [float(x) for x in q]
            jac = self.arm.get_jacobian_matrix(q_list)
            jac = np.asarray(jac, dtype=np.float64)
            if jac.size == 0:
                return float("inf")

            _, s_vals, _ = np.linalg.svd(jac, full_matrices=False)
            if s_vals.size == 0 or np.min(s_vals) < 1e-6:
                return float("inf")
            return float(np.max(s_vals) / np.min(s_vals))
        except Exception:
            return float("inf")

    def cluster_nbvs(self, nbv_list):
        """
        Solve IK for all candidate NBVs and cluster valid solutions in joint space.
        """
        rospy.loginfo("[NBVSelector] Clustering %d NBVs (seed+cache)", len(nbv_list))

        q_list = []
        valid_nbvs = []

        total = len(nbv_list)
        ok = 0
        fail = 0

        solve_calls_before = self._solve_calls

        for nbv in nbv_list:
            q = self.solve_ik_from_camera_pose(nbv.pose)
            if q is None:
                fail += 1
                rospy.logwarn_throttle(
                    self.log_every_sec,
                    "[NBVSelector] IK failed for NBV %d",
                    nbv.index,
                )
                continue
            ok += 1
            q_list.append(q)
            valid_nbvs.append(nbv)

        solve_calls_after = self._solve_calls
        if self.cache_enabled:
            service_calls = max(0, solve_calls_after - solve_calls_before)
            approx_cache_hit = max(0, total - service_calls)
        else:
            approx_cache_hit = 0

        rospy.loginfo(
            "[NBVSelector] IK stats: total=%d, ok=%d, fail=%d, approx_cache_hit=%d, cache_size=%d, neg_cache_size=%d",
            total,
            ok,
            fail,
            approx_cache_hit,
            len(self._ik_cache),
            len(self._neg_cache),
        )

        if not q_list:
            rospy.logerr("[NBVSelector] No valid IK solutions")
            return []

        q_stack = np.vstack(q_list)

        num_q = q_stack.shape[0]
        dist_mat = np.zeros((num_q, num_q), dtype=np.float64)
        for i in range(num_q):
            for j in range(i + 1, num_q):
                d = self._joint_distance(q_stack[i], q_stack[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        clustering = DBSCAN(
            eps=self.eps_joint,
            min_samples=self.min_samples,
            metric="precomputed",
        ).fit(dist_mat)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = NBVCluster(label)
            clusters[label].nbvs.append(valid_nbvs[i])
            clusters[label].q_members.append(q_stack[i])

        cluster_list = []
        for cid, cluster in clusters.items():
            qs = np.vstack(cluster.q_members)
            cluster.q_center = np.mean(qs, axis=0)

            # Prefer max PSC if metadata exists, otherwise choose the member
            # closest to the cluster center.
            psc_list = []
            for nbv_i in cluster.nbvs:
                score = nbv_i.psc
                psc_list.append(-1.0 if score is None else float(score))
            psc_arr = np.asarray(psc_list, dtype=np.float32)

            if np.any(psc_arr >= 0.0):
                rep_idx = int(np.argmax(psc_arr))
            else:
                dists = np.linalg.norm(qs - cluster.q_center[None, :], axis=1)
                rep_idx = int(np.argmin(dists))

            cluster.rep_nbv = cluster.nbvs[rep_idx]
            cluster.rep_q = qs[rep_idx].copy()
            cluster_list.append(cluster)

        for cluster in cluster_list:
            cluster.is_singleton = (len(cluster.nbvs) == 1)

        rospy.loginfo(
            "[NBVSelector] Reduced %d NBVs to %d Action-Modes",
            len(nbv_list), len(cluster_list)
        )
        return cluster_list

    def sanity_check(self):
        """Run basic TF and IK sanity checks at startup."""
        rospy.sleep(0.2)

        T_wc = self.get_T_wrist_cam(force_refresh=True)
        if T_wc is None:
            rospy.logwarn(
                "[NBVSelector] SanityCheck: TF extrinsic not ready (wrist->camera)."
            )
        else:
            rospy.loginfo(
                "[NBVSelector] SanityCheck: TF extrinsic wrist->camera OK."
            )

        try:
            current_ee_pose = self.arm.get_current_pose(self.ee_link).pose
            q = self.solve_ik_ee(current_ee_pose)
            if q is None:
                rospy.logwarn(
                    "[NBVSelector] SanityCheck: IK on current EE pose FAILED."
                )
            else:
                rospy.loginfo(
                    "[NBVSelector] SanityCheck: IK on current EE pose OK."
                )
        except Exception as exc:
            rospy.logwarn(
                "[NBVSelector] SanityCheck: cannot get current EE pose: %s",
                str(exc),
            )


# ============================================================================
# Visualization helpers
# ============================================================================
def _color_from_id(i, alpha=0.9):
    """Return a stable pseudo-color from an integer ID."""
    palette = [
        (0.90, 0.10, 0.10),
        (0.10, 0.90, 0.10),
        (0.10, 0.40, 0.95),
        (0.95, 0.80, 0.10),
        (0.75, 0.20, 0.85),
        (0.10, 0.85, 0.85),
        (0.95, 0.45, 0.10),
        (0.60, 0.60, 0.60),
    ]
    r, g, b = palette[int(i) % len(palette)]
    return ColorRGBA(r=r, g=g, b=b, a=float(alpha))


def _pose_rpy_deg(pose):
    """Pose orientation to roll-pitch-yaw in degrees."""
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    roll, pitch, yaw = tft.euler_from_quaternion(quat)
    return np.rad2deg([roll, pitch, yaw])


def _vec_unit(v):
    """Normalize a vector with zero-norm protection."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


# ============================================================================
# ROS node wrapper
# ============================================================================
class NBVSelectorNode(object):
    """ROS wrapper for Layer-2 Action-Mode selection."""

    def __init__(self):
        rospy.init_node("nbv_selector")

        # --------------------------------------------------------------------
        # Core parameters
        # --------------------------------------------------------------------
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.nbv_topic = rospy.get_param("~nbv_topic", "/nbv/valid_view_poses")

        # --------------------------------------------------------------------
        # Visualization parameters
        # --------------------------------------------------------------------
        self.viz_enable = rospy.get_param("~viz_enable", True)
        self.viz_show_members = rospy.get_param("~viz_show_members", True)
        self.viz_line_to_target = rospy.get_param("~viz_line_to_target", True)
        self.viz_arrow_len = float(rospy.get_param("~viz_arrow_len", 0.12))
        self.viz_member_scale = float(rospy.get_param("~viz_member_scale", 0.025))
        self.viz_target_scale = float(rospy.get_param("~viz_target_scale", 0.04))
        self.viz_text_scale = float(rospy.get_param("~viz_text_scale", 0.06))

        # --------------------------------------------------------------------
        # Weighted joint motion cost
        # --------------------------------------------------------------------
        self.joint_weights = rospy.get_param(
            "~joint_weights", [1.5, 2.0, 2.0, 0.8, 0.6, 0.5]
        )
        self.joint_weights = np.asarray(self.joint_weights, dtype=np.float64)

        # --------------------------------------------------------------------
        # Printing parameters
        # --------------------------------------------------------------------
        self.print_enable = rospy.get_param("~print_enable", True)
        self.print_per_joint = rospy.get_param("~print_per_joint", True)
        self.print_deg = rospy.get_param("~print_deg", True)
        self.print_max_modes = int(rospy.get_param("~print_max_modes", 20))

        # --------------------------------------------------------------------
        # CSV diagnostics
        # --------------------------------------------------------------------
        self.save_csv = rospy.get_param("~save_csv", True)
        self.out_dir = rospy.get_param("~out_dir", "/tmp")

        ts = rospy.Time.now().to_sec()
        self.csv_path = "%s/nbv_action_modes_%d.csv" % (self.out_dir, int(ts))

        self._csv_file = None
        self._csv_writer = None
        self._nbv_set_counter = 0

        if self.save_csv:
            import csv

            self._csv_file = open(self.csv_path, "w")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "stamp",
                "nbv_set_id",
                "mode_id",
                "cluster_id",
                "cluster_size",
                "is_singleton",
                "dq_max",
                "dq_mean",
                "dq_std",
                "rep_nbv_idx",
                "psc",
                "view_dist",
                "dq_exec",
                "m_limit",
                "m_sing",
                "m_exec",
                "psc_norm",
                "dq_mean_norm",
                "dq_exec_norm",
                "m_exec_norm",
                "score",
            ])
            rospy.loginfo("[NBVSelector] Saving diagnostics to %s", self.csv_path)

        rospy.on_shutdown(self._on_shutdown)

        # --------------------------------------------------------------------
        # Submodules
        # --------------------------------------------------------------------
        self.task_clusterer = TaskSpaceClusterer(
            eps_pos=rospy.get_param("~task_eps_pos", 0.02),
            eps_dir=rospy.get_param("~task_eps_dir", 0.20),
            min_samples=rospy.get_param("~task_min_samples", 1),
            max_rep_per_cluster=rospy.get_param("~task_max_rep", 3),
            noise_rep=rospy.get_param("~task_noise_rep", 6),
            w_psc=rospy.get_param("~task_w_psc", 0.1),
            w_vdist=rospy.get_param("~task_w_vdist", 0.2),
            vdist_pref=rospy.get_param("~task_vdist_pref", 0.30),
            vdist_sigma=rospy.get_param("~task_vdist_sigma", 0.10),
        )

        self.clusterer = JointSpaceClusterer(
            move_group=rospy.get_param("~move_group", "fr3_arm"),
            base_frame=self.base_frame,
            wrist_link=rospy.get_param("~wrist_link", "hand_base_visual_link"),
            camera_link=rospy.get_param("~camera_link", "in_hand_camera_link"),
            eps_joint=rospy.get_param("~eps_joint", 0.1),
            min_samples=rospy.get_param("~min_samples", 2),
            use_action_seed=rospy.get_param("~use_action_seed", True),
            seed_reset_every_n=rospy.get_param("~seed_reset_every_n", 80),
            ik_timeout=rospy.get_param("~ik_timeout", 0.15),
            cache_enabled=rospy.get_param("~cache_enabled", True),
            cache_pos_res=rospy.get_param("~cache_pos_res", 0.003),
            cache_ang_res_deg=rospy.get_param("~cache_ang_res_deg", 2.0),
            neg_cache_ttl_sec=rospy.get_param("~neg_cache_ttl_sec", 3.0),
            max_cache_size=rospy.get_param("~max_cache_size", 6000),
            max_neg_cache_size=rospy.get_param("~max_neg_cache_size", 6000),
        )

        # --------------------------------------------------------------------
        # Publishers
        # --------------------------------------------------------------------
        self.marker_pub = rospy.Publisher(
            "/nbv/action_modes_markers", MarkerArray, queue_size=1
        )
        self.reps_pub = rospy.Publisher(
            "/nbv/action_modes_reps", PoseArray, queue_size=1, latch=True
        )
        self.mode0_members_pub = rospy.Publisher(
            "/nbv/action_mode0_members", PoseArray, queue_size=1, latch=True
        )
        self.mode0_members_meta_pub = rospy.Publisher(
            "/nbv/action_mode0_member_meta",
            Float32MultiArray,
            queue_size=1,
            latch=True,
        )

        # Best-mode summary outputs
        self.best_mode_score_pub = rospy.Publisher(
            "/nbv/best_mode_score", Float32, queue_size=1, latch=True
        )
        self.best_mode_dq_exec_pub = rospy.Publisher(
            "/nbv/best_mode_dq_exec", Float32, queue_size=1, latch=True
        )
        self.best_mode_m_limit_pub = rospy.Publisher(
            "/nbv/best_mode_m_limit", Float32, queue_size=1, latch=True
        )
        self.best_mode_m_sing_pub = rospy.Publisher(
            "/nbv/best_mode_m_sing", Float32, queue_size=1, latch=True
        )
        self.best_mode_m_exec_pub = rospy.Publisher(
            "/nbv/best_mode_m_exec", Float32, queue_size=1, latch=True
        )

        # --------------------------------------------------------------------
        # State
        # --------------------------------------------------------------------
        self.target_center = None
        rospy.Subscriber("/apple/center", PointStamped, self._target_cb, queue_size=1)
        rospy.loginfo("[NBVSelector] Waiting for target_center...")

        self._last_meta = None
        self._last_meta_time = rospy.Time(0)
        rospy.Subscriber("/nbv/valid_view_meta", Float32MultiArray, self._meta_cb, queue_size=1)

        # --------------------------------------------------------------------
        # Scheduling and buffering
        # --------------------------------------------------------------------
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

        self._meta_max_age = float(rospy.get_param("~meta_max_age", 1.0))

        tick_hz = float(rospy.get_param("~tick_hz", 10.0))
        self._timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, tick_hz)),
            self._tick,
        )

    def _on_shutdown(self):
        """Flush and close CSV file on shutdown."""
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass

    def _target_cb(self, msg):
        """Cache target center."""
        self.target_center = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=np.float64,
        )

    def _meta_cb(self, msg):
        """Cache Layer-1 metadata array."""
        self._last_meta = msg.data[:]
        self._last_meta_time = rospy.Time.now()

    def nbv_cb(self, msg):
        """Cache latest NBV pose array."""
        if self.target_center is None:
            return
        with self._lock:
            self._latest_msg = msg

    # ========================================================================
    # Helper: current joint state
    # ========================================================================
    def _get_current_q(self, joint_names):
        """Extract current active-joint vector from cached JointState."""
        js = self.clusterer._last_js
        if js is None:
            return None
        name_to_pos = dict(zip(js.name, js.position))
        try:
            return np.array([name_to_pos[name] for name in joint_names], dtype=np.float64)
        except KeyError:
            return None

    # ========================================================================
    # Helper: normalization and ranking
    # ========================================================================
    def _robust_minmax(self, arr, lo_q=10.0, hi_q=90.0, fallback=(0.0, 1.0)):
        """Robust percentile-based min/max."""
        arr = [float(x) for x in arr if x is not None and np.isfinite(x)]
        if len(arr) == 0:
            return float(fallback[0]), float(fallback[1])

        lo = float(np.percentile(arr, lo_q))
        hi = float(np.percentile(arr, hi_q))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        return lo, hi

    def _norm01(self, x, lo, hi):
        """Normalize scalar to [0, 1]."""
        if x is None:
            return 0.0
        try:
            xv = float(x)
        except Exception:
            return 0.0
        if not np.isfinite(xv):
            return 0.0

        val = (xv - float(lo)) / (float(hi) - float(lo) + 1e-12)
        return float(np.clip(val, 0.0, 1.0))

    def _compute_m_limit(self, q, joint_names):
        """
        Compute conservative joint-limit margin in [0, 1].

        The minimum normalized distance to any configured joint bound is used.
        """
        joint_limits = getattr(self.clusterer, "_joint_limits", None)
        if not isinstance(joint_limits, dict) or len(joint_limits) == 0:
            rospy.logwarn_throttle(1.0, "[NBVSelector] _joint_limits not ready; m_limit=0")
            return 0.0

        dist_safe = float(rospy.get_param("~limit_safe_rad", 0.60))
        worst_margin = 1e9
        used_any = False

        for name, val in zip(joint_names, q):
            if name not in joint_limits:
                continue
            used_any = True
            lo, hi = joint_limits[name]

            dist_lo = float(val - lo)
            dist_hi = float(hi - val)

            if dist_lo < -1e-6 or dist_hi < -1e-6:
                return 0.0

            dist = dist_lo if dist_lo < dist_hi else dist_hi
            margin = float(np.clip(dist / max(dist_safe, 1e-6), 0.0, 1.0))
            if margin < worst_margin:
                worst_margin = margin

        if not used_any:
            return 0.0
        return float(worst_margin)

    def _compute_m_sing(self, q):
        """
        Compute singularity margin in [0, 1] from Jacobian condition number.
        """
        cond = self.clusterer.compute_jacobian_condition(q)
        if not np.isfinite(cond) or cond <= 0:
            return 0.0

        cond_good = float(rospy.get_param("~cond_good", 80.0))
        cond_bad = float(rospy.get_param("~cond_bad", 400.0))

        if cond <= cond_good:
            return 1.0
        if cond >= cond_bad:
            return 0.0

        return float(1.0 - (cond - cond_good) / (cond_bad - cond_good))

    def _weighted_dq(self, q_a, q_b):
        """Weighted wrapped joint-space distance."""
        qa = np.asarray(q_a, dtype=np.float64)
        qb = np.asarray(q_b, dtype=np.float64)

        if qa.shape != qb.shape:
            return 1e18

        dq = self.clusterer._wrap_diff(qa, qb)
        weights = self.joint_weights

        if weights.size != dq.size:
            val = float(np.linalg.norm(dq))
            return val if np.isfinite(val) else 1e18

        val = float(np.sqrt(np.sum((weights * dq) ** 2)))
        return val if np.isfinite(val) else 1e18

    def _hash_pose_array(self, pose_array):
        """Build a compact hash for repeated pose-array detection."""
        ang_res = np.deg2rad(self._hash_ang_res_deg)
        buf = []

        for pose in pose_array.poses:
            px = int(np.round(pose.position.x / self._hash_pos_res))
            py = int(np.round(pose.position.y / self._hash_pos_res))
            pz = int(np.round(pose.position.z / self._hash_pos_res))

            quat = [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
            roll, pitch, yaw = tft.euler_from_quaternion(quat)
            ir = int(np.round(roll / ang_res))
            ip = int(np.round(pitch / ang_res))
            iy = int(np.round(yaw / ang_res))

            buf.append((px, py, pz, ir, ip, iy))

        md5 = hashlib.md5()
        md5.update(np.asarray(buf, dtype=np.int32).tobytes())
        return md5.hexdigest()

    # ========================================================================
    # Helper: publishing
    # ========================================================================
    def _publish_mode0_members(self, best_cluster, frame_id):
        """
        Publish all members of the highest-ranked Action-Mode, together with
        aligned PSC metadata.
        """
        if best_cluster is None:
            return

        pose_array = PoseArray()
        pose_array.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        for nbv in best_cluster.nbvs:
            pose_array.poses.append(nbv.pose)
        self.mode0_members_pub.publish(pose_array)

        meta = Float32MultiArray()
        meta.data = []
        for nbv in best_cluster.nbvs:
            p = getattr(nbv, "psc", None)
            p = float(p) if (p is not None and np.isfinite(p) and p >= 0.0) else float("nan")
            meta.data.append(p)
        self.mode0_members_meta_pub.publish(meta)

    def _publish_best_mode_summaries(self, best_cluster):
        """
        Publish best-mode summary scalars.

        If no valid best mode exists, publish NaN.
        """
        def _f(x):
            try:
                val = float(x)
            except Exception:
                val = float("nan")
            if not np.isfinite(val):
                val = float("nan")
            return val

        if best_cluster is None:
            self.best_mode_score_pub.publish(Float32(data=float("nan")))
            self.best_mode_dq_exec_pub.publish(Float32(data=float("nan")))
            self.best_mode_m_limit_pub.publish(Float32(data=float("nan")))
            self.best_mode_m_sing_pub.publish(Float32(data=float("nan")))
            self.best_mode_m_exec_pub.publish(Float32(data=float("nan")))
            return

        self.best_mode_score_pub.publish(Float32(data=_f(best_cluster.score)))
        self.best_mode_dq_exec_pub.publish(Float32(data=_f(best_cluster.dq_exec)))
        self.best_mode_m_limit_pub.publish(Float32(data=_f(best_cluster.m_limit)))
        self.best_mode_m_sing_pub.publish(Float32(data=_f(best_cluster.m_sing)))
        self.best_mode_m_exec_pub.publish(Float32(data=_f(best_cluster.m_exec)))

    # ========================================================================
    # Visualization
    # ========================================================================
    def _make_marker(self, mid, mtype, ns, frame_id):
        """Create a standard RViz marker with common fields initialized."""
        mk = Marker()
        mk.header.frame_id = frame_id
        mk.header.stamp = rospy.Time.now()
        mk.ns = ns
        mk.id = int(mid)
        mk.type = int(mtype)
        mk.action = Marker.ADD
        mk.lifetime = rospy.Duration(5.0)
        mk.pose.orientation.w = 1.0
        return mk

    def _publish_markers_and_reps(self, clusters):
        """
        Publish Action-Mode RViz markers and representative pose array.
        """
        if not self.viz_enable:
            reps_pose_array = PoseArray()
            reps_pose_array.header = Header(stamp=rospy.Time.now(), frame_id=self.base_frame)
            for cluster in clusters:
                exec_pose = cluster.best_nbv.pose if cluster.best_nbv is not None else cluster.rep_nbv.pose
                reps_pose_array.poses.append(exec_pose)
            self.reps_pub.publish(reps_pose_array)
            return

        marker_array = MarkerArray()
        frame = self.base_frame
        now = rospy.Time.now()
        mid = 0

        mk_clear = Marker()
        mk_clear.header.frame_id = frame
        mk_clear.header.stamp = now
        mk_clear.ns = "clear"
        mk_clear.id = 0
        mk_clear.action = Marker.DELETEALL
        marker_array.markers.append(mk_clear)
        mid += 1

        if self.target_center is not None:
            mk_t = self._make_marker(mid, Marker.SPHERE, "target", frame)
            mk_t.pose.position.x = float(self.target_center[0])
            mk_t.pose.position.y = float(self.target_center[1])
            mk_t.pose.position.z = float(self.target_center[2])
            mk_t.scale.x = self.viz_target_scale
            mk_t.scale.y = self.viz_target_scale
            mk_t.scale.z = self.viz_target_scale
            mk_t.color = ColorRGBA(1.0, 1.0, 1.0, 0.9)
            marker_array.markers.append(mk_t)
            mid += 1

        reps_pose_array = PoseArray()
        reps_pose_array.header = Header(stamp=rospy.Time.now(), frame_id=frame)

        for idx, cluster in enumerate(clusters):
            col = _color_from_id(idx, alpha=0.6)
            ns_base = "mode_%d" % int(cluster.mode_id)

            exec_pose = cluster.best_nbv.pose if cluster.best_nbv is not None else cluster.rep_nbv.pose
            rep_pose = exec_pose
            reps_pose_array.poses.append(rep_pose)

            rep_pos = np.array(
                [rep_pose.position.x, rep_pose.position.y, rep_pose.position.z],
                dtype=np.float64,
            )

            if self.target_center is not None:
                vdir = _vec_unit(self.target_center - rep_pos)
            else:
                vdir = np.array([0.0, 0.0, 1.0])

            from geometry_msgs.msg import Point

            mk_a = self._make_marker(mid, Marker.ARROW, ns_base + "/rep_arrow", frame)
            mk_a.color = col
            mk_a.scale.x = 0.01
            mk_a.scale.y = 0.02
            mk_a.scale.z = 0.02

            p0 = rep_pos
            p1 = rep_pos + vdir * self.viz_arrow_len
            mk_a.points = [
                Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
            ]
            marker_array.markers.append(mk_a)
            mid += 1

            mk_s = self._make_marker(mid, Marker.SPHERE, ns_base + "/rep_pos", frame)
            mk_s.pose.position.x = float(rep_pos[0])
            mk_s.pose.position.y = float(rep_pos[1])
            mk_s.pose.position.z = float(rep_pos[2])
            mk_s.scale.x = self.viz_member_scale * 1.3
            mk_s.scale.y = self.viz_member_scale * 1.3
            mk_s.scale.z = self.viz_member_scale * 1.3
            mk_s.color = ColorRGBA(col.r, col.g, col.b, 1.0)
            marker_array.markers.append(mk_s)
            mid += 1

            mk_txt = self._make_marker(mid, Marker.TEXT_VIEW_FACING, ns_base + "/text", frame)
            mk_txt.pose.position.x = float(rep_pos[0])
            mk_txt.pose.position.y = float(rep_pos[1])
            mk_txt.pose.position.z = float(rep_pos[2] + 0.06)
            mk_txt.scale.z = self.viz_text_scale
            mk_txt.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
            mk_txt.text = "Mode %d | n=%d | score=%.2f" % (
                int(cluster.mode_id),
                int(len(cluster.nbvs)),
                float(cluster.score),
            )
            marker_array.markers.append(mk_txt)
            mid += 1

        self.marker_pub.publish(marker_array)
        self.reps_pub.publish(reps_pose_array)

    # ========================================================================
    # Printing
    # ========================================================================
    def _print_action_modes(self, clusters, joint_names):
        """Print Action-Mode summary table to terminal."""
        if not self.print_enable:
            return
        if not clusters:
            rospy.logwarn("[NBVSelector] No Action-Modes to print.")
            return

        rospy.loginfo("========== Action-Modes ==========")
        for cluster in clusters[:min(len(clusters), self.print_max_modes)]:
            rospy.loginfo(
                "Mode %d | n=%d | dq_mean=%.4f | dq_exec=%.3f | "
                "m_limit=%.3f m_sing=%.3f m_exec=%.3f | score=%.3f",
                int(cluster.mode_id),
                int(len(cluster.nbvs)),
                float(cluster.dq_mean),
                float(cluster.dq_exec),
                float(cluster.m_limit),
                float(cluster.m_sing),
                float(cluster.m_exec),
                float(cluster.score),
            )
        rospy.loginfo("==================================")

    # ========================================================================
    # Timer tick and main processing
    # ========================================================================
    def _tick(self, _evt):
        """Timer-driven scheduler for NBV processing."""
        if self.target_center is None:
            return

        now = rospy.Time.now().to_sec()
        if (now - self._last_proc_stamp) < self._min_period:
            return

        with self._lock:
            msg = self._latest_msg
            if msg is None:
                return
            if self._busy and self._drop_if_busy:
                return

            if self._hash_enable:
                try:
                    hval = self._hash_pose_array(msg)
                    if self._last_hash is not None and hval == self._last_hash:
                        return
                except Exception:
                    hval = None
            else:
                hval = None

            meta = None
            if self._last_meta is not None:
                expected = int(rospy.get_param("~meta_stride", 3)) * len(msg.poses)
                if len(self._last_meta) >= expected:
                    meta = self._last_meta[:]
                    age = (rospy.Time.now() - self._last_meta_time).to_sec()
                    if age > self._meta_max_age:
                        rospy.logwarn_throttle(
                            1.0,
                            "[NBVSelector] meta is old: age=%.3fs (max_age=%.3f), but still using it",
                            age,
                            self._meta_max_age,
                        )

            self._busy = True
            self._last_proc_stamp = now
            if hval is not None:
                self._last_hash = hval

        try:
            self._process(msg, meta)
        finally:
            with self._lock:
                self._busy = False

    def _process(self, msg, meta_snapshot):
        """
        Full Action-Mode processing pipeline.
        """
        meta_stride = int(rospy.get_param("~meta_stride", 3))
        idx_psc = int(rospy.get_param("~meta_idx_psc", 0))
        idx_view_dist = int(rospy.get_param("~meta_idx_view_dist", 1))
        idx_used_dynamic = int(rospy.get_param("~meta_idx_used_dynamic", 2))

        stamp = rospy.Time.now().to_sec()
        nbv_set_id = self._nbv_set_counter
        self._nbv_set_counter += 1

        if msg is None or len(msg.poses) == 0:
            return

        nbvs = [NBV(i, pose) for i, pose in enumerate(msg.poses)]

        if meta_snapshot is not None and len(meta_snapshot) >= meta_stride * len(nbvs):
            for nbv in nbvs:
                i = int(nbv.index)
                base = meta_stride * i
                try:
                    nbv.psc = float(meta_snapshot[base + idx_psc])
                except Exception:
                    pass
                try:
                    nbv.view_dist = float(meta_snapshot[base + idx_view_dist])
                except Exception:
                    pass
                try:
                    nbv.used_dynamic = float(meta_snapshot[base + idx_used_dynamic])
                except Exception:
                    pass

        # --------------------------------------------------------------------
        # A) Task-space reduction
        # --------------------------------------------------------------------
        reps = self.task_clusterer.cluster(nbvs, target_center=self.target_center)
        if not reps:
            rospy.logwarn_throttle(1.0, "[NBVSelector] No reps after task-space reduction.")
            self._publish_best_mode_summaries(None)
            return

        # --------------------------------------------------------------------
        # B) Joint-space clustering
        # --------------------------------------------------------------------
        clusters = self.clusterer.cluster_nbvs(reps)
        if not clusters:
            rospy.logwarn_throttle(1.0, "[NBVSelector] No Action-Modes after joint-space clustering.")
            self._publish_best_mode_summaries(None)
            return

        joint_names = self.clusterer.arm.get_active_joints()

        # --------------------------------------------------------------------
        # C) Current joint vector
        # --------------------------------------------------------------------
        q_current = self._get_current_q(joint_names)
        if (q_current is None) or (not np.all(np.isfinite(q_current))):
            if (
                getattr(self.clusterer, "_last_good_q", None) is not None
                and np.all(np.isfinite(self.clusterer._last_good_q))
            ):
                q_current = self.clusterer._last_good_q.copy()
            else:
                q_current = np.zeros((len(joint_names),), dtype=np.float64)

            rospy.logwarn_throttle(
                1.0,
                "[NBVSelector] q_current invalid; using fallback."
            )

        # --------------------------------------------------------------------
        # D) Mode-internal execution representative selection
        # --------------------------------------------------------------------
        m_exec_thresh = float(rospy.get_param("~m_exec_thresh", 0.05))
        pen_w = float(rospy.get_param("~mexec_pen_w", 2.0))

        kept = []
        for cluster in clusters:
            qs = np.vstack(cluster.q_members)

            dq_to_center = np.array(
                [self.clusterer._joint_distance(qs[i], cluster.q_center) for i in range(qs.shape[0])],
                dtype=np.float64,
            )
            cluster.dq_max = float(np.max(dq_to_center))
            cluster.dq_mean = float(np.mean(dq_to_center))
            cluster.dq_std = float(np.std(dq_to_center))

            have_safe = False
            best_safe_idx, best_safe_cost, best_safe_psc = None, 1e18, -1e18
            best_any_idx, best_any_cost, best_any_psc = 0, 1e18, -1e18

            for i_m in range(qs.shape[0]):
                q_m = qs[i_m]
                dq_cost = self._weighted_dq(q_current, q_m)
                if not np.isfinite(dq_cost):
                    continue

                m_lim = self._compute_m_limit(q_m, joint_names)
                m_sing = self._compute_m_sing(q_m)
                m_exec = float(min(m_lim, m_sing))

                cost = float(dq_cost + pen_w * (1.0 - m_exec))

                psc_m = cluster.nbvs[i_m].psc
                psc_m = (
                    float(psc_m)
                    if (psc_m is not None and np.isfinite(psc_m) and psc_m >= 0.0)
                    else -1e18
                )

                if (cost < best_any_cost - 1e-9) or (
                    abs(cost - best_any_cost) <= 1e-9 and psc_m > best_any_psc
                ):
                    best_any_cost = cost
                    best_any_idx = int(i_m)
                    best_any_psc = psc_m

                if m_exec >= m_exec_thresh:
                    have_safe = True
                    if (cost < best_safe_cost - 1e-9) or (
                        abs(cost - best_safe_cost) <= 1e-9 and psc_m > best_safe_psc
                    ):
                        best_safe_cost = cost
                        best_safe_idx = int(i_m)
                        best_safe_psc = psc_m

            if have_safe and best_safe_idx is not None:
                best_idx = best_safe_idx
                best_cost = best_safe_cost
                cluster.force_unsafe = False
            else:
                best_idx = best_any_idx
                best_cost = best_any_cost
                cluster.force_unsafe = True

            cluster.q_best = qs[best_idx].copy()
            cluster.best_nbv = cluster.nbvs[best_idx]
            cluster.dq_exec = float(best_cost)

            cluster.m_limit = float(self._compute_m_limit(cluster.q_best, joint_names))
            cluster.m_sing = float(self._compute_m_sing(cluster.q_best))
            cluster.m_exec = float(min(cluster.m_limit, cluster.m_sing))

            kept.append(cluster)

        if not kept:
            self._publish_best_mode_summaries(None)
            return

        # --------------------------------------------------------------------
        # E) Normalize and rank modes
        # --------------------------------------------------------------------
        psc_list = []
        dq_mean_list = []
        dq_exec_list = []

        for cluster in kept:
            nbv_ref = cluster.best_nbv if cluster.best_nbv is not None else cluster.rep_nbv
            psc = getattr(nbv_ref, "psc", None)
            psc = (
                float(psc)
                if (psc is not None and np.isfinite(psc) and psc >= 0.0)
                else 0.0
            )
            psc_list.append(psc)
            dq_mean_list.append(float(cluster.dq_mean))
            dq_exec_list.append(float(cluster.dq_exec))

        psc_lo, psc_hi = self._robust_minmax(psc_list, fallback=(0.0, 1.0))
        dqmean_lo, dqmean_hi = self._robust_minmax(dq_mean_list, fallback=(0.0, 1.0))
        dqexec_lo, dqexec_hi = self._robust_minmax(dq_exec_list, fallback=(0.0, 1.0))

        w_psc = float(rospy.get_param("~score_w_psc", 1.0))
        w_compact = float(rospy.get_param("~score_w_compact", 0.6))
        w_exec = float(rospy.get_param("~score_w_exec", 0.8))
        w_mexec = float(rospy.get_param("~score_w_mexec", 1.0))

        for cluster in kept:
            nbv_ref = cluster.best_nbv if cluster.best_nbv is not None else cluster.rep_nbv
            psc = getattr(nbv_ref, "psc", None)
            psc = (
                float(psc)
                if (psc is not None and np.isfinite(psc) and psc >= 0.0)
                else 0.0
            )

            cluster.psc_norm = self._norm01(psc, psc_lo, psc_hi)
            cluster.dq_mean_norm = self._norm01(cluster.dq_mean, dqmean_lo, dqmean_hi)
            cluster.dq_exec_norm = self._norm01(cluster.dq_exec, dqexec_lo, dqexec_hi)
            cluster.m_exec_norm = float(np.clip(cluster.m_exec, 0.0, 1.0))

            compact_term = (1.0 - cluster.dq_mean_norm) if (not cluster.is_singleton) else 0.0
            cluster.score = (
                w_psc * cluster.psc_norm
                + w_compact * compact_term
                + w_exec * (1.0 - cluster.dq_exec_norm)
                + w_mexec * cluster.m_exec_norm
            )

        kept_sorted = sorted(
            kept,
            key=lambda x: float(getattr(x, "score", -1e18)),
            reverse=True,
        )

        for mode_id, cluster in enumerate(kept_sorted):
            cluster.mode_id = int(mode_id)

        best = kept_sorted[0] if kept_sorted else None

        # --------------------------------------------------------------------
        # F) Publish selector summaries and outputs
        # --------------------------------------------------------------------
        self._publish_best_mode_summaries(best)
        self._publish_mode0_members(best, self.base_frame)
        self._publish_markers_and_reps(kept_sorted)
        self._print_action_modes(kept_sorted, joint_names)

        # --------------------------------------------------------------------
        # G) Save CSV diagnostics
        # --------------------------------------------------------------------
        if self.save_csv and self._csv_writer is not None:
            for cluster in kept_sorted:
                nbv_ref = cluster.best_nbv if cluster.best_nbv is not None else cluster.rep_nbv

                psc = getattr(nbv_ref, "psc", None)
                psc = float(psc) if (psc is not None and np.isfinite(float(psc))) else -1.0

                vd = getattr(nbv_ref, "view_dist", None)
                vd = float(vd) if (vd is not None and np.isfinite(float(vd))) else -1.0

                rep_idx = int(cluster.rep_nbv.index) if cluster.rep_nbv is not None else -1

                self._csv_writer.writerow([
                    stamp,
                    nbv_set_id,
                    int(cluster.mode_id),
                    int(cluster.cluster_id),
                    int(len(cluster.nbvs)),
                    int(cluster.is_singleton),
                    float(getattr(cluster, "dq_max", -1.0)),
                    float(getattr(cluster, "dq_mean", -1.0)),
                    float(getattr(cluster, "dq_std", -1.0)),
                    rep_idx,
                    psc,
                    vd,
                    float(getattr(cluster, "dq_exec", -1.0)),
                    float(getattr(cluster, "m_limit", -1.0)),
                    float(getattr(cluster, "m_sing", -1.0)),
                    float(getattr(cluster, "m_exec", -1.0)),
                    float(getattr(cluster, "psc_norm", 0.0)),
                    float(getattr(cluster, "dq_mean_norm", 0.0)),
                    float(getattr(cluster, "dq_exec_norm", 0.0)),
                    float(getattr(cluster, "m_exec_norm", 0.0)),
                    float(getattr(cluster, "score", -1e9)),
                ])

            try:
                self._csv_file.flush()
            except Exception:
                pass


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    try:
        node = NBVSelectorNode()
        rospy.Subscriber(node.nbv_topic, PoseArray, node.nbv_cb, queue_size=1)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass