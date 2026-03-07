#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nbv_selector_example.py (paper-viz enhanced)
- Keep original Layer-2 selection logic untouched
- Add "paper visualization" outputs:
    1) IK-feasible set (PoseArray)
    2) Cluster members as thin arrows + representatives as thick arrows (MarkerArray)
    3) Target apple as semi-transparent red sphere
"""

import rospy
import numpy as np
import moveit_commander
from sklearn.cluster import DBSCAN

import tf2_ros
import tf.transformations as tft

from geometry_msgs.msg import PoseArray, Pose, PointStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA, Float32MultiArray

import threading
import hashlib
import copy


# ============================================================
# Basic data structures
# ============================================================
class NBV(object):
    def __init__(self, index, pose):
        self.index = index
        self.pose = pose
        self.psc = None
        self.view_dist = None
        self.used_dynamic = None


class NBVCluster(object):
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.nbvs = []
        self.q_members = []

        self.q_center = None
        self.rep_nbv = None
        self.rep_q = None

        self.dq_max = None
        self.dq_mean = None
        self.dq_std = None

        self.dq_exec = None
        self.m_limit = None
        self.m_sing = None
        self.m_exec = None

        self.q_best = None
        self.best_nbv = None

        self.psc_norm = None
        self.dq_mean_norm = None
        self.dq_exec_norm = None
        self.m_exec_norm = None
        self.score = None

        self.mode_id = None
        self.is_singleton = False


# ============================================================
# Task-space pre-clusterer (direction + pos, meta-based reps)
# ============================================================
class TaskSpaceClusterer(object):
    def __init__(self,
                 eps_pos=0.06,
                 eps_dir=0.35,
                 min_samples=3,
                 max_rep_per_cluster=2,
                 noise_rep=8,
                 w_psc=1.0,
                 w_vdist=0.2,
                 vdist_pref=0.30,
                 vdist_sigma=0.10):

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
        p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
        v = target_center - p
        v /= (np.linalg.norm(v) + 1e-9)
        return p, v

    @staticmethod
    def _safe_float(x, default=None):
        if x is None:
            return default
        try:
            v = float(x)
        except Exception:
            return default
        if not np.isfinite(v):
            return default
        return v

    def _nbv_meta_score(self, nbv, stats):
        p_lo, p_hi, vd_lo, vd_hi = stats
        psc = self._safe_float(getattr(nbv, "psc", None), None)
        vd  = self._safe_float(getattr(nbv, "view_dist", None), None)

        if psc is None and vd is None:
            return -1e9

        if psc is not None and psc >= 0.0:
            p_n = (psc - p_lo) / (p_hi - p_lo)
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
        psc_vals = []
        vd_vals = []
        for n in nbvs:
            p = self._safe_float(getattr(n, "psc", None), None)
            v = self._safe_float(getattr(n, "view_dist", None), None)
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
        if not nbvs:
            return []

        Z = []
        for nbv in nbvs:
            _, v = self._pose_to_feat(nbv.pose, target_center)
            Z.append(v)
        Z = np.vstack(Z)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)

        N = len(nbvs)
        D = np.zeros((N, N), dtype=np.float64)

        P = np.zeros((N, 3), dtype=np.float64)
        for i, nbv in enumerate(nbvs):
            p, _ = self._pose_to_feat(nbv.pose, target_center)
            P[i] = p

        for i in range(N):
            cos = np.clip(np.dot(Z[i], Z.T), -1.0, 1.0)
            dz = np.arccos(cos) / max(self.eps_dir, 1e-9)
            dp = np.linalg.norm(P[i] - P, axis=1) / max(self.eps_pos, 1e-9)
            D[i] = np.sqrt(dz * dz + dp * dp)

        clustering = DBSCAN(eps=1.0, min_samples=self.min_samples, metric="precomputed").fit(D)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(nbvs[i])

        stats = self._build_norm_stats(nbvs)
        reps = []
        for label, members in clusters.items():
            if label == -1:
                members_sorted = sorted(members, key=lambda n: self._nbv_meta_score(n, stats), reverse=True)
                reps.extend(members_sorted[:max(self.max_rep, self.noise_rep)])
                continue

            members_sorted = sorted(
                members,
                key=lambda n: (self._nbv_meta_score(n, stats), -int(getattr(n, "index", 0))),
                reverse=True
            )
            if len(members_sorted) > 0 and self._nbv_meta_score(members_sorted[0], stats) <= -1e8:
                members_sorted = members

            chosen = []
            for cand in members_sorted:
                if len(chosen) >= self.max_rep:
                    break
                _, vc = self._pose_to_feat(cand.pose, target_center)
                ok = True
                for prev in chosen:
                    _, vp = self._pose_to_feat(prev.pose, target_center)
                    ang = np.arccos(np.clip(np.dot(vc, vp), -1.0, 1.0))
                    if ang < 0.08:
                        ok = False
                        break
                if ok:
                    chosen.append(cand)

            if len(chosen) < self.max_rep:
                for cand in members_sorted:
                    if cand in chosen:
                        continue
                    chosen.append(cand)
                    if len(chosen) >= self.max_rep:
                        break

            reps.extend(chosen)

        rospy.loginfo("[NBVSelector] Task-space reduced %d → %d reps (%d clusters)",
                      len(nbvs), len(reps), len(clusters))
        return reps


# ============================================================
# Joint-space clusterer (IK + DBSCAN) with debug cache for paper viz
# ============================================================
class JointSpaceClusterer(object):
    def __init__(self,
                 move_group="fr3_arm",
                 base_frame="base_link",
                 wrist_link="hand_base_visual_link",
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
                 log_every_sec=1.0):

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

        # paper-viz debug cache (filled each cluster_nbvs call)
        self._last_valid_nbvs = []
        self._last_Q = None
        self._last_labels = None

        rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)

        rospy.loginfo("[NBVSelector] Waiting for /compute_ik ...")
        rospy.wait_for_service("/compute_ik")
        self.ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        rospy.loginfo("[NBVSelector] /compute_ik connected.")

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        self._T_wrist_cam = None
        self._T_wrist_cam_stamp = rospy.Time(0)

        self.ee_link = wrist_link

        # ---- manual joint limits ----
        MANUAL_POS_LIMITS = {
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
            if jn in MANUAL_POS_LIMITS:
                self._joint_limits[jn] = tuple(MANUAL_POS_LIMITS[jn])
            else:
                missing.append(jn)

        rospy.loginfo("[NBVSelector] Active joints: %s", ", ".join(active))
        if missing:
            rospy.logwarn("[NBVSelector] Manual limits missing joints: %s", ", ".join(missing))
            rospy.logwarn("[NBVSelector] If MoveIt joints are not j1~j6, update MANUAL_POS_LIMITS keys.")

    def _js_cb(self, msg):
        self._last_js = msg

    @staticmethod
    def pose_to_T(pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        T = tft.quaternion_matrix(q)
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T

    @staticmethod
    def T_to_pose(T):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        q = tft.quaternion_from_matrix(T)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return pose

    @staticmethod
    def _wrap_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _normalize_q_into_limits(self, q, joint_names, eps=1e-6):
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

    def _joint_distance(self, q_a, q_b):
        dq = self._wrap_diff(q_a, q_b)

        norm = np.ones_like(dq)
        active = self.arm.get_active_joints()
        for i, name in enumerate(active):
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
        tfm = self.tfbuf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(timeout))
        t = tfm.transform.translation
        r = tfm.transform.rotation
        T = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
        T[0, 3], T[1, 3], T[2, 3] = t.x, t.y, t.z
        return T

    def get_T_wrist_cam(self, force_refresh=False):
        now = rospy.Time.now()
        if (self._T_wrist_cam is not None) and (not force_refresh):
            if (now - self._T_wrist_cam_stamp).to_sec() < 5.0:
                return self._T_wrist_cam
        try:
            T_wc = self.lookup_T(self.wrist_link, self.camera_link, timeout=0.5)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[NBVSelector] TF extrinsic %s->%s failed: %s",
                                   self.wrist_link, self.camera_link, str(e))
            return None
        self._T_wrist_cam = T_wc
        self._T_wrist_cam_stamp = now
        return self._T_wrist_cam

    def camera_pose_to_ee_pose(self, cam_pose):
        T_ec = self.get_T_wrist_cam()
        if T_ec is None:
            return None
        T_bc = self.pose_to_T(cam_pose)
        T_be = np.matmul(T_bc, np.linalg.inv(T_ec))
        return self.T_to_pose(T_be)

    def _quat_to_rpy(self, quat_xyzw):
        roll, pitch, yaw = tft.euler_from_quaternion(quat_xyzw)
        return float(roll), float(pitch), float(yaw)

    def _wrist_pose_key(self, ee_pose):
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
        ts = self._neg_cache.get(key, None)
        if ts is None:
            return False
        return (now_sec - ts) < self.neg_cache_ttl_sec

    def _neg_cache_put(self, key, now_sec):
        if len(self._neg_cache) >= self.max_neg_cache_size:
            self._neg_cache.clear()
        self._neg_cache[key] = now_sec

    def _ik_cache_put(self, key, q):
        if len(self._ik_cache) >= self.max_cache_size:
            self._ik_cache.clear()
        self._ik_cache[key] = q.copy()

    def _make_robot_state_seed(self):
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

    def solve_ik_ee(self, ee_pose):
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
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, "[NBVSelector] IK service call failed: %s", str(e))
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            rospy.logwarn_throttle(1.0, "[NBVSelector] IK failed, error_code=%d", res.error_code.val)
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        js = res.solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        joint_names = self.arm.get_active_joints()
        try:
            q = np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
        except KeyError:
            rospy.logwarn_throttle(1.0, "[NBVSelector] IK solution missing expected joint names.")
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        q, ok_norm = self._normalize_q_into_limits(q, joint_names)
        if not ok_norm:
            rospy.logwarn_throttle(1.0, "[NBVSelector] IK solution out of joint limits after normalization -> reject")
            if key is not None:
                self._neg_cache_put(key, now_sec)
            return None

        self._last_good_q = q.copy()
        if key is not None:
            self._ik_cache_put(key, q)
        return q

    def solve_ik_from_camera_pose(self, cam_pose):
        ee_pose = self.camera_pose_to_ee_pose(cam_pose)
        if ee_pose is None:
            return None
        return self.solve_ik_ee(ee_pose)

    def compute_jacobian_condition(self, q):
        try:
            q_list = [float(x) for x in q]
            J = self.arm.get_jacobian_matrix(q_list)
            J = np.asarray(J, dtype=np.float64)
            if J.size == 0:
                return float("inf")
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            if S.size == 0 or np.min(S) < 1e-6:
                return float("inf")
            return float(np.max(S) / np.min(S))
        except Exception as e:
            rospy.logwarn_throttle(1.0, "[NBVSelector] Jacobian computation failed: %s", str(e))
            return float("inf")

    def cluster_nbvs(self, nbv_list):
        rospy.loginfo("[NBVSelector] Clustering %d NBVs (seed+cache)", len(nbv_list))

        Q = []
        valid_nbvs = []

        total = len(nbv_list)
        ok = 0
        fail = 0

        solve_calls_before = self._solve_calls

        for nbv in nbv_list:
            q = self.solve_ik_from_camera_pose(nbv.pose)
            if q is None:
                fail += 1
                continue
            ok += 1
            Q.append(q)
            valid_nbvs.append(nbv)

        solve_calls_after = self._solve_calls
        if self.cache_enabled:
            service_calls = max(0, solve_calls_after - solve_calls_before)
            approx_cache_hit = max(0, total - service_calls)
        else:
            approx_cache_hit = 0

        rospy.loginfo("[NBVSelector] IK stats: total=%d ok=%d fail=%d approx_cache_hit=%d cache=%d neg=%d",
                      total, ok, fail, approx_cache_hit, len(self._ik_cache), len(self._neg_cache))

        # cache for paper viz
        self._last_valid_nbvs = list(valid_nbvs)
        self._last_Q = None
        self._last_labels = None

        if not Q:
            rospy.logwarn("[NBVSelector] No valid IK solutions.")
            return []

        Q = np.vstack(Q)
        self._last_Q = Q.copy()

        N = Q.shape[0]
        D = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                d = self._joint_distance(Q[i], Q[j])
                D[i, j] = d
                D[j, i] = d

        clustering = DBSCAN(
            eps=self.eps_joint,
            min_samples=self.min_samples,
            metric="precomputed"
        ).fit(D)

        labels = clustering.labels_
        self._last_labels = labels.copy()

        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, NBVCluster(label))
            clusters[label].nbvs.append(valid_nbvs[i])
            clusters[label].q_members.append(Q[i])

        cluster_list = []
        for cid, cluster in clusters.items():
            qs = np.vstack(cluster.q_members)
            cluster.q_center = np.mean(qs, axis=0)
            cluster.q_center, _ = self._normalize_q_into_limits(cluster.q_center, self.arm.get_active_joints())

            # representative: prefer max psc else nearest center
            psc_list = []
            for nbv_i in cluster.nbvs:
                c = nbv_i.psc
                psc_list.append(-1.0 if c is None else float(c))
            psc_arr = np.asarray(psc_list, dtype=np.float32)

            if np.any(psc_arr >= 0.0):
                rep_idx = int(np.argmax(psc_arr))
            else:
                dists = np.linalg.norm(qs - cluster.q_center[None, :], axis=1)
                rep_idx = int(np.argmin(dists))

            cluster.rep_nbv = cluster.nbvs[rep_idx]
            cluster.rep_q = qs[rep_idx].copy()
            cluster_list.append(cluster)

        # compactness gate (keep your original behavior)
        max_diam_thr = float(rospy.get_param("~max_diam_norm", 0.25))
        for c in cluster_list:
            c.is_singleton = (len(c.nbvs) == 1)

        rospy.loginfo("[NBVSelector] Reduced %d reps to %d Action-Modes",
                      len(nbv_list), len(cluster_list))
        return cluster_list


# ============================================================
# Visualization helpers
# ============================================================
def _color_from_id(i, alpha=0.9):
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


def _vec_unit(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


# ============================================================
# ROS Node Wrapper
# ============================================================
class NBVSelectorNode(object):
    def __init__(self):
        rospy.init_node("nbv_selector")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.nbv_topic = rospy.get_param("~nbv_topic", "/nbv/valid_view_poses")

        # --- original viz toggles ---
        self.viz_enable = rospy.get_param("~viz_enable", True)
        self.viz_show_members = rospy.get_param("~viz_show_members", True)
        self.viz_line_to_target = rospy.get_param("~viz_line_to_target", True)
        self.viz_arrow_len = float(rospy.get_param("~viz_arrow_len", 0.12))
        self.viz_member_scale = float(rospy.get_param("~viz_member_scale", 0.025))
        self.viz_target_scale = float(rospy.get_param("~viz_target_scale", 0.04))
        self.viz_text_scale = float(rospy.get_param("~viz_text_scale", 0.06))

        # --- paper viz toggles (NEW) ---
        self.paper_viz_enable = bool(rospy.get_param("~paper_viz_enable", True))
        self.paper_viz_publish_ik_feasible = bool(rospy.get_param("~paper_viz_publish_ik_feasible", True))
        self.paper_viz_show_unclustered = bool(rospy.get_param("~paper_viz_show_unclustered", False))
        self.paper_viz_show_text = bool(rospy.get_param("~paper_viz_show_text", False))
        self.paper_viz_member_alpha = float(rospy.get_param("~paper_viz_member_alpha", 0.25))
        self.paper_viz_rep_alpha = float(rospy.get_param("~paper_viz_rep_alpha", 0.95))
        self.paper_viz_member_len = float(rospy.get_param("~paper_viz_member_len", 0.10))
        self.paper_viz_rep_len = float(rospy.get_param("~paper_viz_rep_len", 0.14))
        self.paper_viz_member_scale_x = float(rospy.get_param("~paper_viz_member_scale_x", 0.006))
        self.paper_viz_member_scale_y = float(rospy.get_param("~paper_viz_member_scale_y", 0.012))
        self.paper_viz_rep_scale_x = float(rospy.get_param("~paper_viz_rep_scale_x", 0.012))
        self.paper_viz_rep_scale_y = float(rospy.get_param("~paper_viz_rep_scale_y", 0.028))
        self.paper_viz_target_alpha = float(rospy.get_param("~paper_viz_target_alpha", 0.45))
        self.paper_viz_lifetime = float(rospy.get_param("~paper_viz_lifetime", 5.0))
        self.paper_viz_disable_task_reduction = bool(rospy.get_param("~paper_viz_disable_task_reduction", False))

        # --- execution cost weights ---
        self.joint_weights = rospy.get_param("~joint_weights", [1.5, 2.0, 2.0, 0.8, 0.6, 0.5])
        self.joint_weights = np.asarray(self.joint_weights, dtype=np.float64)

        # --- diagnostics output (kept) ---
        self.save_csv = rospy.get_param("~save_csv", True)
        self.out_dir = rospy.get_param("~out_dir", "/tmp")
        ts = rospy.Time.now().to_sec()
        self.csv_path = f"{self.out_dir}/nbv_action_modes_{int(ts)}.csv"
        self._csv_file = None
        self._csv_writer = None
        self._nbv_set_counter = 0

        if self.save_csv:
            import csv
            self._csv_file = open(self.csv_path, "w")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "stamp","nbv_set_id","mode_id","cluster_id","cluster_size","is_singleton",
                "dq_max","dq_mean","dq_std","rep_nbv_idx","psc","view_dist",
                "dq_exec","m_limit","m_sing","m_exec",
                "psc_norm","dq_mean_norm","dq_exec_norm","m_exec_norm","score",
            ])
            rospy.loginfo("[NBVSelector] Saving diagnostics to %s", self.csv_path)

        rospy.on_shutdown(self._on_shutdown)

        # modules
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

        # pubs (original)
        self.marker_pub = rospy.Publisher("/nbv/action_modes_markers", MarkerArray, queue_size=1)
        self.reps_pub = rospy.Publisher("/nbv/action_modes_reps", PoseArray, queue_size=1)

        # paper pubs (NEW)
        self.ik_feasible_pub = rospy.Publisher("/nbv/ik_feasible_view_poses", PoseArray, queue_size=1)
        self.paper_marker_pub = rospy.Publisher("/nbv/action_modes_paper_markers", MarkerArray, queue_size=1)

        # state
        self.target_center = None
        self.nbv_sub = None
        rospy.Subscriber("/apple/center", PointStamped, self._target_cb, queue_size=1)
        rospy.loginfo("[NBVSelector] Waiting for target_center...")

        self._last_meta = None
        self._last_meta_time = rospy.Time(0)
        rospy.Subscriber("/nbv/valid_view_meta", Float32MultiArray, self._meta_cb, queue_size=1)

        # scheduling/buffer
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
        self._timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, tick_hz)), self._tick)

    def _on_shutdown(self):
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass

    def _target_cb(self, msg):
        if self.target_center is None:
            rospy.loginfo("[NBVSelector] target_center received, enabling NBV processing")
            self.nbv_sub = rospy.Subscriber(self.nbv_topic, PoseArray, self.nbv_cb, queue_size=1)
        self.target_center = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def _meta_cb(self, msg):
        self._last_meta = msg.data[:]
        self._last_meta_time = rospy.Time.now()

    def nbv_cb(self, msg):
        if self.target_center is None:
            return
        with self._lock:
            self._latest_msg = msg

    def _hash_pose_array(self, pose_array):
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
                    h = self._hash_pose_array(msg)
                    if self._last_hash is not None and h == self._last_hash:
                        return
                except Exception:
                    h = None
            else:
                h = None

            meta = None
            if self._last_meta is not None:
                expected = int(rospy.get_param("~meta_stride", 3)) * len(msg.poses)
                if len(self._last_meta) >= expected:
                    meta = self._last_meta[:]
                    age = (rospy.Time.now() - self._last_meta_time).to_sec()
                    if age > self._meta_max_age:
                        rospy.logwarn_throttle(1.0, "[NBVSelector] meta old: age=%.3fs", age)

            self._busy = True
            self._last_proc_stamp = now
            if h is not None:
                self._last_hash = h

        try:
            self._process(msg, meta)
        finally:
            with self._lock:
                self._busy = False

    # ======= paper viz publisher (NEW) =======
    def _publish_paper_markers(self, clusters):
        if not self.paper_viz_enable:
            return
        from geometry_msgs.msg import Point

        frame = self.base_frame
        now = rospy.Time.now()
        ma = MarkerArray()

        mk_clear = Marker()
        mk_clear.header.frame_id = frame
        mk_clear.header.stamp = now
        mk_clear.ns = "paper/clear"
        mk_clear.id = 0
        mk_clear.action = Marker.DELETEALL
        ma.markers.append(mk_clear)
        mid = 1

        # target apple sphere (semi-transparent red)
        if self.target_center is not None:
            mk_t = Marker()
            mk_t.header.frame_id = frame
            mk_t.header.stamp = now
            mk_t.ns = "paper/target"
            mk_t.id = mid
            mk_t.type = Marker.SPHERE
            mk_t.action = Marker.ADD
            mk_t.lifetime = rospy.Duration(0.0)
            mk_t.pose.position.x = float(self.target_center[0])
            mk_t.pose.position.y = float(self.target_center[1])
            mk_t.pose.position.z = float(self.target_center[2])
            mk_t.pose.orientation.w = 1.0
            mk_t.scale.x = self.viz_target_scale
            mk_t.scale.y = self.viz_target_scale
            mk_t.scale.z = self.viz_target_scale
            mk_t.color = ColorRGBA(1.0, 0.2, 0.2, float(self.paper_viz_target_alpha))
            ma.markers.append(mk_t)
            mid += 1

        # Use cached labels from clusterer to draw thin member arrows
        valid_nbvs = getattr(self.clusterer, "_last_valid_nbvs", []) or []
        labels = getattr(self.clusterer, "_last_labels", None)

        # Build map: valid_nbv_index_in_valid_list -> (mode_id or None)
        # We color by mode_id (after sorting), so build nbv->mode mapping from clusters
        nbv_obj_to_mode = {}
        for c in clusters:
            for n in c.nbvs:
                nbv_obj_to_mode[id(n)] = int(c.mode_id)

        # members: thin arrows
        if valid_nbvs and labels is not None:
            for i, n in enumerate(valid_nbvs):
                lab = int(labels[i]) if i < len(labels) else -1
                if (lab == -1) and (not self.paper_viz_show_unclustered):
                    continue

                mode_id = nbv_obj_to_mode.get(id(n), None)
                if mode_id is None:
                    # not kept in final list (e.g., filtered singleton cap), skip for paper
                    continue

                col = _color_from_id(mode_id, alpha=self.paper_viz_member_alpha)

                p = n.pose.position
                pos = np.array([p.x, p.y, p.z], dtype=np.float64)
                vdir = _vec_unit(self.target_center - pos) if self.target_center is not None else np.array([0.0, 0.0, 1.0])

                mk = Marker()
                mk.header.frame_id = frame
                mk.header.stamp = now
                mk.ns = f"paper/members/mode_{mode_id}"
                mk.id = mid
                mk.type = Marker.ARROW
                mk.action = Marker.ADD
                mk.lifetime = rospy.Duration(0.0)
                mk.pose.orientation.w = 1.0
                mk.color = col
                mk.scale.x = self.paper_viz_member_scale_x
                mk.scale.y = self.paper_viz_member_scale_y
                mk.scale.z = self.paper_viz_member_scale_y
                p0 = pos
                p1 = pos + vdir * float(self.paper_viz_member_len)
                mk.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                             Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
                ma.markers.append(mk)
                mid += 1

        # reps: thick arrows at best pose
        for c in clusters:
            mode_id = int(c.mode_id)
            col = _color_from_id(mode_id, alpha=self.paper_viz_rep_alpha)

            exec_pose = c.best_nbv.pose if c.best_nbv is not None else c.rep_nbv.pose
            pos = np.array([exec_pose.position.x, exec_pose.position.y, exec_pose.position.z], dtype=np.float64)
            vdir = _vec_unit(self.target_center - pos) if self.target_center is not None else np.array([0.0, 0.0, 1.0])

            mk = Marker()
            mk.header.frame_id = frame
            mk.header.stamp = now
            mk.ns = f"paper/rep/mode_{mode_id}"
            mk.id = mid
            mk.type = Marker.ARROW
            mk.action = Marker.ADD
            mk.lifetime = rospy.Duration(0.0)
            mk.pose.orientation.w = 1.0
            mk.color = col
            mk.scale.x = self.paper_viz_rep_scale_x
            mk.scale.y = self.paper_viz_rep_scale_y
            mk.scale.z = self.paper_viz_rep_scale_y
            p0 = pos
            p1 = pos + vdir * float(self.paper_viz_rep_len)
            mk.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                         Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
            ma.markers.append(mk)
            mid += 1

            if self.paper_viz_show_text:
                mk_txt = Marker()
                mk_txt.header.frame_id = frame
                mk_txt.header.stamp = now
                mk_txt.ns = f"paper/text/mode_{mode_id}"
                mk_txt.id = mid
                mk_txt.type = Marker.TEXT_VIEW_FACING
                mk_txt.action = Marker.ADD
                mk_txt.lifetime = rospy.Duration(0.0)
                mk_txt.pose.position.x = float(pos[0])
                mk_txt.pose.position.y = float(pos[1])
                mk_txt.pose.position.z = float(pos[2] + 0.06)
                mk_txt.pose.orientation.w = 1.0
                mk_txt.scale.z = self.viz_text_scale
                mk_txt.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
                mk_txt.text = f"Mode {mode_id} n={len(c.nbvs)}"
                ma.markers.append(mk_txt)
                mid += 1

        self.paper_marker_pub.publish(ma)

    def _publish_ik_feasible_posearray(self):
        if (not self.paper_viz_enable) or (not self.paper_viz_publish_ik_feasible):
            return
        valid_nbvs = getattr(self.clusterer, "_last_valid_nbvs", []) or []
        pa = PoseArray()
        pa.header = Header(stamp=rospy.Time.now(), frame_id=self.base_frame)
        pa.poses = [n.pose for n in valid_nbvs]
        self.ik_feasible_pub.publish(pa)

    # ======= scoring helpers (kept from your code, minimally included) =======
    def _robust_minmax(self, arr, lo_q=10.0, hi_q=90.0, fallback=(0.0, 1.0)):
        arr = [float(x) for x in arr if x is not None and np.isfinite(x)]
        if len(arr) == 0:
            return float(fallback[0]), float(fallback[1])
        lo = float(np.percentile(arr, lo_q))
        hi = float(np.percentile(arr, hi_q))
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        return lo, hi

    def _norm01(self, x, lo, hi):
        if x is None:
            return 0.0
        try:
            xv = float(x)
        except Exception:
            return 0.0
        if not np.isfinite(xv):
            return 0.0
        v = (xv - float(lo)) / (float(hi) - float(lo) + 1e-12)
        return float(np.clip(v, 0.0, 1.0))

    def _get_current_q(self, joint_names):
        js = self.clusterer._last_js
        if js is None:
            return None
        name_to_pos = dict(zip(js.name, js.position))
        try:
            return np.array([name_to_pos[n] for n in joint_names], dtype=np.float64)
        except KeyError:
            return None

    def _compute_m_limit(self, q, joint_names):
        jl = getattr(self.clusterer, "_joint_limits", None)
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
        qa = np.asarray(q_a, dtype=np.float64)
        qb = np.asarray(q_b, dtype=np.float64)
        if qa.shape != qb.shape:
            return 1e18
        dq = self.clusterer._wrap_diff(qa, qb)
        w = self.joint_weights
        if w.size != dq.size:
            return float(np.linalg.norm(dq))
        return float(np.sqrt(np.sum((w * dq) ** 2)))

    def _process(self, msg, meta_snapshot):
        META_STRIDE = int(rospy.get_param("~meta_stride", 3))
        IDX_PSC = int(rospy.get_param("~meta_idx_psc", 0))
        IDX_VIEW_DIST = int(rospy.get_param("~meta_idx_view_dist", 1))
        IDX_USED_DYNAMIC = int(rospy.get_param("~meta_idx_used_dynamic", 2))

        stamp = rospy.Time.now().to_sec()
        nbv_set_id = self._nbv_set_counter
        self._nbv_set_counter += 1

        if self.target_center is None:
            return
        if msg is None or len(msg.poses) == 0:
            return

        nbvs = [NBV(i, pose) for i, pose in enumerate(msg.poses)]

        if meta_snapshot is not None and len(meta_snapshot) >= META_STRIDE * len(nbvs):
            for nbv in nbvs:
                i = int(nbv.index)
                base = META_STRIDE * i
                try:
                    nbv.psc = float(meta_snapshot[base + IDX_PSC])
                except Exception:
                    pass
                try:
                    nbv.view_dist = float(meta_snapshot[base + IDX_VIEW_DIST])
                except Exception:
                    pass
                try:
                    nbv.used_dynamic = float(meta_snapshot[base + IDX_USED_DYNAMIC])
                except Exception:
                    pass

        # A) task-space reduction (optionally disable for paper viz)
        if self.paper_viz_enable and self.paper_viz_disable_task_reduction:
            reps = nbvs
        else:
            reps = self.task_clusterer.cluster(nbvs, target_center=self.target_center)

        if not reps:
            rospy.logwarn_throttle(1.0, "[NBVSelector] No reps after task-space reduction.")
            return

        # B) joint-space clustering (includes IK inside)
        clusters = self.clusterer.cluster_nbvs(reps)
        if not clusters:
            rospy.logwarn_throttle(1.0, "[NBVSelector] No Action-Modes after clustering.")
            return

        # publish IK feasible set (NEW)
        self._publish_ik_feasible_posearray()

        joint_names = self.clusterer.arm.get_active_joints()
        q_current = self._get_current_q(joint_names)
        if (q_current is None) or (not np.all(np.isfinite(q_current))):
            if getattr(self.clusterer, "_last_good_q", None) is not None and np.all(np.isfinite(self.clusterer._last_good_q)):
                q_current = self.clusterer._last_good_q.copy()
            else:
                q_current = np.zeros((len(joint_names),), dtype=np.float64)

        # C/D) choose q_best + margins
        m_exec_thresh = float(rospy.get_param("~m_exec_thresh", 0.05))
        pen_w = float(rospy.get_param("~mexec_pen_w", 2.0))

        kept = []
        for c in clusters:
            qs = np.vstack(c.q_members)

            dq_to_center = np.array([self.clusterer._joint_distance(qs[i], c.q_center) for i in range(qs.shape[0])], dtype=np.float64)
            c.dq_max = float(np.max(dq_to_center))
            c.dq_mean = float(np.mean(dq_to_center))
            c.dq_std = float(np.std(dq_to_center))

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

                psc_m = c.nbvs[i_m].psc
                psc_m = float(psc_m) if (psc_m is not None and np.isfinite(psc_m) and psc_m >= 0.0) else -1e18

                if (cost < best_any_cost - 1e-9) or (abs(cost - best_any_cost) <= 1e-9 and psc_m > best_any_psc):
                    best_any_cost, best_any_idx, best_any_psc = cost, int(i_m), psc_m

                if m_exec >= m_exec_thresh:
                    have_safe = True
                    if (cost < best_safe_cost - 1e-9) or (abs(cost - best_safe_cost) <= 1e-9 and psc_m > best_safe_psc):
                        best_safe_cost, best_safe_idx, best_safe_psc = cost, int(i_m), psc_m

            best_idx = best_safe_idx if (have_safe and best_safe_idx is not None) else best_any_idx
            best_cost = best_safe_cost if (have_safe and best_safe_idx is not None) else best_any_cost

            c.q_best = qs[best_idx].copy()
            c.best_nbv = c.nbvs[best_idx]
            c.dq_exec = float(best_cost)

            c.m_limit = float(self._compute_m_limit(c.q_best, joint_names))
            c.m_sing = float(self._compute_m_sing(c.q_best))
            c.m_exec = float(min(c.m_limit, c.m_sing))

            c.is_singleton = (len(c.nbvs) == 1)
            kept.append(c)

        if not kept:
            return

        # normalize + score (keep your weights)
        psc_list, dq_mean_list, dq_exec_list, m_exec_list = [], [], [], []
        for c in kept:
            nbv_ref = c.best_nbv if c.best_nbv is not None else c.rep_nbv
            psc = getattr(nbv_ref, "psc", None)
            psc = float(psc) if (psc is not None and np.isfinite(psc) and psc >= 0.0) else 0.0
            psc_list.append(psc)
            dq_mean_list.append(float(c.dq_mean))
            dq_exec_list.append(float(c.dq_exec))
            m_exec_list.append(float(c.m_exec))

        psc_lo, psc_hi = self._robust_minmax(psc_list, fallback=(0.0, 1.0))
        dqmean_lo, dqmean_hi = self._robust_minmax(dq_mean_list, fallback=(0.0, 1.0))
        dqexec_lo, dqexec_hi = self._robust_minmax(dq_exec_list, fallback=(0.0, 1.0))

        w_psc = float(rospy.get_param("~score_w_psc", 1.0))
        w_pscact = float(rospy.get_param("~score_w_pscact", 0.6))
        w_exec = float(rospy.get_param("~score_w_exec", 0.8))
        w_mexec = float(rospy.get_param("~score_w_mexec", 1.0))

        for c in kept:
            nbv_ref = c.best_nbv if c.best_nbv is not None else c.rep_nbv
            psc = getattr(nbv_ref, "psc", None)
            psc = float(psc) if (psc is not None and np.isfinite(psc) and psc >= 0.0) else 0.0

            psc_n = self._norm01(psc, psc_lo, psc_hi)
            dq_mean_n = self._norm01(c.dq_mean, dqmean_lo, dqmean_hi)
            dq_exec_n = self._norm01(c.dq_exec, dqexec_lo, dqexec_hi)
            m_exec_n = float(np.clip(c.m_exec, 0.0, 1.0))

            c.psc_norm = psc_n
            c.dq_mean_norm = dq_mean_n
            c.dq_exec_norm = dq_exec_n
            c.m_exec_norm = m_exec_n

            pscact_term = (1.0 - dq_mean_n) if not c.is_singleton else 0.0

            c.score = (
                w_psc * psc_n
                + w_pscact * pscact_term
                + w_exec * (1.0 - dq_exec_n)
                + w_mexec * m_exec_n
            )

        kept_sorted = sorted(kept, key=lambda x: float(getattr(x, "score", -1e18)), reverse=True)
        for mid, c in enumerate(kept_sorted):
            c.mode_id = int(mid)

        # (optional) singleton cap (keep your behavior)
        MAX_SINGLETON = int(rospy.get_param("~max_singleton_modes", 5))
        multi = [c for c in kept_sorted if not getattr(c, "is_singleton", False)]
        single = [c for c in kept_sorted if getattr(c, "is_singleton", False)]
        single = sorted(single, key=lambda c: c.score, reverse=True)[:MAX_SINGLETON]
        final_modes = sorted(multi + single, key=lambda c: c.score, reverse=True)
        kept_sorted = final_modes

        # paper markers (NEW)
        self._publish_paper_markers(kept_sorted)

        if rospy.get_param("~paper_freeze_after_publish", False):
            with self._lock:
                self._latest_msg = None

        # csv (kept, minimal)
        if self.save_csv and self._csv_writer is not None:
            for c in kept_sorted:
                nbv_ref = c.best_nbv if c.best_nbv is not None else c.rep_nbv
                psc = getattr(nbv_ref, "psc", None)
                psc = float(psc) if (psc is not None and np.isfinite(float(psc))) else -1.0
                vd = getattr(nbv_ref, "view_dist", None)
                vd = float(vd) if (vd is not None and np.isfinite(float(vd))) else -1.0
                rep_idx = int(c.rep_nbv.index) if c.rep_nbv is not None else -1
                self._csv_writer.writerow([
                    stamp, nbv_set_id,
                    int(c.mode_id), int(c.cluster_id), int(len(c.nbvs)), int(c.is_singleton),
                    float(getattr(c, "dq_max", -1.0)),
                    float(getattr(c, "dq_mean", -1.0)),
                    float(getattr(c, "dq_std", -1.0)),
                    rep_idx, psc, vd,
                    float(getattr(c, "dq_exec", -1.0)),
                    float(getattr(c, "m_limit", -1.0)),
                    float(getattr(c, "m_sing", -1.0)),
                    float(getattr(c, "m_exec", -1.0)),
                    float(getattr(c, "psc_norm", 0.0)),
                    float(getattr(c, "dq_mean_norm", 0.0)),
                    float(getattr(c, "dq_exec_norm", 0.0)),
                    float(getattr(c, "m_exec_norm", 0.0)),
                    float(getattr(c, "score", -1e9)),
                ])
            try:
                self._csv_file.flush()
            except Exception:
                pass


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    try:
        NBVSelectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


