#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
silhouette_nbv_analyzer.py

Layer-1 geometric NBV generator with PSC-based utility scoring.

Overview
--------
This module implements the geometric perception layer of the active vision
pipeline. Its responsibility is to generate geometrically valid and
perceptually informative Next-Best-View (NBV) candidates in SE(3), without
considering joint-space feasibility or execution cost.

All action-level reasoning, including IK feasibility, singularity avoidance,
joint motion cost, representative selection, and final execution-oriented
decision making, is intentionally deferred to Layer-2.

Layer-1 answers only one question:

    "Which viewpoints are geometrically meaningful and perceptually beneficial?"

It does not answer:

    - Can the robot reach this pose?
    - What is the execution cost?
    - Which NBV should finally be executed?

Those questions are handled by the downstream selection layer.

Core responsibilities
---------------------
1. Silhouette-driven candidate generation
   - Treat each silhouette as a structured geometric entity
   - Generate outward-facing viewpoint hypotheses

2. Conservative workspace pruning
   - Remove obviously unreachable regions
   - Use only a lightweight geometric envelope
   - No IK involved

3. Occlusion-aware filtering
   - KDTree-based raycasting against occupied voxels
   - Single-ray line-of-sight gating with temporal voting
   - Preserve geometric visibility consistency

4. Orientation resolution
   - Constrain the final viewing direction inside a silhouette-centered cone
   - Bias the direction toward the current camera direction
   - Improve kinematic friendliness without invoking IK

5. Dynamic view-distance refinement
   - Optimize viewpoint position along the silhouette axis
   - Hard constraints:
       * workspace
       * clearance
       * anchor visibility
   - Soft costs:
       * reach-boundary proximity
       * yaw deviation
       * arm-direction alignment
       * distance regularization

6. PSC (Predicted Surface Coverage) scoring
   - Patch-based spherical discretization
   - Z-buffer-based expected coverage estimation
   - Measures expected gain on unobserved surface area
   - Independent of robot kinematics

Published outputs
-----------------
/nbv/valid_view_poses : geometry_msgs/PoseArray
    Final valid NBV poses in the base frame.

/nbv/valid_view_meta : std_msgs/Float32MultiArray
    Packed as:
        [psc, view_dist, used_dynamic] * N

    where:
        psc          : predicted surface coverage in [0, 1]
        view_dist    : selected viewing distance
        used_dynamic : 1 if dynamic distance refinement was applied

Architectural role
------------------
Layer-1 = perception-driven geometric reasoning
Layer-2 = execution-aware decision layer

This separation provides:
    - clean responsibility boundaries
    - modular experimentation
    - better theoretical clarity
    - no entanglement between geometry and action cost
"""

import threading
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf import transformations as tft
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class NBVConfig:
    # ------------------------------------------------------------------------
    # Frames and sensor configuration
    # ------------------------------------------------------------------------
    base_frame: str = "base_link"
    camera_frame: str = "in_hand_camera_color_optical_frame"

    # ------------------------------------------------------------------------
    # ROS topics
    # ------------------------------------------------------------------------
    topic_occ: str = "/octomap_point_cloud_centers"
    topic_front: str = "/nbv/Silhouette_points"
    topic_apple_surface_obs: str = "/segmented_apple_pointcloud"
    topic_apple_surface_completed: str = "/completed_apple_pointcloud"

    pub_candidates: str = "/nbv/candidate_views"
    pub_valid: str = "/nbv/valid_views"
    pub_invalid: str = "/nbv/invalid_views"
    pub_valid_poses: str = "/nbv/valid_view_poses"
    pub_valid_markers: str = "/nbv/valid_view_markers"

    # ------------------------------------------------------------------------
    # Geometric candidate generation
    # ------------------------------------------------------------------------
    view_distance: float = 0.20
    max_Silhouettes: int = 400

    # ------------------------------------------------------------------------
    # Pipeline switches
    # ------------------------------------------------------------------------
    enable_workspace_filter: bool = True
    enable_raycast_filter: bool = True
    enable_view_orient_pref: bool = True
    enable_dynamic_view_distance: bool = True

    # ------------------------------------------------------------------------
    # Workspace constraints
    # ------------------------------------------------------------------------
    reach_r_min: float = 0.10
    reach_r_max: float = 0.80
    reach_z_min: float = 0.10
    reach_z_max: float = 0.70
    reach_yaw_max: float = np.deg2rad(170)
    cam_yaw_fallback_mode: str = "hold"

    # ------------------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------------------
    ray_step: float = 0.01
    min_raycast_rays: int = 3
    max_ray_len: float = 0.6
    occ_radius: float = 0.01
    thickness: float = 0.01
    Silhouette_ray_stride: int = 2
    ray_start_margin: float = 0.02
    ray_end_margin: float = 0.03
    blocked_ratio_thresh: float = 0.6

    candidate_key_res: float = 0.005
    candidate_key_res_anchor: float = 0.03
    candidate_key_res_view: float = 0.06

    tv_key_use_view: bool = True
    tv_score_max: int = 5
    tv_score_min: int = -5
    tv_inc: int = 1
    tv_dec: int = 1
    tv_th_valid: int = 3
    tv_th_invalid: int = -2
    tv_ttl_sec: float = 5.0

    max_steps_cap: int = 150

    # ------------------------------------------------------------------------
    # Occupancy inflation
    # ------------------------------------------------------------------------
    enable_occ_inflation: bool = True
    occ_voxel_res: float = 0.01
    occ_inflation_radius: float = 0.025
    occ_inflation_cap_points: int = 350000
    occ_downsample_res: float = 0.01

    # ------------------------------------------------------------------------
    # View orientation preference
    # ------------------------------------------------------------------------
    theta_max_deg: float = 60.0
    w_Silhouette: float = 1.0
    w_arm: float = 0.4

    # ------------------------------------------------------------------------
    # Dynamic distance search
    # ------------------------------------------------------------------------
    d_min: float = 0.20
    d_max: float = 0.30
    d_samples: int = 15

    w_dist_reach: float = 1.0
    w_dist_yaw: float = 0.6
    w_dist_arm: float = 0.4

    dist_tiebreak_eps: float = 0.02
    dist_tiebreak_mode: str = "near"

    # ------------------------------------------------------------------------
    # Expected completion margins
    # ------------------------------------------------------------------------
    ec_yaw_margin_deg: float = 15.0
    ec_pitch_margin_deg: float = 8.0

    # ------------------------------------------------------------------------
    # TF and logging
    # ------------------------------------------------------------------------
    tf_timeout_sec: float = 1.0
    tf_retry_once: bool = True
    log_every_sec: float = 1.0

    # ------------------------------------------------------------------------
    # Pose construction
    # ------------------------------------------------------------------------
    up_vector_base: Optional[np.ndarray] = None

    # ------------------------------------------------------------------------
    # PSC (z-buffer only)
    # ------------------------------------------------------------------------
    psc_enable_zbuffer: bool = True
    psc_img_w: int = 160
    psc_img_h: int = 120
    psc_fov_deg: float = 45.0

    psc_min_completed_points: int = 80
    psc_voxel_res: float = 0.004
    psc_use_radius_band: bool = True
    psc_radius_tol: float = 0.015
    psc_log_throttle_sec: float = 1.0

    def __post_init__(self):
        if self.up_vector_base is None:
            self.up_vector_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.theta_max = np.deg2rad(self.theta_max_deg)


# ============================================================================
# Silhouette entity abstraction
# ============================================================================
@dataclass
class SilhouetteEntity:
    """
    Conceptual silhouette entity.

    Attributes
    ----------
    anchor_base : np.ndarray
        Silhouette anchor point in the base frame.
    d_Silhouette_base : np.ndarray
        Geometric silhouette axis in the base frame.
    yaw_center : float
        Workspace-conditioned yaw preference.
    """
    anchor_base: np.ndarray
    d_Silhouette_base: np.ndarray
    yaw_center: float


# ============================================================================
# Utility functions
# ============================================================================
def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """
    Convert PointCloud2 to an (N, 3) float32 array.

    This implementation reads x/y/z directly from the raw byte buffer for speed.
    """
    if msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)

    off_x = off_y = off_z = None
    for field in msg.fields:
        if field.name == "x":
            off_x = field.offset
        elif field.name == "y":
            off_y = field.offset
        elif field.name == "z":
            off_z = field.offset

    if off_x is None or off_y is None or off_z is None:
        return np.zeros((0, 3), dtype=np.float32)

    n_points = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    expected = n_points * msg.point_step
    if buf.size < expected:
        return np.zeros((0, 3), dtype=np.float32)

    buf = buf[:expected].reshape(n_points, msg.point_step)

    x = buf[:, off_x:off_x + 4].view(np.float32).reshape(-1)
    y = buf[:, off_y:off_y + 4].view(np.float32).reshape(-1)
    z = buf[:, off_z:off_z + 4].view(np.float32).reshape(-1)

    pts = np.stack((x, y, z), axis=1)
    valid = np.isfinite(pts).all(axis=1)
    if not np.all(valid):
        pts = pts[valid]

    return pts.astype(np.float32, copy=False)


def xyz_to_pc2(points: np.ndarray, frame: str) -> PointCloud2:
    """Convert an (N, 3) array to a PointCloud2 message."""
    header = Header(stamp=rospy.Time.now(), frame_id=frame)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    pts = np.asarray(points, dtype=np.float32)
    return pc2.create_cloud(header, fields, pts)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector with numerical protection."""
    return v / (np.linalg.norm(v) + 1e-9)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Normalize each row of a matrix."""
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def angle_wrap(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_T(trans, quat) -> np.ndarray:
    """Build a homogeneous transform matrix from translation and quaternion."""
    transform = tft.quaternion_matrix(quat)
    transform[0:3, 3] = np.array(trans, dtype=np.float64)
    return transform


def clamp_to_cone(v: np.ndarray, axis: np.ndarray, theta_max: float) -> np.ndarray:
    """
    Clamp a direction vector to remain inside a cone centered on `axis`.
    """
    axis = normalize(axis)
    d_view = normalize(v)

    dot = float(np.clip(np.dot(axis, d_view), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta <= theta_max + 1e-9:
        return d_view

    ortho = d_view - axis * dot
    ortho_n = np.linalg.norm(ortho)

    if ortho_n < 1e-9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, axis)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ortho = tmp - axis * np.dot(tmp, axis)
        ortho = normalize(ortho)
    else:
        ortho = ortho / ortho_n

    return normalize(axis * np.cos(theta_max) + ortho * np.sin(theta_max))


def look_at_quat(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a quaternion so that camera +Z aligns with `forward`
    and camera +Y aligns as closely as possible with `up`.
    """
    fwd = normalize(forward)
    up_vec = normalize(up)

    right = np.cross(up_vec, fwd)
    if np.linalg.norm(right) < 1e-9:
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(up_vec, fwd)
    right = normalize(right)
    up2 = normalize(np.cross(fwd, right))

    rot = np.eye(4, dtype=np.float64)
    rot[0:3, 0] = right
    rot[0:3, 1] = up2
    rot[0:3, 2] = fwd

    quat = tft.quaternion_from_matrix(rot)
    return quat


# ============================================================================
# Workspace constraints
# ============================================================================
class WorkspaceConstraint:
    """Reachability constraints expressed in the base frame."""

    def __init__(self, tf_listener: tf.TransformListener, config: NBVConfig):
        self.tf = tf_listener
        self.config = config
        self.last_yaw_center = 0.0

    def transform_point_to_base(self, p: np.ndarray, src_frame: str) -> Optional[np.ndarray]:
        """Transform a point into the base frame."""
        if src_frame == self.config.base_frame:
            return np.array(p, dtype=np.float64)

        trans, quat = self._lookup_tf_latest(self.config.base_frame, src_frame)
        if trans is None:
            return None

        transform = make_T(trans, quat)
        ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        return (transform @ ph)[:3]

    def transform_vec_to_base(self, v: np.ndarray, src_frame: str) -> Optional[np.ndarray]:
        """Rotate a vector into the base frame without translation."""
        if src_frame == self.config.base_frame:
            return np.array(v, dtype=np.float64)

        trans, quat = self._lookup_tf_latest(self.config.base_frame, src_frame)
        if trans is None:
            return None

        rot = tft.quaternion_matrix(quat)[0:3, 0:3]
        return rot @ np.array(v, dtype=np.float64)

    def _lookup_tf_latest(self, dst: str, src: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Lookup latest TF with one optional retry."""
        try:
            trans, quat = self.tf.lookupTransform(dst, src, rospy.Time(0))
            return trans, quat
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc_1:
            if not self.config.tf_retry_once:
                rospy.logwarn_throttle(2.0, f"[WS] TF failed {src}->{dst}: {exc_1}")
                return None, None
            try:
                self.tf.waitForTransform(
                    dst, src, rospy.Time(0), rospy.Duration(self.config.tf_timeout_sec)
                )
                trans, quat = self.tf.lookupTransform(dst, src, rospy.Time(0))
                return trans, quat
            except Exception as exc_2:
                rospy.logwarn_throttle(
                    2.0, f"[WS] TF failed {src}->{dst}: {exc_1} | retry: {exc_2}"
                )
                return None, None

    def get_camera_forward_base(self) -> np.ndarray:
        """Get current camera +Z direction expressed in the base frame."""
        trans, quat = self._lookup_tf_latest(self.config.base_frame, self.config.camera_frame)
        if trans is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        rot = tft.quaternion_matrix(quat)[0:3, 0:3]
        z_cam = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normalize(z_cam)

    def get_camera_yaw_center(self) -> float:
        """Get the preferred yaw center derived from the current camera direction."""
        trans, quat = self._lookup_tf_latest(self.config.base_frame, self.config.camera_frame)
        if trans is None:
            if self.config.cam_yaw_fallback_mode == "fixed":
                return 0.0
            return self.last_yaw_center

        rot = tft.quaternion_matrix(quat)[0:3, 0:3]
        z_cam = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        yaw_center = float(angle_wrap(np.arctan2(z_cam[1], z_cam[0]) + np.pi))
        self.last_yaw_center = yaw_center
        return yaw_center

    def is_reachable_base(self, p_base: np.ndarray, yaw_center: float) -> bool:
        """Check whether a point is inside the conservative workspace envelope."""
        radius = float(np.linalg.norm(p_base[0:2]))
        z_val = float(p_base[2])

        if not (self.config.reach_r_min <= radius <= self.config.reach_r_max):
            return False
        if not (self.config.reach_z_min <= z_val <= self.config.reach_z_max):
            return False

        yaw_p = float(np.arctan2(p_base[1], p_base[0]))
        if abs(angle_wrap(yaw_p - yaw_center)) > float(self.config.reach_yaw_max):
            return False
        return True

    def filter_candidates(
        self,
        p_view_src: np.ndarray,
        src_frame: str,
        f_payload_src: Optional[np.ndarray] = None,
    ):
        """Filter candidate viewpoints by the workspace envelope."""
        yaw_center = self.get_camera_yaw_center()

        if p_view_src is None or p_view_src.shape[0] == 0:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                None if f_payload_src is None else np.zeros((0, f_payload_src.shape[1]), np.float32),
                {"yaw_center": float(yaw_center), "kept": 0, "killed": 0},
            )

        if src_frame == self.config.base_frame:
            p_base_all = p_view_src.astype(np.float32, copy=False)
        else:
            trans, quat = self._lookup_tf_latest(self.config.base_frame, src_frame)
            if trans is None:
                killed = int(p_view_src.shape[0])
                return (
                    np.zeros((0, 3), np.float32),
                    np.zeros((0, 3), np.float32),
                    None if f_payload_src is None else np.zeros((0, f_payload_src.shape[1]), np.float32),
                    {"yaw_center": float(yaw_center), "kept": 0, "killed": killed},
                )

            transform = make_T(trans, quat)
            rot = transform[0:3, 0:3].astype(np.float64)
            trans_vec = transform[0:3, 3].astype(np.float64)
            pts = p_view_src.astype(np.float64)
            p_base_all = (rot @ pts.T).T + trans_vec
            p_base_all = p_base_all.astype(np.float32, copy=False)

        radius = np.linalg.norm(p_base_all[:, 0:2], axis=1)
        z_val = p_base_all[:, 2]
        yaw_p = np.arctan2(p_base_all[:, 1], p_base_all[:, 0])
        yaw_ok = np.abs((yaw_p - yaw_center + np.pi) % (2 * np.pi) - np.pi) <= float(
            self.config.reach_yaw_max
        )

        mask = (
            (radius >= float(self.config.reach_r_min))
            & (radius <= float(self.config.reach_r_max))
            & (z_val >= float(self.config.reach_z_min))
            & (z_val <= float(self.config.reach_z_max))
            & yaw_ok
        )

        kept_src = p_view_src[mask].astype(np.float32, copy=False)
        kept_base = p_base_all[mask].astype(np.float32, copy=False)

        if f_payload_src is not None:
            kept_payload = f_payload_src[mask].astype(np.float32, copy=False)
        else:
            kept_payload = None

        killed = int(p_view_src.shape[0] - kept_src.shape[0])
        stats = {
            "yaw_center": float(yaw_center),
            "kept": int(kept_src.shape[0]),
            "killed": killed,
        }
        return kept_src, kept_base, kept_payload, stats

    def transform_points_to_base(self, pts: np.ndarray, src_frame: str) -> Optional[np.ndarray]:
        """Transform a batch of points into the base frame."""
        if pts is None or pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if src_frame == self.config.base_frame:
            return pts.astype(np.float32, copy=False)

        trans, quat = self._lookup_tf_latest(self.config.base_frame, src_frame)
        if trans is None:
            return None
        transform = make_T(trans, quat)
        rot = transform[0:3, 0:3]
        trans_vec = transform[0:3, 3]
        out = (rot @ pts.T).T + trans_vec
        return out.astype(np.float32, copy=False)

    def rotate_vecs_to_base(self, vecs: np.ndarray, src_frame: str) -> Optional[np.ndarray]:
        """Rotate a batch of vectors into the base frame."""
        if vecs is None or vecs.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if src_frame == self.config.base_frame:
            return vecs.astype(np.float32, copy=False)

        trans, quat = self._lookup_tf_latest(self.config.base_frame, src_frame)
        if trans is None:
            return None
        rot = tft.quaternion_matrix(quat)[0:3, 0:3]
        out = (rot @ vecs.T).T
        return out.astype(np.float32, copy=False)


# ============================================================================
# Temporal validity filter for raycast decisions
# ============================================================================
class TemporalValidityFilter:
    """Temporal vote accumulator for candidate validity decisions."""

    def __init__(
        self,
        score_max=5,
        score_min=-5,
        inc=1,
        dec=1,
        th_valid=3,
        th_invalid=-2,
        ttl_sec=5.0,
    ):
        self.scores = {}
        self.last_seen = {}

        self.score_max = score_max
        self.score_min = score_min
        self.inc = inc
        self.dec = dec
        self.th_valid = th_valid
        self.th_invalid = th_invalid

        self.ttl_sec = ttl_sec
        self._last_gc = rospy.Time.now().to_sec()

    def update(self, key, is_valid):
        """Update score for one key based on the current observation."""
        now = rospy.Time.now().to_sec()
        self.last_seen[key] = now

        score = self.scores.get(key, 0)
        score = score + self.inc if is_valid else score - self.dec
        score = max(self.score_min, min(self.score_max, score))
        self.scores[key] = score

        if now - self._last_gc > 1.0:
            self._last_gc = now
            dead_keys = [k for k, t in self.last_seen.items() if now - t > self.ttl_sec]
            for key_dead in dead_keys:
                self.last_seen.pop(key_dead, None)
                self.scores.pop(key_dead, None)

        return score

    def decision(self, key):
        """Return temporal decision: True / False / None."""
        score = self.scores.get(key, 0)
        if score >= self.th_valid:
            return True
        if score <= self.th_invalid:
            return False
        return None


class SilhouetteTemporalStabilizer:
    """
    Temporal stabilizer for silhouette anchors.

    Key design:
      - key = (frame_tag, qx, qy, qz)
      - frame_tag uses crc32(frame_id) for stable reproducibility
      - TTL-based garbage collection
      - unseen-key decay is rate-limited
    """

    def __init__(
        self,
        score_max=5,
        score_min=-5,
        inc=1,
        dec=1,
        th_valid=2,
        th_invalid=-2,
        ttl_sec=3.0,
        key_res=0.03,
    ):
        self.scores = {}
        self.last_seen = {}

        self.score_max = int(score_max)
        self.score_min = int(score_min)
        self.inc = int(inc)
        self.dec = int(dec)
        self.th_valid = int(th_valid)
        self.th_invalid = int(th_invalid)

        self.ttl_sec = float(ttl_sec)
        self.key_res = float(key_res)

        self._last_gc = rospy.Time.now().to_sec()
        self._last_decay = rospy.Time.now().to_sec()

    def _frame_tag(self, frame_id: str) -> int:
        """Stable frame tag derived from frame_id."""
        return int(zlib.crc32(frame_id.encode("utf-8")) & 0xFFFF)

    def _quant(self, value: float) -> int:
        """Quantize a coordinate to the stabilizer grid."""
        return int(np.floor(value / self.key_res + 0.5))

    def _make_key(self, p: np.ndarray, frame_id: str):
        """Build stabilizer key for one anchor point."""
        fid = self._frame_tag(frame_id)
        return (
            fid,
            self._quant(float(p[0])),
            self._quant(float(p[1])),
            self._quant(float(p[2])),
        )

    def filter(self, Silhouette_pts: np.ndarray, frame_id: str) -> np.ndarray:
        """
        Filter silhouette anchors using temporal voting.

        Returns
        -------
        np.ndarray
            Stable silhouette anchors.
        """
        if len(self.scores) > 5000:
            rospy.logwarn("[SilhouetteTV] too many keys, reset stabilizer")
            self.scores.clear()
            self.last_seen.clear()

        if Silhouette_pts is None or Silhouette_pts.shape[0] == 0:
            self._gc()
            return np.zeros((0, 3), dtype=np.float32)

        now = rospy.Time.now().to_sec()

        seen_keys = set()
        for p in Silhouette_pts:
            key = self._make_key(p, frame_id)
            seen_keys.add(key)

            score = self.scores.get(key, 0) + self.inc
            score = max(self.score_min, min(self.score_max, score))
            self.scores[key] = score
            self.last_seen[key] = now

        if now - self._last_decay > 1.0:
            self._last_decay = now
            decay_after = 0.5
            for key, t_val in list(self.last_seen.items()):
                if key in seen_keys:
                    continue
                if now - t_val < decay_after:
                    continue
                score = self.scores.get(key, 0) - self.dec
                score = max(self.score_min, min(self.score_max, score))
                self.scores[key] = score

        self._gc(now)

        mask = []
        for p in Silhouette_pts:
            key = self._make_key(p, frame_id)
            mask.append(self.scores.get(key, 0) >= self.th_valid)

        mask = np.asarray(mask, dtype=bool)
        stable_pts = Silhouette_pts[mask]

        if stable_pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return stable_pts.astype(np.float32, copy=False)

    def _gc(self, now=None):
        """Remove expired stabilizer keys."""
        if now is None:
            now = rospy.Time.now().to_sec()

        if now - self._last_gc < 1.0:
            return
        self._last_gc = now

        dead = [k for k, t_val in self.last_seen.items() if now - t_val > self.ttl_sec]
        for key in dead:
            self.last_seen.pop(key, None)
            self.scores.pop(key, None)


# ============================================================================
# Frame transform cache
# ============================================================================
class FrameTransformerCache:
    """Cache helper for frame-to-frame rotation/translation lookup."""

    def __init__(self, tf_listener: tf.TransformListener, config: NBVConfig):
        self.tf = tf_listener
        self.config = config

    def lookup_Rt(self, dst: str, src: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Lookup rotation and translation from src to dst."""
        if dst == src:
            rot = np.eye(3, dtype=np.float64)
            trans = np.zeros((3,), dtype=np.float64)
            return rot, trans
        try:
            trans, quat = self.tf.lookupTransform(dst, src, rospy.Time(0))
        except Exception:
            try:
                self.tf.waitForTransform(
                    dst, src, rospy.Time(0), rospy.Duration(self.config.tf_timeout_sec)
                )
                trans, quat = self.tf.lookupTransform(dst, src, rospy.Time(0))
            except Exception:
                return None

        transform = make_T(trans, quat)
        rot = transform[0:3, 0:3].astype(np.float64, copy=True)
        trans_vec = transform[0:3, 3].astype(np.float64, copy=True)
        return rot, trans_vec

    @staticmethod
    def apply_Rt(pts: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to a batch of points."""
        pts = np.asarray(pts, dtype=np.float64)
        out = (rot @ pts.T).T + trans
        return out.astype(np.float32, copy=False)


# ============================================================================
# Occlusion model
# ============================================================================
class OcclusionModel:
    """KDTree-based occupancy model with optional inflation."""

    def __init__(self, config: NBVConfig):
        self.config = config
        self.occ_tree: Optional[cKDTree] = None
        self.occ_frame: Optional[str] = None

        self._last_tree_t = 0.0
        self._tree_min_dt = 0.3

    def update(self, pts_xyz: np.ndarray, frame_id: str):
        """Update the internal occupancy KDTree."""
        now = rospy.Time.now().to_sec()
        if now - self._last_tree_t < self._tree_min_dt:
            return
        self._last_tree_t = now

        if pts_xyz is None or pts_xyz.shape[0] < 50:
            return

        pts = np.asarray(pts_xyz, dtype=np.float32)

        if getattr(self.config, "occ_downsample_res", 0.0) and self.config.occ_downsample_res > 1e-6:
            pts = self._voxel_downsample(pts, self.config.occ_downsample_res)

        if getattr(self.config, "enable_occ_inflation", False):
            pts = self._inflate_occupied_fast(
                pts,
                res=self.config.occ_voxel_res,
                radius=self.config.occ_inflation_radius,
                cap=self.config.occ_inflation_cap_points,
            )

        self.occ_tree = cKDTree(pts)
        self.occ_frame = frame_id

    def ready(self) -> bool:
        """Return whether the occupancy model is ready."""
        return (self.occ_tree is not None) and (self.occ_frame is not None)

    def is_clear_point(self, p_occ: np.ndarray, clearance: float) -> bool:
        """Check whether a point is clear from occupied voxels."""
        if not self.ready():
            return True
        dist_nn, _ = self.occ_tree.query(np.asarray(p_occ, dtype=np.float64), k=1)
        return float(dist_nn) > float(clearance)

    def is_blocked(self, p0_occ: np.ndarray, p1_occ: np.ndarray, cap_len: Optional[float] = None) -> bool:
        """
        Test whether line-of-sight between two points is blocked.

        Parameters
        ----------
        p0_occ : np.ndarray
            Ray start point in occupancy frame.
        p1_occ : np.ndarray
            Ray end point in occupancy frame.
        cap_len : float or None
            Optional cap on traversal length.
        """
        if not self.ready():
            return False

        p0 = np.asarray(p0_occ, dtype=np.float64).reshape(3)
        p1 = np.asarray(p1_occ, dtype=np.float64).reshape(3)

        vec = p1 - p0
        dist = float(np.linalg.norm(vec))
        if dist < 1e-3:
            return False

        ray_dir = vec / (dist + 1e-9)

        start = min(float(self.config.ray_start_margin), dist * 0.4)
        end = max(0.0, dist - float(self.config.ray_end_margin))
        if end <= start + 1e-3:
            return False

        travel = end - start
        if cap_len is not None:
            travel = min(float(travel), float(cap_len))

        step = float(self.config.ray_step)
        steps = max(1, int(travel / step))
        steps = min(steps, int(getattr(self.config, "max_steps_cap", 150)))

        t_vals = start + step * np.arange(steps, dtype=np.float64)
        ray_points = p0[None, :] + t_vals[:, None] * ray_dir[None, :]

        eff_r = float(self.config.occ_radius + self.config.thickness)
        dist_nn, _ = self.occ_tree.query(ray_points, k=1)
        return bool(np.any(dist_nn < eff_r))

    def _voxel_downsample(self, pts: np.ndarray, res: float) -> np.ndarray:
        """Voxel downsample a point set by integer quantization."""
        if pts is None or pts.shape[0] == 0:
            return pts
        res = float(res)
        quant = np.floor(pts / res + 0.5).astype(np.int32)
        uniq = np.unique(quant, axis=0)
        return uniq.astype(np.float32) * res

    def _inflate_occupied_fast(self, pts: np.ndarray, res: float, radius: float, cap: int) -> np.ndarray:
        """Inflate occupied voxels inside a spherical offset set."""
        if pts is None or pts.shape[0] == 0:
            return pts
        res = float(res)
        radius = float(radius)
        r_vox = int(np.ceil(radius / res))
        if r_vox <= 0:
            return pts

        quant = np.floor(pts / res + 0.5).astype(np.int32)
        quant = np.unique(quant, axis=0)

        rr2 = r_vox * r_vox
        offsets = []
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dz in range(-r_vox, r_vox + 1):
                    if dx * dx + dy * dy + dz * dz <= rr2:
                        offsets.append((dx, dy, dz))
        offsets = np.asarray(offsets, dtype=np.int32)

        out_list = []
        chunk = 40000
        for start in range(0, quant.shape[0], chunk):
            qq = quant[start:start + chunk]
            inflated = (qq[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            out_list.append(inflated)
            if sum(arr.shape[0] for arr in out_list) > cap * 2:
                break

        out = np.vstack(out_list)
        out = np.unique(out, axis=0)
        if out.shape[0] > cap:
            rospy.logwarn_throttle(
                2.0,
                f"[OccInflate] too many voxels {out.shape[0]} > cap {cap}, shrink radius",
            )
            return quant.astype(np.float32) * res

        return out.astype(np.float32) * res


# ============================================================================
# Raycast gate
# ============================================================================
class RaycastGate:
    """Single-ray visibility gate with temporal voting."""

    def __init__(self, config: NBVConfig, occ: OcclusionModel):
        self.config = config
        self.occ = occ
        self.tv = TemporalValidityFilter(
            score_max=config.tv_score_max,
            score_min=config.tv_score_min,
            inc=config.tv_inc,
            dec=config.tv_dec,
            th_valid=config.tv_th_valid,
            th_invalid=config.tv_th_invalid,
            ttl_sec=config.tv_ttl_sec,
        )

    def _quant(self, value: float, res: float) -> int:
        """Quantize scalar value with given resolution."""
        return int(np.floor(float(value) / float(res) + 0.5))

    def _make_key(self, anchor_occ: np.ndarray, view_occ: Optional[np.ndarray]) -> Tuple[int, ...]:
        """Build temporal voting key."""
        res_anchor = float(self.config.candidate_key_res_anchor)
        ax, ay, az = [self._quant(v, res_anchor) for v in anchor_occ]

        if getattr(self.config, "tv_key_use_view", True) is False or view_occ is None:
            return ax, ay, az

        res_view = float(self.config.candidate_key_res_view)
        vx, vy, vz = [self._quant(v, res_view) for v in view_occ]
        return ax, ay, az, vx, vy, vz

    def filter(
        self,
        p_view_occ: np.ndarray,
        p_view_base: np.ndarray,
        target_occ: np.ndarray,
        anchor_occ: np.ndarray,
    ):
        """
        Filter candidate viewpoints by single-ray visibility gating.
        """
        if p_view_occ is None or p_view_occ.shape[0] == 0:
            z3 = np.zeros((0, 3), np.float32)
            return z3, z3, np.zeros((0,), dtype=bool), {"valid": 0, "invalid": 0}

        eff_clear = float(self.config.occ_radius + self.config.thickness)
        n_views = int(p_view_occ.shape[0])

        valid_mask = np.ones((n_views,), dtype=bool)
        per_view_target = target_occ.ndim == 2

        for i in range(n_views):
            p0 = p_view_occ[i]
            key = self._make_key(anchor_occ[i], p0)

            if not self.occ.is_clear_point(p0, clearance=eff_clear):
                raw_valid = False
            else:
                tgt = target_occ[i] if per_view_target else target_occ
                raw_valid = not self.occ.is_blocked(
                    p0, tgt, cap_len=float(self.config.max_ray_len)
                )

            self.tv.update(key, raw_valid)
            decision = self.tv.decision(key)
            if decision is None:
                decision = raw_valid

            valid_mask[i] = bool(decision)

        p_valid = p_view_base[valid_mask].astype(np.float32, copy=False)
        p_invalid = p_view_base[~valid_mask].astype(np.float32, copy=False)

        return p_valid, p_invalid, valid_mask, {
            "valid": int(p_valid.shape[0]),
            "invalid": int(p_invalid.shape[0]),
        }


# ============================================================================
# PSC scorer
# ============================================================================
class PSCScorer:
    """Predicted Surface Coverage scorer."""

    def __init__(self, phi_bins=16, theta_bins=32):
        self.phi_bins = int(phi_bins)
        self.theta_bins = int(theta_bins)
        self.covered = np.zeros((self.phi_bins, self.theta_bins), dtype=bool)

        patch_dirs = []
        for i in range(self.phi_bins):
            for j in range(self.theta_bins):
                phi = (i + 0.5) / self.phi_bins * np.pi
                theta = (j + 0.5) / self.theta_bins * 2 * np.pi
                patch_dirs.append(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ]
                )
        self.patch_dirs = np.asarray(patch_dirs, dtype=np.float32)

    def clear_coverage(self):
        """Reset current coverage state."""
        self.covered[:] = False

    def update_from_observed_points(self, xyz: np.ndarray, center: np.ndarray):
        """Mark observed spherical bins as covered."""
        if xyz is None or xyz.shape[0] < 10:
            return

        center = np.asarray(center, dtype=np.float64).reshape(3)
        vecs = np.asarray(xyz, dtype=np.float64) - center[None, :]
        radius = np.linalg.norm(vecs, axis=1)
        valid = radius > 1e-6
        vecs = vecs[valid] / radius[valid, None]

        phi = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
        theta = np.arctan2(vecs[:, 1], vecs[:, 0])
        theta[theta < 0] += 2 * np.pi

        phi_idx = np.clip((phi / np.pi * self.phi_bins).astype(int), 0, self.phi_bins - 1)
        theta_idx = np.clip(
            (theta / (2 * np.pi) * self.theta_bins).astype(int), 0, self.theta_bins - 1
        )

        self.covered[phi_idx, theta_idx] = True

    def score_views(
        self,
        p_views: np.ndarray,
        d_views: np.ndarray,
        center: np.ndarray,
        radius: float,
        occ: OcclusionModel,
        base_to_occ_T: Tuple[np.ndarray, np.ndarray],
        fov_deg: float = 45.0,
        yaw_margin_deg: float = 0.0,
        pitch_margin_deg: float = 0.0,
        num_patch_samples: int = 160,
        seed: int = 0,
        align_power: float = 4.0,
    ) -> np.ndarray:
        """
        Multi-ray PSC scoring. Retained for completeness.
        """
        n_views = int(p_views.shape[0])
        gains = np.zeros((n_views,), dtype=np.float32)
        if n_views == 0:
            return gains
        if (not occ.ready()) or (base_to_occ_T is None):
            return gains

        center = np.asarray(center, dtype=np.float64).reshape(3)
        p_views = np.asarray(p_views, dtype=np.float64)
        d_views = np.asarray(d_views, dtype=np.float64)

        extra = float(max(yaw_margin_deg, pitch_margin_deg))
        fov_cos = float(np.cos(np.deg2rad((fov_deg * 0.5) + extra)))

        uncovered_mask = ~self.covered.reshape(-1)
        dirs_all = self.patch_dirs[uncovered_mask]
        n_patches = int(dirs_all.shape[0])
        if n_patches == 0:
            return gains

        m_samples = int(min(max(1, num_patch_samples), n_patches))
        rng = np.random.default_rng(int(seed))
        if m_samples < n_patches:
            idx = rng.choice(n_patches, size=m_samples, replace=False)
            dirs = dirs_all[idx].astype(np.float64)
        else:
            dirs = dirs_all.astype(np.float64)

        p_patch_base = center[None, :] + float(radius) * dirs
        n_patch = dirs

        rot_bo, trans_bo = base_to_occ_T
        rot_bo = np.asarray(rot_bo, dtype=np.float64).reshape(3, 3)
        trans_bo = np.asarray(trans_bo, dtype=np.float64).reshape(3)
        p_patch_occ = (rot_bo @ p_patch_base.T).T + trans_bo

        for i in range(n_views):
            p_view = p_views[i]
            d_view = d_views[i]
            dn = np.linalg.norm(d_view)
            if dn < 1e-9:
                continue
            d_view = d_view / dn

            vec = p_patch_base - p_view[None, :]
            dist = np.linalg.norm(vec, axis=1)
            valid = dist > 1e-4
            if not np.any(valid):
                continue

            v_unit = vec[valid] / dist[valid, None]
            pp_occ = p_patch_occ[valid]
            n_val = n_patch[valid]

            front = np.einsum("ij,ij->i", n_val, v_unit) < 0.0
            if not np.any(front):
                continue
            v_unit = v_unit[front]
            pp_occ = pp_occ[front]

            align = np.einsum("ij,j->i", v_unit, d_view)
            in_fov = align > fov_cos
            if not np.any(in_fov):
                continue
            v_unit = v_unit[in_fov]
            pp_occ = pp_occ[in_fov]
            align = align[in_fov]

            p_view_occ = (rot_bo @ p_view.reshape(3, 1)).reshape(3) + trans_bo
            weights = np.clip(align, 0.0, 1.0) ** float(align_power)

            vis_sum = 0.0
            w_sum = 0.0

            for p_patch_o, weight in zip(pp_occ, weights):
                weight = float(weight)
                w_sum += weight
                if occ.is_blocked(p_view_occ, p_patch_o, cap_len=None):
                    continue
                vis_sum += weight

            uncovered_ratio = float(m_samples) / float(max(1, n_patches))
            gains[i] = float(uncovered_ratio * (vis_sum / max(w_sum, 1e-9)))

        return gains.astype(np.float32, copy=False)

    def score_views_zbuffer(
        self,
        p_views,
        q_views,
        center,
        completed_xyz,
        img_w=160,
        img_h=120,
        fov_deg=45.0,
        up_base=None,
        radius=None,
        radius_tol=0.015,
        voxel_res=0.004,
    ):
        """
        Z-buffer-based PSC scorer.
        """
        n_views = int(p_views.shape[0])
        gains = np.full((n_views,), -1.0, dtype=np.float32)

        if n_views == 0:
            return gains
        if completed_xyz is None or completed_xyz.shape[0] < 50:
            return gains

        center = np.asarray(center, dtype=np.float64).reshape(3)
        pts = np.asarray(completed_xyz, dtype=np.float64)

        if radius is not None and radius > 1e-6:
            rr = np.linalg.norm(pts - center[None, :], axis=1)
            mask = np.abs(rr - float(radius)) < float(radius_tol)
            pts = pts[mask]
            if pts.shape[0] < 50:
                return gains

        if voxel_res is not None and voxel_res > 1e-6:
            quant = np.floor(pts / voxel_res + 0.5).astype(np.int32)
            quant = np.unique(quant, axis=0)
            pts = quant.astype(np.float64) * float(voxel_res)

        width, height = int(img_w), int(img_h)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        focal = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fy = float(focal)

        uncovered = (~self.covered).reshape(-1)
        total_uncovered = int(np.count_nonzero(uncovered))
        if total_uncovered == 0:
            return gains

        def quat_to_R(quat):
            x, y, z, w = [float(v) for v in quat]
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            return np.array(
                [
                    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
                ],
                dtype=np.float64,
            )

        for i in range(n_views):
            trans = np.asarray(p_views[i], dtype=np.float64).reshape(3)
            rot_wc = quat_to_R(q_views[i])
            rot_cw = rot_wc.T

            pc = (rot_cw @ (pts - trans[None, :]).T).T
            z_val = pc[:, 2]
            front = z_val > 1e-4
            if not np.any(front):
                continue
            pc = pc[front]
            z_val = z_val[front]
            pw = pts[front]

            u = fx * (pc[:, 0] / z_val) + cx
            v = fy * (pc[:, 1] / z_val) + cy
            ui = u.astype(np.int32)
            vi = v.astype(np.int32)

            in_img = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            if not np.any(in_img):
                continue

            ui = ui[in_img]
            vi = vi[in_img]
            z_val = z_val[in_img]
            pw = pw[in_img]

            pix = vi * width + ui

            order = np.lexsort((z_val, pix))
            pix_s = pix[order]
            pw_s = pw[order]

            _, first_idx = np.unique(pix_s, return_index=True)
            vis_pts = pw_s[first_idx]

            if vis_pts.shape[0] < 10:
                continue

            dirs = vis_pts - center[None, :]
            dn = np.linalg.norm(dirs, axis=1)
            valid = dn > 1e-6
            dirs = dirs[valid] / dn[valid, None]
            if dirs.shape[0] == 0:
                continue

            phi = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
            theta = np.arctan2(dirs[:, 1], dirs[:, 0])
            theta[theta < 0] += 2 * np.pi

            phi_idx = np.clip(
                (phi / np.pi * self.phi_bins).astype(np.int32), 0, self.phi_bins - 1
            )
            th_idx = np.clip(
                (theta / (2 * np.pi) * self.theta_bins).astype(np.int32),
                0,
                self.theta_bins - 1,
            )

            bins = phi_idx * self.theta_bins + th_idx
            bins = np.unique(bins)

            newly = bins[uncovered[bins]]
            gain = float(newly.size) / float(total_uncovered)
            gains[i] = gain

        return gains


# ============================================================================
# Orientation preference
# ============================================================================
class ViewOrientationPreference:
    """Resolve final viewing direction inside a silhouette-centered cone."""

    def __init__(self, config: NBVConfig):
        self.config = config
        self.theta_max = config.theta_max
        self.wf = config.w_Silhouette
        self.wa = config.w_arm
        self.up = config.up_vector_base

    def resolve(
        self,
        d_Silhouette_base: np.ndarray,
        d_arm_base: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resolve final viewing directions and quaternions.

        Returns
        -------
        d_view_base : np.ndarray
            Final camera forward vectors in base frame.
        q_view_base : np.ndarray
            Corresponding quaternions.
        cost : np.ndarray
            Alignment cost with current arm direction.
        """
        if d_Silhouette_base is None or d_Silhouette_base.shape[0] == 0:
            return (
                np.zeros((0, 3), np.float64),
                np.zeros((0, 4), np.float64),
                np.zeros((0,), np.float64),
            )

        d_arm = normalize(np.array(d_arm_base, dtype=np.float64))

        d_view_list = []
        q_view_base = []
        cost = []

        for vf in d_Silhouette_base:
            vf = normalize(np.array(vf, dtype=np.float64))

            d_view = normalize(self.wf * vf + self.wa * d_arm)
            d_view = clamp_to_cone(d_view, vf, self.theta_max)

            angle_cost = float(np.arccos(np.clip(np.dot(d_view, d_arm), -1.0, 1.0)))
            quat = look_at_quat(forward=d_view, up=self.up)

            d_view_list.append(d_view)
            q_view_base.append(quat)
            cost.append(angle_cost)

        return (
            np.vstack(d_view_list),
            np.vstack(q_view_base),
            np.array(cost, dtype=np.float64),
        )


# ============================================================================
# Dynamic view distance optimizer
# ============================================================================
class DynamicViewDistance:
    """Optimize viewpoint position along the silhouette axis."""

    def __init__(self, config: NBVConfig):
        self.config = config
        self.d_mid = 0.5 * (config.d_min + config.d_max)

    def _reach_ratio_cost(self, p_base: np.ndarray) -> float:
        """Penalize positions close to workspace boundary."""
        radius = np.linalg.norm(p_base[0:2])
        if radius < self.config.reach_r_min:
            return 2.0
        ratio = (radius - self.config.reach_r_min) / (
            self.config.reach_r_max - self.config.reach_r_min
        )
        return ratio * ratio

    def _yaw_cost(self, p_base: np.ndarray, yaw_center: float) -> float:
        """Penalize yaw deviation from preferred yaw center."""
        yaw_p = np.arctan2(p_base[1], p_base[0])
        yaw_diff = abs(angle_wrap(yaw_p - yaw_center))
        return yaw_diff / max(self.config.reach_yaw_max, 1e-6)

    def _arm_alignment_cost(self, d_view: np.ndarray, d_arm: np.ndarray) -> float:
        """Penalize misalignment with the current arm/camera direction."""
        d_view = normalize(np.array(d_view, dtype=np.float64))
        d_arm = normalize(np.array(d_arm, dtype=np.float64))
        dot = float(np.clip(np.dot(d_view, d_arm), -1.0, 1.0))
        return (1.0 - dot) / 2.0

    def _distance_cost(self, dist: float) -> float:
        """Penalize deviation from the middle distance."""
        return abs(dist - self.d_mid) / max(self.d_mid, 1e-6)

    def _direction_consistency_cost(self, v_geom: np.ndarray, d_view_base: np.ndarray) -> float:
        """Penalize mismatch between geometric and final viewing directions."""
        dot = np.clip(np.dot(v_geom, d_view_base), -1.0, 1.0)
        angle_diff = np.arccos(dot)
        return angle_diff / np.pi


# ============================================================================
# Main analyzer
# ============================================================================
class SilhouetteNBVAnalyzer:
    """Main Layer-1 silhouette-based geometric NBV analyzer."""

    def __init__(self):
        rospy.init_node("Silhouette_nbv_analyzer")
        self._base_to_occ_T = None

        self.config = NBVConfig()
        self.tf = tf.TransformListener()

        self.workspace = WorkspaceConstraint(self.tf, self.config)
        self.orient_pref = ViewOrientationPreference(self.config)
        self.view_dist = DynamicViewDistance(self.config)

        self.tf_cache = FrameTransformerCache(self.tf, self.config)
        self.occ = OcclusionModel(self.config)
        self.gate = RaycastGate(self.config, self.occ)
        self.psc = PSCScorer(phi_bins=16, theta_bins=32)

        self._apple_surface_completed_pts_base = None

        self.Silhouette_stabilizer = SilhouetteTemporalStabilizer(
            score_max=5,
            score_min=-5,
            inc=1,
            dec=1,
            th_valid=2,
            th_invalid=-2,
            ttl_sec=2.0,
            key_res=0.02,
        )

        self._Silhouette_center_ema = None
        self._center_alpha = 0.2

        self._nbv_meta = []

        self._apple_center = None
        self._apple_center_time = rospy.Time(0)
        self._apple_radius = 0.04
        self._apple_surface_pts_base = None

        self._setup_subscribers()
        self._setup_publishers()

        rospy.logwarn("=== Silhouette NBV Analyzer started ===")

        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() and (rospy.Time.now().to_sec() - t0) < 2.0:
            try:
                self.tf.lookupTransform(
                    self.config.base_frame,
                    "in_hand_camera_color_optical_frame",
                    rospy.Time(0),
                )
                break
            except Exception:
                rospy.sleep(0.05)

        self._Silhouette_lock = threading.Lock()
        self._latest_Silhouette = None
        self._latest_Silhouette_frame = None
        self._processing = False

        self._timer = rospy.Timer(rospy.Duration(0.2), self._timer_cb)

    def Silhouette_cb(self, msg):
        """Cache the latest silhouette cloud for timer-driven processing."""
        if not self.occ.ready():
            rospy.logwarn_throttle(3.0, "Waiting for occupied point cloud KDTree...")
            return
        pts = pc2_to_xyz(msg)
        if pts.shape[0] < 10:
            return
        with self._Silhouette_lock:
            self._latest_Silhouette = pts
            self._latest_Silhouette_frame = msg.header.frame_id

    def _timer_cb(self, _evt):
        """Timer callback to process the latest cached silhouette cloud."""
        if self._processing:
            return
        with self._Silhouette_lock:
            if self._latest_Silhouette is None:
                return
            pts = self._latest_Silhouette
            frame = self._latest_Silhouette_frame
            self._latest_Silhouette = None
            self._latest_Silhouette_frame = None

        self._processing = True
        try:
            self._process_Silhouette(pts, frame)
        finally:
            self._processing = False

    def _setup_subscribers(self):
        """Create ROS subscribers."""
        rospy.Subscriber(self.config.topic_occ, PointCloud2, self.occ_cb, queue_size=1)
        rospy.Subscriber(self.config.topic_front, PointCloud2, self.Silhouette_cb, queue_size=1)
        rospy.Subscriber("/apple/center", PointStamped, self.apple_center_cb, queue_size=1)
        rospy.Subscriber(
            self.config.topic_apple_surface_obs, PointCloud2, self.apple_surface_cb, queue_size=1
        )
        rospy.Subscriber(
            self.config.topic_apple_surface_completed,
            PointCloud2,
            self.apple_surface_completed_cb,
            queue_size=1,
        )

    def _setup_publishers(self):
        """Create ROS publishers."""
        self.pub_candidates = rospy.Publisher(
            self.config.pub_candidates, PointCloud2, queue_size=1, latch=True
        )
        self.pub_valid = rospy.Publisher(
            self.config.pub_valid, PointCloud2, queue_size=1, latch=True
        )
        self.pub_invalid = rospy.Publisher(
            self.config.pub_invalid, PointCloud2, queue_size=1, latch=True
        )
        self.pub_valid_poses = rospy.Publisher(
            self.config.pub_valid_poses, PoseArray, queue_size=1, latch=True
        )
        self.pub_valid_markers = rospy.Publisher(
            self.config.pub_valid_markers, MarkerArray, queue_size=1, latch=True
        )

        from std_msgs.msg import Float32MultiArray

        self.pub_nbv_meta = rospy.Publisher(
            "/nbv/valid_view_meta",
            Float32MultiArray,
            queue_size=1,
            latch=True,
        )

    def occ_cb(self, msg):
        """Update occupancy model from occupied voxel centers."""
        pts = pc2_to_xyz(msg)
        pts_base = self.workspace.transform_points_to_base(pts, msg.header.frame_id)
        if pts_base is None or pts_base.shape[0] < 50:
            return
        self.occ.update(pts_base, self.config.base_frame)

    def apple_center_cb(self, msg: PointStamped):
        """Cache apple center in the base frame."""
        p = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        src_frame = msg.header.frame_id if msg.header.frame_id else self.config.base_frame

        p_base = self.workspace.transform_point_to_base(p, src_frame)
        if p_base is None:
            rospy.logwarn_throttle(
                1.0,
                f"[AppleCenter] TF failed {src_frame}->{self.config.base_frame}",
            )
            return

        self._apple_center = p_base.astype(np.float32)
        self._apple_center_time = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def apple_surface_cb(self, msg: PointCloud2):
        """Cache observed apple surface points in the base frame."""
        pts = pc2_to_xyz(msg)
        if pts.shape[0] < 10:
            self._apple_surface_pts_base = None
            return

        src_frame = msg.header.frame_id if msg.header.frame_id else self.config.base_frame
        pts_base = self.workspace.transform_points_to_base(pts, src_frame)
        if pts_base is None or pts_base.shape[0] < 10:
            self._apple_surface_pts_base = None
            return

        self._apple_surface_pts_base = np.asarray(pts_base, dtype=np.float32)

    def apple_surface_completed_cb(self, msg: PointCloud2):
        """Cache completed apple surface points in the base frame."""
        pts = pc2_to_xyz(msg)
        if pts.shape[0] < 10:
            self._apple_surface_completed_pts_base = None
            return

        src_frame = msg.header.frame_id if msg.header.frame_id else self.config.base_frame
        pts_base = self.workspace.transform_points_to_base(pts, src_frame)
        if pts_base is None or pts_base.shape[0] < 10:
            self._apple_surface_completed_pts_base = None
            return

        self._apple_surface_completed_pts_base = np.asarray(pts_base, dtype=np.float32)

    def _process_Silhouette(self, Silhouette_pts: np.ndarray, src_frame: str):
        """Main processing pipeline for silhouette points."""
        rospy.logwarn_throttle(
            1.0,
            f"[DBG] Silhouette IN: N={Silhouette_pts.shape[0]} frame={src_frame}",
        )

        Silhouette_pts = self.Silhouette_stabilizer.filter(Silhouette_pts, src_frame)

        rospy.logwarn_throttle(
            1.0,
            f"[DBG] Silhouette AFTER-TV: N={Silhouette_pts.shape[0]} "
            f"(tracked_keys={len(self.Silhouette_stabilizer.scores)})",
        )

        if Silhouette_pts.shape[0] == 0:
            rospy.logwarn_throttle(2.0, "[NBV] no stable silhouette after temporal filter")
            return

        if Silhouette_pts.shape[0] > self.config.max_Silhouettes:
            before_cap = int(Silhouette_pts.shape[0])
            idx = np.random.choice(
                Silhouette_pts.shape[0],
                self.config.max_Silhouettes,
                replace=False,
            )
            Silhouette_pts = Silhouette_pts[idx]
            after_cap = int(Silhouette_pts.shape[0])
            rospy.logwarn_throttle(
                1.0,
                f"[DBG] Silhouette CAP: before={before_cap} "
                f"after={after_cap} max={self.config.max_Silhouettes}",
            )

        rospy.logwarn_throttle(
            1.0,
            f"[SilhouetteTV] stable={Silhouette_pts.shape[0]} "
            f"tracked={len(self.Silhouette_stabilizer.scores)}",
        )

        raw_candidates, d_Silhouette_src, f_anchor_src = self._generate_raw_candidates(
            Silhouette_pts
        )

        rospy.logwarn_throttle(
            1.0,
            f"[DBG] RAW-CAND: N={raw_candidates.shape[0]} "
            f"view_distance={self.config.view_distance}",
        )

        raw_count = raw_candidates.shape[0]

        raw_base = self.workspace.transform_points_to_base(raw_candidates, src_frame)
        if raw_base is not None and raw_base.shape[0] > 0:
            self.pub_candidates.publish(xyz_to_pc2(raw_base, self.config.base_frame))
            rospy.logwarn_throttle(1.0, f"[NBV-RAW] raw_candidates={raw_base.shape[0]}")
        else:
            rospy.logwarn_throttle(1.0, "[NBV-RAW] raw_candidates TF to base failed or empty")

        if self.config.enable_workspace_filter:
            (
                p_view_src,
                p_view_base,
                d_Silhouette_src,
                f_anchor_src,
                ws_stats,
            ) = self._workspace_filter(
                raw_candidates, d_Silhouette_src, f_anchor_src, src_frame
            )
        else:
            (
                p_view_src,
                p_view_base,
                d_Silhouette_src,
                f_anchor_src,
                ws_stats,
            ) = self._skip_workspace_filter(
                raw_candidates, d_Silhouette_src, f_anchor_src, src_frame
            )

        ws_kept = ws_stats["kept"]
        ws_killed = ws_stats["killed"]

        if p_view_base.shape[0] > 0:
            self.pub_candidates.publish(xyz_to_pc2(p_view_base, self.config.base_frame))

        if p_view_src.shape[0] == 0:
            rospy.logwarn_throttle(
                self.config.log_every_sec,
                f"[NBV] raw={raw_count} | ws_keep=0 (kill 100.0%)",
            )
            return

        f_payload_src = np.hstack([d_Silhouette_src, f_anchor_src])

        t_ray0 = rospy.Time.now().to_sec()

        if self.config.enable_raycast_filter:
            (
                p_view_clear_base,
                p_view_blocked_base,
                valid_vf_src,
                valid_anchor_src,
                ray_stats,
            ) = self._raycast_filter(
                p_view_src,
                p_view_base,
                Silhouette_pts,
                d_Silhouette_src,
                f_anchor_src,
                f_payload_src,
                src_frame,
            )
        else:
            (
                p_view_clear_base,
                p_view_blocked_base,
                valid_vf_src,
                valid_anchor_src,
                ray_stats,
            ) = self._skip_raycast_filter(
                p_view_base, d_Silhouette_src, f_anchor_src
            )

        ray_valid = ray_stats["valid"]
        ray_invalid = ray_stats["invalid"]

        t_ray1 = rospy.Time.now().to_sec()
        rospy.logwarn(f"[TIME] raycast_filter = {t_ray1 - t_ray0:.3f}s")

        if p_view_clear_base.shape[0] > 0:
            if self.config.enable_view_orient_pref:
                rospy.logwarn(
                    f"[TIME] orient_block BEGIN N={p_view_clear_base.shape[0]}"
                )
                t_orient0 = rospy.Time.now().to_sec()
                self._orientation_preference_optimization(
                    p_view_clear_base,
                    valid_vf_src,
                    valid_anchor_src,
                    src_frame,
                )
                t_orient1 = rospy.Time.now().to_sec()
                rospy.logwarn(
                    f"[TIME] orient_block END dt={t_orient1 - t_orient0:.3f}s"
                )
            else:
                self._publish_geometric_nbv(
                    p_view_clear_base,
                    valid_vf_src,
                    src_frame,
                )

        self._log_statistics(raw_count, ws_kept, ws_killed, ray_valid, ray_invalid)

    def _generate_raw_candidates(
        self, Silhouette_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate raw outward-facing viewpoint candidates from silhouette points.

        The silhouette center is smoothed using EMA to stabilize candidate
        geometry across frames.
        """
        center_now = np.mean(Silhouette_pts, axis=0)

        if self._Silhouette_center_ema is None:
            self._Silhouette_center_ema = center_now
        else:
            alpha = float(self._center_alpha)
            self._Silhouette_center_ema = (
                (1.0 - alpha) * self._Silhouette_center_ema + alpha * center_now
            )

        center = self._Silhouette_center_ema
        dirs = normalize_rows(Silhouette_pts - center)
        p_view_src = Silhouette_pts + dirs * float(self.config.view_distance)

        d_Silhouette_src = (-dirs).astype(np.float32)
        f_anchor_src = Silhouette_pts.astype(np.float32)

        return p_view_src.astype(np.float32), d_Silhouette_src, f_anchor_src

    def _workspace_filter(
        self,
        p_view_src: np.ndarray,
        d_Silhouette_src: np.ndarray,
        f_anchor_src: np.ndarray,
        src_frame: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Filter raw candidates by workspace constraints."""
        f_payload_src = np.hstack([d_Silhouette_src, f_anchor_src])

        p_view_src, p_view_base, payload_kept, ws_stats = self.workspace.filter_candidates(
            p_view_src,
            src_frame,
            f_payload_src=f_payload_src,
        )

        if payload_kept is not None and payload_kept.shape[0] > 0:
            d_Silhouette_src = payload_kept[:, 0:3]
            f_anchor_src = payload_kept[:, 3:6]
        else:
            d_Silhouette_src = np.zeros((0, 3), dtype=np.float32)
            f_anchor_src = np.zeros((0, 3), dtype=np.float32)

        return p_view_src, p_view_base, d_Silhouette_src, f_anchor_src, ws_stats

    def _skip_workspace_filter(self, p_view_src, d_Silhouette_src, f_anchor_src, src_frame):
        """Transform candidates to base frame without workspace pruning."""
        base_list = []
        src_list = []
        d_Silhouette_list = []
        f_anchor_list = []

        for p_val, vf, anchor in zip(p_view_src, d_Silhouette_src, f_anchor_src):
            p_base = self.workspace.transform_point_to_base(p_val, src_frame)
            if p_base is None:
                continue
            src_list.append(p_val)
            base_list.append(p_base)
            d_Silhouette_list.append(vf)
            f_anchor_list.append(anchor)

        p_view_src_kept = (
            np.array(src_list, dtype=np.float32)
            if src_list
            else np.zeros((0, 3), np.float32)
        )
        p_view_base = (
            np.array(base_list, dtype=np.float32)
            if base_list
            else np.zeros((0, 3), np.float32)
        )
        d_Silhouette_src = (
            np.array(d_Silhouette_list, dtype=np.float32)
            if d_Silhouette_list
            else np.zeros((0, 3), np.float32)
        )
        f_anchor_src = (
            np.array(f_anchor_list, dtype=np.float32)
            if f_anchor_list
            else np.zeros((0, 3), np.float32)
        )

        ws_stats = {
            "kept": int(p_view_base.shape[0]),
            "killed": int(p_view_src.shape[0] - p_view_base.shape[0]),
            "yaw_center": float(self.workspace.get_camera_yaw_center()),
        }

        return p_view_src_kept, p_view_base, d_Silhouette_src, f_anchor_src, ws_stats

    def _raycast_filter(
        self,
        p_view_src,
        p_view_base,
        Silhouette_pts_src,
        d_Silhouette_src,
        f_anchor_src,
        f_payload_src,
        src_frame,
    ):
        """
        Apply single-ray visibility gating.

        TF is only used to prepare coordinates in the occupancy frame.
        The actual gate is TF-free and uses OcclusionModel.is_blocked().
        """
        if (p_view_src is None) or (p_view_src.shape[0] == 0) or (not self.occ.ready()):
            z3 = np.zeros((0, 3), np.float32)
            return z3, z3, z3, z3, {"valid": 0, "invalid": 0}

        occ_frame = self.occ.occ_frame

        if occ_frame == src_frame:
            rot_so = np.eye(3, dtype=np.float64)
            trans_so = np.zeros((3,), dtype=np.float64)
        else:
            rt_so = self.tf_cache.lookup_Rt(occ_frame, src_frame)
            if rt_so is None:
                z3 = np.zeros((0, 3), np.float32)
                return z3, z3, z3, z3, {"valid": 0, "invalid": 0}
            rot_so, trans_so = rt_so

        p_view_occ = FrameTransformerCache.apply_Rt(p_view_src, rot_so, trans_so)
        anchor_occ = FrameTransformerCache.apply_Rt(f_anchor_src, rot_so, trans_so)

        rt_bo = self.tf_cache.lookup_Rt(occ_frame, self.config.base_frame)
        if rt_bo is not None:
            self._base_to_occ_T = rt_bo
        else:
            self._base_to_occ_T = None

        apple_center_occ = None
        if (self._apple_center is not None) and (self._base_to_occ_T is not None):
            rot_bo, trans_bo = self._base_to_occ_T
            center = np.asarray(self._apple_center, dtype=np.float64).reshape(3)
            apple_center_occ = (rot_bo @ center.reshape(3, 1)).reshape(3) + trans_bo

        target_occ = apple_center_occ if (apple_center_occ is not None) else anchor_occ

        p_view_clear_base, p_view_blocked_base, valid_mask, ray_stats = self.gate.filter(
            p_view_occ=p_view_occ,
            p_view_base=p_view_base,
            target_occ=target_occ,
            anchor_occ=anchor_occ,
        )

        if p_view_clear_base.shape[0] > 0:
            self.pub_valid.publish(xyz_to_pc2(p_view_clear_base, self.config.base_frame))
        if p_view_blocked_base.shape[0] > 0:
            self.pub_invalid.publish(xyz_to_pc2(p_view_blocked_base, self.config.base_frame))

        if valid_mask.shape[0] > 0 and np.any(valid_mask):
            valid_vf_src = d_Silhouette_src[valid_mask].astype(np.float32, copy=False)
            valid_anchor_src = f_anchor_src[valid_mask].astype(np.float32, copy=False)
        else:
            valid_vf_src = np.zeros((0, 3), np.float32)
            valid_anchor_src = np.zeros((0, 3), np.float32)

        return (
            p_view_clear_base,
            p_view_blocked_base,
            valid_vf_src,
            valid_anchor_src,
            ray_stats,
        )

    def _skip_raycast_filter(
        self,
        p_view_base: np.ndarray,
        d_Silhouette_src: np.ndarray,
        f_anchor_src: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Skip visibility filtering and keep all candidates."""
        self.pub_valid.publish(xyz_to_pc2(p_view_base, self.config.base_frame))
        ray_stats = {"valid": p_view_base.shape[0], "invalid": 0}
        return (
            p_view_base,
            np.zeros((0, 3), dtype=np.float32),
            d_Silhouette_src,
            f_anchor_src,
            ray_stats,
        )

    def _orientation_preference_optimization(
        self,
        p_view_clear_base: np.ndarray,
        valid_vf_src: np.ndarray,
        valid_anchor_src: np.ndarray,
        src_frame: str,
    ):
        """Resolve final orientation, dynamic distance, PSC, and publishing."""

        def _diag(msg, **kw):
            extra = " | ".join([f"{k}={v}" for k, v in kw.items()])
            rospy.logwarn_throttle(
                1.0,
                f"[NBV-DIAG] {msg}" + (f" | {extra}" if extra else ""),
            )

        if p_view_clear_base is None or p_view_clear_base.shape[0] == 0:
            _diag(
                "skip orientation: no clear views",
                n=0 if p_view_clear_base is None else int(p_view_clear_base.shape[0]),
            )
            return

        d_Silhouette_base = self.workspace.rotate_vecs_to_base(valid_vf_src, src_frame)
        anchor_base = self.workspace.transform_points_to_base(valid_anchor_src, src_frame)

        if d_Silhouette_base is None or anchor_base is None:
            _diag(
                "skip orientation: TF failed vf/anchor->base",
                src_frame=src_frame,
                base=self.config.base_frame,
            )
            return

        if d_Silhouette_base.shape[0] == 0 or anchor_base.shape[0] == 0:
            _diag(
                "skip orientation: empty vf/anchor after TF",
                n_vf=0 if d_Silhouette_base is None else int(d_Silhouette_base.shape[0]),
                n_anchor=0 if anchor_base is None else int(anchor_base.shape[0]),
            )
            return

        n_keep = int(
            min(
                p_view_clear_base.shape[0],
                d_Silhouette_base.shape[0],
                anchor_base.shape[0],
            )
        )
        if n_keep <= 0:
            _diag(
                "skip orientation: N==0 after alignment",
                n_view=int(p_view_clear_base.shape[0]),
                n_vf=int(d_Silhouette_base.shape[0]),
                n_anchor=int(anchor_base.shape[0]),
            )
            return

        p_nominal_base = p_view_clear_base[:n_keep].astype(np.float32, copy=False)
        d_Silhouette_base = d_Silhouette_base[:n_keep].astype(np.float32, copy=False)
        anchor_base = anchor_base[:n_keep].astype(np.float32, copy=False)

        d_arm_base = self.workspace.get_camera_forward_base()
        yaw_center = self.workspace.get_camera_yaw_center()

        d_view_base, q_view_base, _orient_cost = self.orient_pref.resolve(
            d_Silhouette_base=d_Silhouette_base.astype(np.float64),
            d_arm_base=d_arm_base.astype(np.float64),
        )
        d_view_base = d_view_base.astype(np.float32, copy=False)
        q_view_base = q_view_base.astype(np.float32, copy=False)

        if self.config.enable_dynamic_view_distance:
            (
                refined_pos,
                refined_dirs,
                refined_quats,
                used_dynamic,
                view_dists,
            ) = self._dynamic_view_distance_optimization(
                p_nominal_base=p_nominal_base,
                anchor_base=anchor_base,
                d_Silhouette_base=d_Silhouette_base,
                d_view_base=d_view_base,
                q_cam=q_view_base,
                yaw_center=float(yaw_center),
                d_arm_base=d_arm_base,
            )
        else:
            refined_pos = p_nominal_base
            refined_dirs = d_view_base
            refined_quats = q_view_base
            used_dynamic = np.zeros((n_keep,), dtype=bool)
            view_dists = np.linalg.norm(anchor_base - refined_pos, axis=1).astype(np.float32)

        rospy.logwarn_throttle(
            1.0,
            f"[DYN-DIST] N={len(view_dists)} "
            f"used={int(np.count_nonzero(used_dynamic))}/{len(used_dynamic)} "
            f"d[min,max,mean]=({float(np.min(view_dists)):.3f},"
            f"{float(np.max(view_dists)):.3f},"
            f"{float(np.mean(view_dists)):.3f})",
        )

        self.psc.clear_coverage()
        if (self._apple_center is not None) and (self._apple_surface_pts_base is not None):
            self.psc.update_from_observed_points(
                xyz=self._apple_surface_pts_base,
                center=self._apple_center,
            )

        psc_vals = self.compute_psc(
            refined_pos=refined_pos,
            refined_quats=refined_quats,
        )

        self._nbv_meta = [
            {
                "psc": float(psc_val),
                "view_dist": float(vd),
                "used_dynamic": bool(flag),
            }
            for psc_val, vd, flag in zip(psc_vals, view_dists, used_dynamic)
        ]

        self._publish_results(
            refined_pos,
            refined_quats,
            refined_dirs,
            used_dynamic,
        )

    def _dynamic_view_distance_optimization(
        self,
        p_nominal_base,
        anchor_base: np.ndarray,
        d_Silhouette_base: np.ndarray,
        d_view_base: np.ndarray,
        q_cam: np.ndarray,
        yaw_center: float,
        d_arm_base: np.ndarray,
    ):
        """Search along the silhouette axis for a better viewing distance."""
        if self._base_to_occ_T is None:
            used_dynamic = np.zeros((p_nominal_base.shape[0],), dtype=bool)
            view_dists = np.linalg.norm(anchor_base - p_nominal_base, axis=1).astype(
                np.float32
            )
            return p_nominal_base, d_view_base, q_cam, used_dynamic, view_dists

        rot, trans = self._base_to_occ_T

        refined_pos = []
        refined_dirs = []
        refined_quats = []
        used_dynamic = []
        view_dists = []

        d_samples = max(2, int(self.config.d_samples))
        d_grid = np.linspace(self.config.d_min, self.config.d_max, d_samples).astype(
            np.float64
        )

        eff_clear = float(self.config.occ_radius + self.config.thickness)

        n_views = anchor_base.shape[0]
        for idx in range(n_views):
            p_anchor = anchor_base[idx].astype(np.float64)
            vf = normalize(d_Silhouette_base[idx].astype(np.float64))
            v_view = normalize(d_view_base[idx].astype(np.float64))
            quat = q_cam[idx]

            d0 = float(
                np.linalg.norm(p_anchor - p_nominal_base[idx].astype(np.float64))
            )

            d_candidates = d_grid
            best_p = None

            anchor_occ = rot @ p_anchor + trans
            cand_list = []

            for dist in d_candidates:
                p = p_anchor - float(dist) * vf

                if not self.workspace.is_reachable_base(p, yaw_center):
                    continue

                p_occ = rot @ p + trans
                if not self.occ.is_clear_point(p_occ, clearance=eff_clear):
                    continue

                if self.occ.is_blocked(
                    p_occ,
                    anchor_occ,
                    cap_len=float(self.config.max_ray_len),
                ):
                    continue

                c_reach = self.view_dist._reach_ratio_cost(p)
                c_yaw = self.view_dist._yaw_cost(p, yaw_center)
                c_arm = self.view_dist._arm_alignment_cost(v_view, d_arm_base)
                c_dist = self.view_dist._distance_cost(float(dist))
                c_nom = abs(float(dist) - d0) / max(
                    (self.config.d_max - self.config.d_min), 1e-6
                )

                actual_view = normalize(p_anchor - p)
                c_dir = self.view_dist._direction_consistency_cost(actual_view, v_view)

                cost = (
                    float(self.config.w_dist_reach) * float(c_reach)
                    + float(self.config.w_dist_yaw) * float(c_yaw)
                    + float(self.config.w_dist_arm) * float(c_arm)
                    + 0.05 * float(c_dist)
                    + 0.2 * float(c_nom)
                    + 0.2 * float(c_dir)
                )

                d_norm = (float(dist) - float(self.config.d_min)) / max(
                    float(self.config.d_max - self.config.d_min), 1e-6
                )
                eps = float(getattr(self.config, "dist_tiebreak_eps", 0.0))
                mode = str(getattr(self.config, "dist_tiebreak_mode", "near"))

                if eps > 0.0:
                    if mode == "near":
                        cost += eps * d_norm
                    elif mode == "far":
                        cost += eps * (1.0 - d_norm)

                cand_list.append((float(cost), p.copy()))

            if len(cand_list) > 0:
                cand_list.sort(key=lambda x: x[0])
                _, best_p = cand_list[0]

            if best_p is None:
                best_p = p_nominal_base[idx].astype(np.float64)
                used_dynamic.append(False)
            else:
                used_dynamic.append(True)

            refined_pos.append(best_p.astype(np.float32))
            if self._apple_center is not None:
                d_real = normalize(
                    self._apple_center.astype(np.float64) - best_p.astype(np.float64)
                )
            else:
                d_real = normalize(p_anchor - best_p.astype(np.float64))
            refined_dirs.append(d_real.astype(np.float32))
            refined_quats.append(quat)
            view_dists.append(float(np.linalg.norm(p_anchor - best_p)))

        refined_pos = np.vstack(refined_pos) if len(refined_pos) else p_nominal_base
        refined_dirs = np.vstack(refined_dirs) if len(refined_dirs) else d_view_base
        refined_quats = np.vstack(refined_quats) if len(refined_quats) else q_cam

        return (
            refined_pos,
            refined_dirs,
            refined_quats,
            np.array(used_dynamic, dtype=bool),
            np.array(view_dists, dtype=np.float32),
        )

    def _publish_results(
        self,
        refined_positions: np.ndarray,
        refined_quats: np.ndarray,
        refined_dirs: np.ndarray,
        used_dynamic: np.ndarray,
    ):
        """Publish NBV meta, poses, and markers."""
        if (not isinstance(getattr(self, "_nbv_meta", None), list)) or (
            len(self._nbv_meta) != refined_positions.shape[0]
        ):
            self._nbv_meta = [
                {"psc": -1.0, "view_dist": -1.0, "used_dynamic": False}
                for _ in range(refined_positions.shape[0])
            ]

        rospy.logdebug_throttle(
            1.0,
            f"[NBV-META] publish {len(self._nbv_meta)} metas",
        )

        from std_msgs.msg import Float32MultiArray

        meta_msg = Float32MultiArray()
        for meta in self._nbv_meta:
            meta_msg.data.extend(
                [
                    float(meta.get("psc", -1.0)),
                    float(meta.get("view_dist", -1.0)),
                    1.0 if meta.get("used_dynamic", False) else 0.0,
                ]
            )
        self.pub_nbv_meta.publish(meta_msg)

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = self.config.base_frame

        for pos, quat in zip(refined_positions, refined_quats):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, pos)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(
                float, quat
            )
            pose_array.poses.append(pose)

        self.pub_valid_poses.publish(pose_array)

        marker_array = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.header.frame_id = self.config.base_frame
        clear.header.stamp = rospy.Time.now()
        clear.ns = "nbv_views"
        marker_array.markers.append(clear)

        for idx in range(refined_positions.shape[0]):
            pos = refined_positions[idx]
            direction = refined_dirs[idx]
            is_dyn = bool(used_dynamic[idx])

            marker = Marker()
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.header.frame_id = self.config.base_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "nbv_views"
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.scale.x = 0.02
            marker.scale.y = 0.04
            marker.scale.z = 0.06

            if is_dyn:
                marker.color.r, marker.color.g, marker.color.b = 0.2, 0.4, 1.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.2, 1.0, 0.2
            marker.color.a = 1.0

            p0 = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            p1 = Point(
                x=float(pos[0] + 0.15 * direction[0]),
                y=float(pos[1] + 0.15 * direction[1]),
                z=float(pos[2] + 0.15 * direction[2]),
            )
            marker.points = [p0, p1]

            marker_array.markers.append(marker)

        self.pub_valid_markers.publish(marker_array)
        rospy.logwarn_throttle(
            1.0,
            f"[PUB] poses={len(pose_array.poses)} "
            f"markers={len(marker_array.markers)} meta_len={len(meta_msg.data)}",
        )

    def _publish_geometric_nbv(
        self,
        p_view_base: np.ndarray,
        d_Silhouette_src: np.ndarray,
        src_frame: str,
    ):
        """
        Publish pure geometric NBVs without orientation preference.

        Position:
            raycast-filtered candidate positions
        Direction:
            transformed silhouette geometric directions
        """
        pos_base_list = []
        d_view_base_list = []
        q_view_list = []

        for p_base, vf in zip(p_view_base, d_Silhouette_src):
            vb = self.workspace.transform_vec_to_base(vf, src_frame)
            if vb is None or np.linalg.norm(vb) < 1e-6:
                continue

            vb = normalize(vb)
            quat = look_at_quat(forward=vb, up=self.config.up_vector_base)

            pos_base_list.append(p_base)
            d_view_base_list.append(vb)
            q_view_list.append(quat)

        if len(pos_base_list) == 0:
            rospy.logwarn_throttle(1.0, "[NBV] geometric NBV produced 0 valid views")
            return

        refined_pos = np.vstack(pos_base_list).astype(np.float32)
        refined_dirs = np.vstack(d_view_base_list)
        refined_quats = np.vstack(q_view_list)
        used_dynamic = np.zeros((refined_pos.shape[0],), dtype=bool)

        self._nbv_meta = [
            {"psc": -1.0, "view_dist": -1.0, "used_dynamic": False}
            for _ in range(refined_pos.shape[0])
        ]
        self._publish_results(refined_pos, refined_quats, refined_dirs, used_dynamic)

    def _log_statistics(
        self,
        raw_count: int,
        ws_kept: int,
        ws_killed: int,
        ray_valid: int,
        ray_invalid: int,
    ):
        """Log pipeline statistics."""
        ws_total = max(1, raw_count)
        ray_total = max(1, ray_valid + ray_invalid)

        ws_kill_ratio = ws_killed / ws_total
        ray_kill_ratio = ray_invalid / ray_total

        rospy.logwarn_throttle(
            self.config.log_every_sec,
            f"[NBV] raw={raw_count} | "
            f"ws_keep={ws_kept} (kill {ws_kill_ratio * 100:.1f}%) | "
            f"ray_valid={ray_valid} ray_invalid={ray_invalid} "
            f"(kill {ray_kill_ratio * 100:.1f}%)",
        )

    def compute_psc(
        self,
        refined_pos: np.ndarray,
        refined_quats: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PSC using the z-buffer scorer only.

        Returns
        -------
        np.ndarray
            PSC values in [0, 1], or -1 when unavailable.
        """
        n_views = int(refined_pos.shape[0])
        psc_vals = np.full((n_views,), -1.0, dtype=np.float32)
        if n_views == 0:
            return psc_vals

        if self._apple_center is None:
            rospy.logwarn_throttle(
                float(self.config.psc_log_throttle_sec),
                "[PSC-ZB] skipped: apple_center missing",
            )
            return psc_vals

        comp = self._apple_surface_completed_pts_base
        n_comp = 0 if comp is None else int(comp.shape[0])
        if comp is None or n_comp < int(self.config.psc_min_completed_points):
            rospy.logwarn_throttle(
                float(self.config.psc_log_throttle_sec),
                f"[PSC-ZB] skipped: completed surface missing or too small (N={n_comp})",
            )
            return psc_vals

        if not bool(self.config.psc_enable_zbuffer):
            rospy.logwarn_throttle(
                float(self.config.psc_log_throttle_sec),
                "[PSC-ZB] skipped: psc_enable_zbuffer=False",
            )
            return psc_vals

        img_w = int(self.config.psc_img_w)
        img_h = int(self.config.psc_img_h)
        fov = float(self.config.psc_fov_deg)

        voxel_res = float(self.config.psc_voxel_res)
        radius_tol = float(self.config.psc_radius_tol)

        if bool(self.config.psc_use_radius_band):
            radius = float(self._apple_radius)
        else:
            radius = None

        t0 = rospy.Time.now().to_sec()
        psc_vals = self.psc.score_views_zbuffer(
            p_views=refined_pos,
            q_views=refined_quats,
            center=self._apple_center,
            completed_xyz=comp,
            img_w=img_w,
            img_h=img_h,
            fov_deg=fov,
            radius=radius,
            radius_tol=radius_tol,
            voxel_res=voxel_res,
        )
        t1 = rospy.Time.now().to_sec()

        rospy.logwarn(f"[TIME] PSC(z-buffer) = {t1 - t0:.3f}s")
        rospy.logwarn_throttle(
            float(self.config.psc_log_throttle_sec),
            f"[PSC-ZB] N={n_views} mean={float(np.mean(psc_vals)):.4f} "
            f"min={float(np.min(psc_vals)):.4f} max={float(np.max(psc_vals)):.4f} "
            f"img=({img_w}x{img_h}) fov={fov:.1f} voxel={voxel_res:.4f} "
            f"radius_band={'ON' if radius is not None else 'OFF'} "
            f"tol={radius_tol:.3f} compN={n_comp}",
        )

        return psc_vals.astype(np.float32, copy=False)


# ============================================================================
# Main entry
# ============================================================================
if __name__ == "__main__":
    analyzer = SilhouetteNBVAnalyzer()
    rospy.spin()