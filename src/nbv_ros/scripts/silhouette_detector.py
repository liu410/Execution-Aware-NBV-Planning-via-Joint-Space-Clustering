#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
silhouette_detector.py

ROS1 node for extracting target-level silhouette points from an observed apple
surface using a published sphere center.

Overview
--------
This node computes a target silhouette (or occlusion silhouette) from the
segmented apple surface point cloud. Instead of re-estimating the sphere center
from the observed surface points, it directly uses the published apple center
from the reconstruction/completion pipeline.

The silhouette is defined using a sphere-based normal model:
    n(p) = normalize(p - center)
    v(p) = normalize(p - camera_position)

A surface point is considered a silhouette point if:
    |dot(n(p), v(p))| < threshold

After silhouette extraction, the points are pushed slightly outward along the
radial direction for clearer visualization in RViz.

Inputs
------
/segmented_apple_pointcloud : sensor_msgs/PointCloud2
    Segmented apple surface points.

/apple/center : geometry_msgs/PointStamped
    Published apple center from the completion pipeline.

/apple/radius : std_msgs/Float32 (optional)
    Published apple radius. If unavailable, the radius is estimated from the
    current surface point cloud and the published center.

TF
--
TF is used for:
1. Transforming the published apple center into the point-cloud frame.
2. Looking up the camera origin in the point-cloud frame.

Output
------
/nbv/Silhouette_points : sensor_msgs/PointCloud2
    Silhouette ring points, offset slightly outward for visualization.

Private parameters
------------------
~cam_frame (str)
    Camera optical frame used to compute viewing directions.
    Default: in_hand_camera_color_optical_frame

~silhouette_cos_thresh (float)
    Silhouette criterion threshold.
    A point is classified as a silhouette point if:
        |dot(n, v)| < silhouette_cos_thresh
    Default: 0.72

~silhouette_radius_offset (float)
    Additional radial offset applied to published silhouette points for
    visualization.
    Default: 0.02
"""

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header

# -------------------------------------------------------------------------
# Default parameters
# -------------------------------------------------------------------------
SILHOUETTE_COS_THRESH = 0.72
SILHOUETTE_RADIUS_OFFSET = 0.02


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def pc2_to_xyz(msg):
    """
    Convert a PointCloud2 message to an (N, 3) float32 NumPy array.
    """
    points = [
        [p[0], p[1], p[2]]
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    ]
    return np.asarray(points, dtype=np.float32)


def xyz_to_pc2(points, frame):
    """
    Convert an (N, 3) NumPy array to a PointCloud2 message.
    """
    header = Header(stamp=rospy.Time.now(), frame_id=frame)
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(header, fields, points.tolist())


def transform_point_ps(tf_buffer, point_stamped_in, target_frame, timeout=0.2):
    """
    Transform a PointStamped into the target frame.

    Parameters
    ----------
    tf_buffer : tf2_ros.Buffer
        TF2 buffer used for transform lookup.
    point_stamped_in : geometry_msgs.msg.PointStamped
        Input point.
    target_frame : str
        Target frame.
    timeout : float
        TF lookup timeout in seconds.

    Returns
    -------
    geometry_msgs.msg.PointStamped or None
        Transformed point if successful, otherwise None.
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            point_stamped_in.header.frame_id,
            point_stamped_in.header.stamp
            if point_stamped_in.header.stamp != rospy.Time()
            else rospy.Time(0),
            rospy.Duration(timeout),
        )
        point_stamped_out = tf2_geometry_msgs.do_transform_point(
            point_stamped_in, transform
        )
        return point_stamped_out
    except Exception:
        return None


def lookup_frame_origin(tf_buffer, target_frame, source_frame, timeout=0.2):
    """
    Get the origin of source_frame expressed in target_frame.

    Parameters
    ----------
    tf_buffer : tf2_ros.Buffer
        TF2 buffer used for transform lookup.
    target_frame : str
        Target frame.
    source_frame : str
        Source frame.
    timeout : float
        TF lookup timeout in seconds.

    Returns
    -------
    np.ndarray or None
        A (3,) float32 vector containing the source-frame origin in target_frame,
        or None if lookup fails.
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout),
        )
        t = transform.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float32)
    except Exception:
        return None


# -------------------------------------------------------------------------
# Main node
# -------------------------------------------------------------------------
class TargetSilhouetteUsePublishedCenter:
    """Silhouette detector that uses the published reconstructed apple center."""

    def __init__(self):
        rospy.init_node("target_silhouette_use_published_center")

        rospy.logwarn("=== Silhouette Detector (using /apple/center) started ===")

        # ---------------------------------------------------------------------
        # TF2
        # ---------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------
        self.cam_frame = rospy.get_param(
            "~cam_frame", "in_hand_camera_color_optical_frame"
        )
        self.cos_thresh = rospy.get_param(
            "~silhouette_cos_thresh", SILHOUETTE_COS_THRESH
        )
        self.radius_offset = rospy.get_param(
            "~silhouette_radius_offset", SILHOUETTE_RADIUS_OFFSET
        )

        # ---------------------------------------------------------------------
        # Cached target information
        # ---------------------------------------------------------------------
        self.latest_center_ps = None
        self.latest_radius = None
        self.last_center_time = rospy.Time(0)

        # ---------------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------------
        self.sub_pc = rospy.Subscriber(
            "/segmented_apple_pointcloud",
            PointCloud2,
            self.cb_pc,
            queue_size=1,
        )
        self.sub_center = rospy.Subscriber(
            "/apple/center",
            PointStamped,
            self.cb_center,
            queue_size=10,
        )
        self.sub_radius = rospy.Subscriber(
            "/apple/radius",
            Float32,
            self.cb_radius,
            queue_size=10,
        )

        # ---------------------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------------------
        self.pub_sil = rospy.Publisher(
            "/nbv/Silhouette_points",
            PointCloud2,
            queue_size=1,
            latch=True,
        )

        rospy.loginfo("Waiting for /apple/center ...")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def cb_center(self, msg: PointStamped):
        """
        Cache the latest published apple center.
        """
        self.latest_center_ps = msg
        self.last_center_time = rospy.Time.now()

    def cb_radius(self, msg: Float32):
        """
        Cache the latest published apple radius.
        """
        self.latest_radius = float(msg.data)

    def cb_pc(self, msg: PointCloud2):
        """
        Process the segmented apple point cloud and publish silhouette points.
        """
        pts = pc2_to_xyz(msg)
        if pts.shape[0] < 100:
            rospy.logwarn_throttle(1.0, "Not enough apple surface points.")
            return

        frame = msg.header.frame_id

        if self.latest_center_ps is None:
            rospy.logwarn_throttle(1.0, "No /apple/center received yet.")
            return

        # ---------------------------------------------------------------------
        # 1. Transform the published apple center into the point-cloud frame
        # ---------------------------------------------------------------------
        center_ps_in = self.latest_center_ps

        center_ps_out = transform_point_ps(
            self.tf_buffer, center_ps_in, frame, timeout=0.2
        )
        if center_ps_out is None:
            rospy.logwarn_throttle(
                1.0,
                f"TF failed: cannot transform center from "
                f"{center_ps_in.header.frame_id} to {frame}",
            )
            return

        center = np.array(
            [
                center_ps_out.point.x,
                center_ps_out.point.y,
                center_ps_out.point.z,
            ],
            dtype=np.float32,
        )

        # ---------------------------------------------------------------------
        # 2. Obtain radius
        #    Prefer the published radius. Otherwise estimate it from the point
        #    cloud while keeping the center fixed to the published center.
        # ---------------------------------------------------------------------
        vecs = pts - center
        dists = np.linalg.norm(vecs, axis=1) + 1e-9

        if (
            self.latest_radius is not None
            and np.isfinite(self.latest_radius)
            and self.latest_radius > 1e-6
        ):
            radius = float(self.latest_radius)
        else:
            radius = float(np.median(dists))

        # ---------------------------------------------------------------------
        # 3. Look up the camera position in the point-cloud frame
        # ---------------------------------------------------------------------
        cam_pos = lookup_frame_origin(
            self.tf_buffer, frame, self.cam_frame, timeout=0.2
        )
        if cam_pos is None:
            rospy.logwarn_throttle(
                1.0,
                f"TF failed: cannot lookup {self.cam_frame} in {frame}",
            )
            return

        # ---------------------------------------------------------------------
        # 4. Compute silhouette criterion
        #    n(p) = normalize(p - center)
        #    v(p) = normalize(p - cam_pos)
        #    silhouette if |dot(n, v)| < threshold
        # ---------------------------------------------------------------------
        view_vecs = pts - cam_pos[None, :]
        view_norm = np.linalg.norm(view_vecs, axis=1) + 1e-9
        view_unit = view_vecs / view_norm[:, None]

        normals = vecs / dists[:, None]
        cos_vals = np.sum(normals * view_unit, axis=1)

        mask = np.abs(cos_vals) < self.cos_thresh
        sil_pts = pts[mask]

        rospy.loginfo_throttle(
            1.0,
            f"[Silhouette] sil_pts={sil_pts.shape[0]}/{pts.shape[0]} "
            f"center_in={center_ps_in.header.frame_id} "
            f"center_out={frame} radius={radius:.4f}",
        )

        if sil_pts.shape[0] == 0:
            rospy.logwarn_throttle(1.0, "No silhouette points found.")
            return

        # ---------------------------------------------------------------------
        # 5. Push silhouette points slightly outward for clearer RViz display
        # ---------------------------------------------------------------------
        sil_dirs = sil_pts - center
        sil_dirs /= np.linalg.norm(sil_dirs, axis=1, keepdims=True) + 1e-9

        sil_pts_out = center + (radius + self.radius_offset) * sil_dirs

        # ---------------------------------------------------------------------
        # 6. Publish
        # ---------------------------------------------------------------------
        self.pub_sil.publish(xyz_to_pc2(sil_pts_out, frame))


if __name__ == "__main__":
    try:
        TargetSilhouetteUsePublishedCenter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass