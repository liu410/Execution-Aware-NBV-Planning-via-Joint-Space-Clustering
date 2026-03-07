#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
point_filter.py

ROS1 node for filtering environment point clouds for mapping backends such as
OctoMap.

Overview
--------
This node receives an input RGB-D point cloud, applies lightweight geometric
filtering, removes points near the target apple and near the camera, and
publishes a reduced point cloud for downstream occupancy mapping.

Design goals
------------
The implementation is explicitly optimized to reduce unnecessary computation:

1. The point-cloud subscriber only caches the latest frame.
   No heavy processing is performed inside the callback.

2. A fixed-rate ROS timer performs processing at ~process_hz.
   This avoids frame backlog and always processes the newest available cloud.

3. Early ROI cropping is applied before entering Open3D.
   The crop is centered around the current apple center and significantly
   reduces the number of points before expensive filtering.

4. Radius outlier removal can be downsampled in time.
   It may be executed every N frames instead of every frame.

5. High-frequency logs are throttled to reduce runtime overhead.

Main processing steps
---------------------
1. Cache the latest input cloud.
2. Optionally crop points around the current target center in the source frame.
3. Convert to Open3D and apply voxel downsampling and radius filtering.
4. Transform the filtered cloud into the target frame.
5. Optionally remove points inside the apple sphere.
6. Optionally remove points near the camera origin.
7. Keep only points within a bounded radius around the target.
8. Publish the final filtered point cloud.

Subscribed topics
-----------------
~input_topic (default: /in_hand_camera/depth/color/points)
    Input RGB-D point cloud.

 /apple_segmentation_mask
    Apple mask topic. Currently cached for compatibility/debugging.

 /in_hand_camera/color/camera_info
    CameraInfo topic. Currently cached for compatibility/debugging.

 /apple/center
    Target apple center used for ROI cropping and local filtering.

Published topics
----------------
~filtered_topic (default: /filtered_points)
    Filtered environment point cloud in the target frame.

Private parameters
------------------
~input_topic (str)
    Input point-cloud topic.
    Default: /in_hand_camera/depth/color/points

~filtered_topic (str)
    Output filtered point-cloud topic.
    Default: /filtered_points

~target_frame (str)
    Output frame for the published cloud.
    Default: base_link

~process_hz (float)
    Timer processing rate in Hz.
    Default: 1.0

~enable_early_roi (bool)
    Whether to crop around the apple center before Open3D filtering.
    Default: True

~roi_margin (float)
    Extra radius margin added to the keep radius for early ROI cropping.
    Default: 0.05

~tf_timeout (float)
    TF lookup timeout in seconds.
    Default: 0.05

~voxel_size (float)
    Open3D voxel downsampling size.
    Default: 0.002

~radius (float)
    Radius outlier search radius.
    Default: 0.02

~min_points (int)
    Minimum neighbor count for radius outlier filtering.
    Default: 180

~radius_filter_every_n (int)
    Apply radius outlier filtering once every N frames.
    1 means apply every frame.
    Default: 1

~apple_keep_radius (float)
    Final published cloud is restricted to this radius around the apple center.
    Default: 0.3

~apple_center_ema_alpha (float)
    EMA update factor for smoothing the apple center.
    Default: 0.2

~apple_center_timeout (float)
    Reserved timeout for future target-center validity checks.
    Default: 0.5

~apple_remove_enable (bool)
    Whether to remove points inside the apple sphere.
    Default: True

~apple_remove_radius (float)
    Radius of the sphere removed around the apple center.
    Default: 0.07

~camera_frame (str)
    Camera frame used for near-camera clearance.
    Default: in_hand_camera_link

~near_clearance_enable (bool)
    Whether to remove points near the camera origin.
    Default: True

~near_clearance_radius (float)
    Radius of the near-camera clearance region.
    Default: 0.30

~use_gpu (bool)
    Whether to use Open3D CUDA tensor mode if available.
    Default: True
"""

import struct
import threading

import cv2
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tft
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header

try:
    import ros_numpy

    ROS_NUMPY_AVAILABLE = True
except Exception:
    ROS_NUMPY_AVAILABLE = False


class IntegratedPointCloudProcessor:
    """Filter environment point clouds around the current apple target."""

    def __init__(self):
        rospy.init_node("integrated_pointcloud_processor", anonymous=True)

        # ---------------------------------------------------------------------
        # Basic configuration
        # ---------------------------------------------------------------------
        self.input_topic = rospy.get_param(
            "~input_topic", "/in_hand_camera/depth/color/points"
        )
        self.filtered_topic = rospy.get_param("~filtered_topic", "/filtered_points")
        self.target_frame = rospy.get_param("~target_frame", "base_link")

        # Fixed-rate timer processing
        self.process_hz = float(rospy.get_param("~process_hz", 1.0))
        self.enable_early_roi = bool(rospy.get_param("~enable_early_roi", True))
        self.roi_margin = float(rospy.get_param("~roi_margin", 0.05))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.05))

        # ---------------------------------------------------------------------
        # Apple center tracking
        # ---------------------------------------------------------------------
        self.apple_center_time = rospy.Time(0)
        self.apple_center_ema = None
        self.apple_center_frozen = None
        self.ema_alpha = float(rospy.get_param("~apple_center_ema_alpha", 0.2))

        self.keep_radius = float(rospy.get_param("~apple_keep_radius", 0.3))
        self.center_timeout = float(rospy.get_param("~apple_center_timeout", 0.5))

        # Apple sphere removal
        self.apple_remove_radius = float(rospy.get_param("~apple_remove_radius", 0.06))
        self.apple_remove_enable = bool(rospy.get_param("~apple_remove_enable", True))
        self.remove_apple = True

        # ---------------------------------------------------------------------
        # Point-cloud filtering parameters
        # ---------------------------------------------------------------------
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.002))
        self.radius = float(rospy.get_param("~radius", 0.02))
        self.min_points = int(rospy.get_param("~min_points", 180))

        self.radius_filter_every_n = int(rospy.get_param("~radius_filter_every_n", 1))
        self._frame_count = 0

        # ---------------------------------------------------------------------
        # Near-camera clearance parameters
        # ---------------------------------------------------------------------
        self.camera_frame = rospy.get_param("~camera_frame", "in_hand_camera_link")
        self.near_clearance_radius = float(
            rospy.get_param("~near_clearance_radius", 0.30)
        )
        self.near_clearance_enable = bool(
            rospy.get_param("~near_clearance_enable", True)
        )

        # ---------------------------------------------------------------------
        # Open3D device selection
        # ---------------------------------------------------------------------
        param_use_gpu = bool(rospy.get_param("~use_gpu", True))
        self.use_gpu = bool(
            param_use_gpu
            and hasattr(o3d, "core")
            and o3d.core.cuda.is_available()
        )

        if self.use_gpu:
            self.device = o3d.core.Device("CUDA:0")
            rospy.loginfo("CUDA is available. Using Open3D tensor backend on GPU.")
        else:
            self.device = o3d.core.Device("CPU:0")
            rospy.loginfo("Using Open3D legacy backend on CPU.")

        # ---------------------------------------------------------------------
        # TF initialization
        # ---------------------------------------------------------------------
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # ---------------------------------------------------------------------
        # ROS publishers and subscribers
        # ---------------------------------------------------------------------
        self.filtered_pub = rospy.Publisher(
            self.filtered_topic, PointCloud2, queue_size=1
        )

        self.bridge = CvBridge()
        self.latest_mask = None
        self.latest_cam_info = None

        rospy.Subscriber(
            "/apple_segmentation_mask",
            Image,
            self.mask_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/in_hand_camera/color/camera_info",
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/apple/center",
            PointStamped,
            self.apple_center_cb,
            queue_size=1,
        )

        # Cache only the latest point-cloud frame
        self.latest_cloud = None
        self.cloud_lock = threading.Lock()
        self.cloud_sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.cache_cloud_cb,
            queue_size=1,
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.process_hz, 1e-6)),
            self.timer_cb,
        )

        rospy.loginfo("=" * 60)
        rospy.loginfo("Integrated PointCloud Processor (Timer + ROI + Filters)")
        rospy.loginfo("Input topic: %s", self.input_topic)
        rospy.loginfo("Filtered topic: %s", self.filtered_topic)
        rospy.loginfo("process_hz: %s", self.process_hz)
        rospy.loginfo(
            "enable_early_roi: %s, roi_margin: %s",
            self.enable_early_roi,
            self.roi_margin,
        )
        rospy.loginfo(
            "voxel_size: %s, radius: %s, min_points: %s",
            self.voxel_size,
            self.radius,
            self.min_points,
        )
        rospy.loginfo("radius_filter_every_n: %s", self.radius_filter_every_n)
        rospy.loginfo(
            "apple_remove_enable: %s, apple_remove_radius: %s",
            self.apple_remove_enable,
            self.apple_remove_radius,
        )
        rospy.loginfo("apple_keep_radius: %s", self.keep_radius)
        rospy.loginfo("=" * 60)

    # -------------------------------------------------------------------------
    # Latest-frame caching and timer-driven processing
    # -------------------------------------------------------------------------
    def cache_cloud_cb(self, cloud_msg: PointCloud2):
        """Cache only the latest point cloud."""
        with self.cloud_lock:
            self.latest_cloud = cloud_msg

    def timer_cb(self, _event):
        """Process the latest cached cloud at a fixed timer rate."""
        with self.cloud_lock:
            cloud_msg = self.latest_cloud
            self.latest_cloud = None

        if cloud_msg is None:
            return

        self.process_one_cloud(cloud_msg)

    # -------------------------------------------------------------------------
    # Auxiliary input callbacks
    # -------------------------------------------------------------------------
    def mask_callback(self, msg: Image):
        """Cache a dilated version of the latest mask for future compatibility."""
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
            kernel = np.ones((5, 5), np.uint8)
            self.latest_mask = cv2.dilate(mask, kernel, iterations=2)
        except Exception as exc:
            rospy.logerr_throttle(1.0, f"Mask decode error: {exc}")

    def camera_info_callback(self, msg: CameraInfo):
        """Cache the latest camera info message."""
        self.latest_cam_info = msg

    # -------------------------------------------------------------------------
    # Apple center update
    # -------------------------------------------------------------------------
    def apple_center_cb(self, msg: PointStamped):
        """Update the smoothed apple center using EMA."""
        center = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)

        if self.apple_center_ema is None:
            self.apple_center_ema = center.copy()
        else:
            alpha = self.ema_alpha
            self.apple_center_ema = (1.0 - alpha) * self.apple_center_ema + alpha * center

        # Freeze the center until a new center message arrives.
        self.apple_center_frozen = self.apple_center_ema.copy()
        self.apple_center_time = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    # -------------------------------------------------------------------------
    # PointCloud2 <-> NumPy conversion
    # -------------------------------------------------------------------------
    def pointcloud2_to_array(self, cloud_msg: PointCloud2):
        """
        Convert PointCloud2 to point and color arrays.

        Returns
        -------
        points : np.ndarray, shape (N, 3)
        colors : np.ndarray, shape (N, 3), range [0, 1]
        """
        if ROS_NUMPY_AVAILABLE:
            try:
                pc = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
                points = np.stack([pc["x"], pc["y"], pc["z"]], axis=-1).astype(np.float32)

                if "rgb" in pc.dtype.names:
                    rgb = pc["rgb"].view(np.uint32)
                    r = ((rgb >> 16) & 255).astype(np.float32) / 255.0
                    g = ((rgb >> 8) & 255).astype(np.float32) / 255.0
                    b = (rgb & 255).astype(np.float32) / 255.0
                    colors = np.stack([r, g, b], axis=-1)
                else:
                    colors = np.full_like(points, 0.5, dtype=np.float32)

                finite = np.isfinite(points).all(axis=1)
                return points[finite], colors[finite]

            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0, f"ros_numpy parsing failed, falling back to pc2.read_points: {exc}"
                )

        points, colors = [], []
        for data in pc2.read_points(cloud_msg, skip_nans=True):
            x_val, y_val, z_val = data[:3]
            points.append([x_val, y_val, z_val])

            if len(data) >= 4:
                rgb_float = data[3]
                try:
                    rgb_uint32 = struct.unpack("I", struct.pack("f", rgb_float))[0]
                    r = (rgb_uint32 >> 16) & 255
                    g = (rgb_uint32 >> 8) & 255
                    b = rgb_uint32 & 255
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
                except Exception:
                    colors.append([0.5, 0.5, 0.5])
            else:
                colors.append([0.5, 0.5, 0.5])

        return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)

    def array_to_pointcloud2(self, points, colors, header):
        """
        Convert point and color arrays into a PointCloud2 message.
        """
        cloud_data = []
        for idx in range(len(points)):
            x_val, y_val, z_val = points[idx]
            r, g, b = colors[idx]
            rgb_uint32 = struct.unpack(
                "I",
                struct.pack(
                    "BBBB",
                    int(b * 255) & 255,
                    int(g * 255) & 255,
                    int(r * 255) & 255,
                    0,
                ),
            )[0]
            cloud_data.append([x_val, y_val, z_val, rgb_uint32])

        fields = [
            pc2.PointField("x", 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField("y", 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField("z", 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField("rgb", 12, pc2.PointField.UINT32, 1),
        ]
        return pc2.create_cloud(header, fields, cloud_data)

    # -------------------------------------------------------------------------
    # TF helpers
    # -------------------------------------------------------------------------
    def lookup_T(self, dst_frame, src_frame, stamp, timeout):
        """
        Look up a 4x4 homogeneous transform matrix.

        Returns
        -------
        np.ndarray or None
            Homogeneous transform matrix from src_frame to dst_frame.
        """
        if dst_frame == src_frame:
            return np.eye(4, dtype=np.float32)

        try:
            tf_msg = self.tfbuf.lookup_transform(
                dst_frame, src_frame, stamp, rospy.Duration(timeout)
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, f"TF lookup failed {dst_frame} <- {src_frame}: {exc}"
            )
            return None

        trans = tf_msg.transform.translation
        quat = tf_msg.transform.rotation
        transform = tft.quaternion_matrix([quat.x, quat.y, quat.z, quat.w]).astype(np.float32)
        transform[0, 3] = trans.x
        transform[1, 3] = trans.y
        transform[2, 3] = trans.z
        return transform

    def transform_points(self, points, src_frame, dst_frame, stamp, timeout):
        """
        Transform an array of 3D points from src_frame to dst_frame.
        """
        if src_frame == dst_frame:
            return points

        transform = self.lookup_T(dst_frame, src_frame, stamp, timeout)
        if transform is None:
            return None

        pts_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
        out = (transform @ pts_h.T).T[:, :3].astype(np.float32)
        return out

    def transform_point(self, point, src_frame, dst_frame, stamp, timeout):
        """
        Transform a single 3D point from src_frame to dst_frame.
        """
        pts = np.asarray(point, dtype=np.float32).reshape(1, 3)
        out = self.transform_points(pts, src_frame, dst_frame, stamp, timeout)
        if out is None:
            return None
        return out.reshape(3,)

    def get_frame_origin_in(self, src_frame, dst_frame, stamp, timeout):
        """
        Get the origin of src_frame expressed in dst_frame.
        """
        if src_frame == dst_frame:
            return np.zeros(3, dtype=np.float32)

        transform = self.lookup_T(dst_frame, src_frame, stamp, timeout)
        if transform is None:
            return None
        return transform[:3, 3].astype(np.float32)

    # -------------------------------------------------------------------------
    # Open3D construction and filtering
    # -------------------------------------------------------------------------
    def create_open3d_pointcloud(self, points, colors):
        """
        Build an Open3D point cloud using either tensor or legacy mode.
        """
        if self.use_gpu:
            pts = o3d.core.Tensor(points, o3d.core.Dtype.Float32, self.device)
            cols = o3d.core.Tensor(colors, o3d.core.Dtype.Float32, self.device)
            pcd = o3d.t.geometry.PointCloud(self.device)
            pcd.point["positions"] = pts
            pcd.point["colors"] = cols
            return pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        return pcd

    def apply_filters(self, pcd, do_radius=True):
        """
        Apply voxel downsampling and optional radius outlier filtering.

        Parameters
        ----------
        pcd : Open3D point cloud
        do_radius : bool
            Whether to apply radius outlier filtering on this frame.
        """
        if self.voxel_size > 0:
            try:
                pcd = pcd.voxel_down_sample(self.voxel_size)
            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0,
                    f"voxel_down_sample failed, falling back to CPU legacy backend: {exc}",
                )
                self.use_gpu = False
                self.device = o3d.core.Device("CPU:0")
                pts = np.asarray(pcd.point["positions"].cpu().numpy(), dtype=np.float32)
                cols = np.asarray(pcd.point["colors"].cpu().numpy(), dtype=np.float32)
                pcd = self.create_open3d_pointcloud(pts, cols)
                pcd = pcd.voxel_down_sample(self.voxel_size)

        if do_radius and self.radius > 0 and self.min_points > 0:
            if self.use_gpu:
                try:
                    pcd, _ = pcd.remove_radius_outliers(
                        nb_points=int(self.min_points),
                        search_radius=float(self.radius),
                    )
                except Exception as exc:
                    rospy.logwarn_throttle(
                        1.0,
                        f"remove_radius_outliers failed, falling back to CPU legacy backend: {exc}",
                    )
                    self.use_gpu = False
                    self.device = o3d.core.Device("CPU:0")
                    pts = np.asarray(pcd.point["positions"].cpu().numpy(), dtype=np.float32)
                    cols = np.asarray(pcd.point["colors"].cpu().numpy(), dtype=np.float32)
                    pcd = self.create_open3d_pointcloud(pts, cols)
                    pcd, ind = pcd.remove_radius_outlier(
                        nb_points=int(self.min_points),
                        radius=float(self.radius),
                    )
                    pcd = pcd.select_by_index(ind)
            else:
                pcd, ind = pcd.remove_radius_outlier(
                    nb_points=int(self.min_points),
                    radius=float(self.radius),
                )
                pcd = pcd.select_by_index(ind)

        return pcd

    def unpack_open3d(self, pcd):
        """
        Extract NumPy point and color arrays from an Open3D point cloud.
        """
        if self.use_gpu:
            pts = pcd.point["positions"].cpu().numpy().astype(np.float32)
            cols = pcd.point["colors"].cpu().numpy().astype(np.float32)
        else:
            pts = np.asarray(pcd.points).astype(np.float32)
            cols = np.asarray(pcd.colors).astype(np.float32)
        return pts, cols

    # -------------------------------------------------------------------------
    # Main timer-driven processing
    # -------------------------------------------------------------------------
    def process_one_cloud(self, cloud_msg: PointCloud2):
        """
        Process one cached point cloud and publish the filtered result.
        """
        try:
            points, colors = self.pointcloud2_to_array(cloud_msg)
            if points.shape[0] == 0:
                return

            # The node only publishes output if a target apple center is available.
            if self.apple_center_frozen is None:
                return

            src_frame = cloud_msg.header.frame_id
            dst_frame = self.target_frame
            stamp = rospy.Time(0)

            # -----------------------------------------------------------------
            # Early ROI cropping before Open3D processing
            # -----------------------------------------------------------------
            if self.enable_early_roi:
                center_in_src = self.transform_point(
                    self.apple_center_frozen,
                    dst_frame,
                    src_frame,
                    stamp,
                    self.tf_timeout,
                )
                if center_in_src is not None:
                    roi_r = float(self.keep_radius + self.roi_margin)
                    dist0 = np.linalg.norm(points - center_in_src[None, :], axis=1)
                    keep0 = dist0 < roi_r
                    points = points[keep0]
                    colors = colors[keep0]
                    if points.shape[0] == 0:
                        return

            # -----------------------------------------------------------------
            # Open3D filtering with optional reduced-frequency radius filtering
            # -----------------------------------------------------------------
            self._frame_count += 1
            do_radius = True
            if self.radius_filter_every_n > 1:
                do_radius = (self._frame_count % self.radius_filter_every_n == 0)

            pcd = self.create_open3d_pointcloud(points, colors)
            pcd = self.apply_filters(pcd, do_radius=do_radius)
            filtered_points, filtered_colors = self.unpack_open3d(pcd)

            if filtered_points.shape[0] == 0:
                return

            # -----------------------------------------------------------------
            # Transform to the target frame only once
            # -----------------------------------------------------------------
            pts_base = self.transform_points(
                filtered_points,
                src_frame,
                dst_frame,
                stamp,
                self.tf_timeout,
            )
            if pts_base is None:
                return

            # -----------------------------------------------------------------
            # Remove points inside the apple sphere
            # -----------------------------------------------------------------
            if (
                self.remove_apple
                and self.apple_remove_enable
                and self.apple_center_frozen is not None
            ):
                center = self.apple_center_frozen
                distances = np.linalg.norm(pts_base - center[None, :], axis=1)
                keep = distances > float(self.apple_remove_radius)
                removed = int(np.sum(~keep))
                if removed > 0:
                    pts_base = pts_base[keep]
                    filtered_colors = filtered_colors[keep]
                    rospy.loginfo_throttle(
                        1.0,
                        f"Apple-sphere removed {removed} points within "
                        f"{self.apple_remove_radius:.3f} m",
                    )

            if pts_base.shape[0] == 0:
                return

            # -----------------------------------------------------------------
            # Remove points near the camera origin
            # -----------------------------------------------------------------
            if self.near_clearance_enable and self.near_clearance_radius > 0:
                cam_origin = self.get_frame_origin_in(
                    self.camera_frame,
                    dst_frame,
                    stamp,
                    self.tf_timeout,
                )
                if cam_origin is not None:
                    dcam = np.linalg.norm(pts_base - cam_origin[None, :], axis=1)
                    keep2 = dcam > float(self.near_clearance_radius)
                    removed_near = int(np.sum(~keep2))
                    pts_base = pts_base[keep2]
                    filtered_colors = filtered_colors[keep2]
                    rospy.loginfo_throttle(
                        1.0,
                        f"Near-clearance removed {removed_near} points within "
                        f"{self.near_clearance_radius:.2f} m",
                    )

            if pts_base.shape[0] == 0:
                return

            # -----------------------------------------------------------------
            # Keep only points near the apple center in the target frame
            # -----------------------------------------------------------------
            center = self.apple_center_frozen
            distances = np.linalg.norm(pts_base - center[None, :], axis=1)
            keep3 = distances < float(self.keep_radius)

            pts_out = pts_base[keep3]
            cols_out = filtered_colors[keep3]
            if pts_out.shape[0] == 0:
                return

            # -----------------------------------------------------------------
            # Publish the filtered point cloud
            # -----------------------------------------------------------------
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = dst_frame

            msg_out = self.array_to_pointcloud2(pts_out, cols_out, header)
            self.filtered_pub.publish(msg_out)

        except Exception as exc:
            rospy.logerr_throttle(1.0, f"Point-cloud filtering failed: {exc}")

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------
    def run(self):
        """Start the ROS spin loop."""
        rospy.loginfo("Point-cloud filtering node started in timer-driven mode.")
        rospy.spin()


if __name__ == "__main__":
    processor = IntegratedPointCloudProcessor()
    processor.run()