#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
integrated_apple_pointcloud_processor.py

Integrated multi-apple point-cloud processor for RGB-D apple perception.

Overview
--------
This node implements a complete geometric processing pipeline for apple surface
extraction and spherical completion using:
    Segmentation -> Detection -> Split -> Completion

The node takes a binary apple mask and an aligned RGB-D point cloud as input,
extracts target point subsets through image-space projection, detects multiple
apple clusters, splits ambiguous merged clusters, completes missing spherical
surface regions, and publishes both reconstructed point clouds and estimated
apple centers/radii.

Main stages
-----------
1. Segmentation
   - Refines the input binary mask
   - Projects the mask into the input point cloud
   - Extracts candidate apple surface points
   - Applies voxel downsampling and radius outlier removal

2. Detection
   - Clusters the segmented points into multiple apple candidates
   - Uses sphere-fitting-based geometric filtering
   - Applies temporal matching for frame-to-frame stability

3. Split
   - Further separates clusters that may contain multiple apples
   - Uses residual-based single-sphere validation
   - Falls back to directional or radial splitting if needed

4. Completion
   - Fits a sphere to each cluster
   - Identifies under-observed spherical regions
   - Synthesizes completion points on missing surface sectors
   - Removes points too close to the observed surface

5. Output
   - Publishes segmented point clouds
   - Publishes completed point clouds
   - Publishes estimated apple centers and radii
   - Publishes pipeline status information

Subscribed topics
-----------------
/in_hand_camera/depth/color/points : sensor_msgs/PointCloud2
    Input aligned RGB-D point cloud.

/apple_segmentation_mask : sensor_msgs/Image
    Binary apple segmentation mask.

/in_hand_camera/color/camera_info : sensor_msgs/CameraInfo
    Camera intrinsics used for image-to-point-cloud projection.

Published topics
----------------
/segmented_apple_pointcloud : sensor_msgs/PointCloud2
    Segmented apple point cloud after mask-based extraction and clustering.

/completed_apple_pointcloud : sensor_msgs/PointCloud2
    Completed spherical apple surface point cloud.

/improved_mask_debug : sensor_msgs/Image
    Refined mask used by the segmentation stage.

/segmenter_status : std_msgs/String
    Segmentation-stage status text.

/shape_completion_status : std_msgs/String
    Completion-stage status text.

/pipeline_status : std_msgs/String
    End-to-end pipeline status text.

/apple/center : geometry_msgs/PointStamped
    Estimated apple center(s) in the output frame.

/apple/radius : std_msgs/Float32
    Estimated apple radius/radii.

Notes
-----
- The node is designed for multi-apple processing.
- Apple centers are estimated after completion refinement.
- Point clouds are handled asynchronously using a versioned processing thread.
"""

import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from scipy import ndimage
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sklearn.cluster import DBSCAN
from std_msgs.msg import Float32, Header, String

try:
    import ros_numpy

    ROS_NUMPY_AVAILABLE = True
except Exception:
    ROS_NUMPY_AVAILABLE = False


class IntegratedApplePointCloudProcessor:
    """Integrated multi-apple point-cloud extraction and spherical completion node."""

    def debug(self, msg):
        """Print debug information only when debug mode is enabled."""
        if self.debug_mode:
            rospy.loginfo(msg)

    def __init__(self):
        rospy.init_node("integrated_apple_pointcloud_processor", anonymous=True)

        self.bridge = CvBridge()
        self.debug_mode = False

        # ---------------------------------------------------------------------
        # Processing state
        # ---------------------------------------------------------------------
        self.data_version = 0
        self.processing = False
        self.last_pc_stamp = None
        self.lock = threading.Lock()
        self.frame_count = 0

        # ---------------------------------------------------------------------
        # Stage 1: segmentation and mask processing parameters
        # ---------------------------------------------------------------------
        self.min_depth = 0.1
        self.max_depth = 2.0

        self.voxel_size = 0.001
        self.radius_outlier_nb_points = 80
        self.radius_outlier_radius = 0.01

        self.dbscan_eps_seg = 0.01
        self.dbscan_min_samples_seg = 6
        self.min_cluster_size = 800
        self.max_clusters = 5

        self.min_apple_diameter = 0.07
        self.max_apple_diameter = 0.09
        self.min_cluster_points = 800

        self.erosion_kernel_size = 5
        self.dilation_kernel_size = 3
        self.mask_smooth_size = 5

        # ---------------------------------------------------------------------
        # Temporal consistency parameters
        # ---------------------------------------------------------------------
        self.temporal_smoothing = True
        self.history_size = 5
        self.cluster_history = deque(maxlen=self.history_size)
        self.stability_threshold = 0.7
        self.position_tolerance = 0.05
        self.size_tolerance = 0.3

        # ---------------------------------------------------------------------
        # TF and frame configuration
        # ---------------------------------------------------------------------
        self.output_frame = "in_hand_camera_color_frame"
        self.center_output_frame = "base_link"
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---------------------------------------------------------------------
        # Stage 2: completion parameters
        # ---------------------------------------------------------------------
        self.dbscan_eps_comp = 0.015
        self.dbscan_min_samples_comp = 10
        self.min_cluster_points_comp = 100

        self.shape_completion_points = 2000
        self.phi_bins = 48
        self.theta_bins = 96
        self.radius_jitter_ratio = 0.0001
        self.merge_distance = 0.02
        self.keep_eps = 0.004
        self.min_region_pts_ratio = 0.0005

        # Completion point color encoded as packed PCL RGB (0xRRGGBB).
        r_val, g_val, b_val = 230, 162, 60
        rgb_uint32 = (r_val << 16) | (g_val << 8) | b_val
        self.completion_rgb = np.frombuffer(
            np.uint32(rgb_uint32).tobytes(), dtype=np.float32
        )[0]

        # ---------------------------------------------------------------------
        # ROS publishers
        # ---------------------------------------------------------------------
        self.segmented_pc_pub = rospy.Publisher(
            "/segmented_apple_pointcloud", PointCloud2, queue_size=1, latch=True
        )
        self.mask_debug_pub = rospy.Publisher(
            "/improved_mask_debug", Image, queue_size=1
        )
        self.completed_pc_pub = rospy.Publisher(
            "/completed_apple_pointcloud", PointCloud2, queue_size=1, latch=True
        )
        self.segmenter_status_pub = rospy.Publisher(
            "/segmenter_status", String, queue_size=2
        )
        self.completion_status_pub = rospy.Publisher(
            "/shape_completion_status", String, queue_size=2
        )
        self.pipeline_status_pub = rospy.Publisher(
            "/pipeline_status", String, queue_size=2
        )
        self.apple_center_pub = rospy.Publisher(
            "/apple/center", PointStamped, queue_size=10
        )
        self.apple_radius_pub = rospy.Publisher(
            "/apple/radius", Float32, queue_size=10
        )

        # ---------------------------------------------------------------------
        # ROS subscribers
        # ---------------------------------------------------------------------
        rospy.Subscriber(
            "/in_hand_camera/depth/color/points",
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1,
            buff_size=2**24,
        )
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

        # ---------------------------------------------------------------------
        # Runtime data buffers
        # ---------------------------------------------------------------------
        self.current_pointcloud = None
        self.current_mask = None
        self.camera_info = None

        self.fx = 0.0
        self.fy = 0.0
        self.ppx = 0.0
        self.ppy = 0.0
        self.image_width = 640
        self.image_height = 480

        self.mask_received = False
        self.pointcloud_received = False

        self.debug("Integrated Apple PointCloud Processor initialized")

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def camera_info_callback(self, msg):
        """Store camera intrinsics from CameraInfo."""
        self.camera_info = msg
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.ppx = msg.K[2]
        self.ppy = msg.K[5]
        rospy.loginfo_once("Camera intrinsics received.")

    def mask_callback(self, msg):
        """Store the latest mask and trigger asynchronous processing."""
        try:
            self.current_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.mask_received = True
            self.data_version += 1
            local_version = self.data_version
            self.try_process(local_version)
        except Exception as exc:
            rospy.logwarn_once(f"Mask callback error: {exc}")

    def pointcloud_callback(self, msg):
        """Store the latest point cloud and trigger asynchronous processing."""
        self.current_pointcloud = msg
        self.pointcloud_received = True
        local_version = self.data_version
        self.try_process(local_version)

    def try_process(self, version):
        """Start a worker thread if all required inputs are available."""
        with self.lock:
            if self.processing:
                return
            if not (self.mask_received and self.pointcloud_received and self.fx != 0):
                return

            self.processing = True
            worker = threading.Thread(
                target=self.process_pipeline,
                args=(version,),
            )
            worker.daemon = True
            worker.start()

    # -------------------------------------------------------------------------
    # Stage 1: mask refinement and segmented point extraction
    # -------------------------------------------------------------------------
    def improve_mask_quality(self, mask):
        """
        Refine the binary mask using thresholding, morphology, smoothing,
        connected-component filtering, and hole filling.
        """
        try:
            _, mask_bin = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_noise)

            if self.mask_smooth_size > 1:
                blurred = cv2.GaussianBlur(
                    mask_bin.astype(np.float32),
                    (self.mask_smooth_size, self.mask_smooth_size),
                    0,
                )
                mask_bin = cv2.threshold(
                    blurred, 100, 255, cv2.THRESH_BINARY
                )[1].astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_bin, connectivity=8
            )
            min_area = 200
            clean_mask = np.zeros_like(mask_bin)

            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_area:
                    clean_mask[labels == label] = 255

            dil_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_kernel_size, self.dilation_kernel_size),
            )
            mask_final = cv2.dilate(clean_mask, dil_kernel, iterations=1)
            mask_final = ndimage.binary_fill_holes(mask_final).astype(np.uint8) * 255

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(mask_final, "mono8")
                debug_msg.header.stamp = rospy.Time.now()
                self.mask_debug_pub.publish(debug_msg)
            except Exception:
                pass

            return mask_final

        except Exception as exc:
            rospy.logwarn_once(f"Mask improvement error: {exc}")
            return mask

    def extract_points_using_projection(self, mask, pointcloud_msg):
        """
        Extract point-cloud points whose image projections lie inside the mask.

        Returns
        -------
        np.ndarray
            Array of shape (N, 4) with columns [x, y, z, rgb].
        """
        if ROS_NUMPY_AVAILABLE:
            try:
                pc = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg)
                xs = pc["x"].astype(np.float64)
                ys = pc["y"].astype(np.float64)
                zs = pc["z"].astype(np.float64)

                if "rgb" in pc.dtype.names:
                    rgbs = pc["rgb"].view(np.float32)
                    pts = np.column_stack((xs, ys, zs, rgbs))
                else:
                    rgbs = np.zeros_like(xs, dtype=np.float32)
                    pts = np.column_stack((xs, ys, zs, rgbs))

                finite_mask = np.isfinite(zs)
                pts = pts[finite_mask]

                z_vals = pts[:, 2]
                valid_depth = (z_vals > self.min_depth) & (z_vals < self.max_depth)
                pts = pts[valid_depth]
                goto_projection = True
            except Exception as exc:
                rospy.logwarn_once(f"ros_numpy failed, falling back to raw read: {exc}")
                goto_projection = False
        else:
            goto_projection = False

        try:
            if not goto_projection:
                raw_gen = pc2.read_points(
                    pointcloud_msg,
                    field_names=("x", "y", "z", "rgb"),
                    skip_nans=True,
                )
                pts = np.array(list(raw_gen), dtype=np.float64)

            if pts.size == 0:
                return np.array([])

            z_vals = pts[:, 2]
            valid_depth = np.isfinite(z_vals) & (z_vals > self.min_depth) & (z_vals < self.max_depth)
            if not np.any(valid_depth):
                return np.array([])

            pts = pts[valid_depth]

            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            z_vals = pts[:, 2]

            u = np.rint((x_vals * self.fx / z_vals) + self.ppx).astype(np.int32)
            v = np.rint((y_vals * self.fy / z_vals) + self.ppy).astype(np.int32)

            in_bounds = (
                (u >= 0)
                & (u < self.image_width)
                & (v >= 0)
                & (v < self.image_height)
            )
            if not np.any(in_bounds):
                return np.array([])

            u = u[in_bounds]
            v = v[in_bounds]
            pts = pts[in_bounds]

            mask_values = mask[v, u]
            inside_mask = mask_values > 0
            if not np.any(inside_mask):
                return np.array([])

            selected = pts[inside_mask]
            if selected.size == 0:
                return np.array([])

            xyz = selected[:, :3].astype(np.float32)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            if self.voxel_size > 0 and len(pcd.points) > 100:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

            if len(pcd.points) > self.radius_outlier_nb_points:
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=self.radius_outlier_nb_points,
                    radius=self.radius_outlier_radius,
                )

            if len(pcd.points) == 0:
                return np.array([])

            filtered_xyz = np.asarray(pcd.points)

            try:
                original_xyz = xyz
                original_rgb = selected[:, 3]
                from scipy.spatial import cKDTree

                tree = cKDTree(original_xyz)
                _, idxs = tree.query(filtered_xyz, k=1)
                mapped_rgb = original_rgb[idxs]
                result = np.column_stack([filtered_xyz, mapped_rgb])
            except Exception:
                zeros_rgb = np.zeros((filtered_xyz.shape[0],), dtype=np.float32)
                result = np.column_stack([filtered_xyz, zeros_rgb])

            return result

        except Exception as exc:
            rospy.logerr(f"Error extracting points: {exc}")
            return np.array([])

    # -------------------------------------------------------------------------
    # Cluster feature estimation and temporal matching
    # -------------------------------------------------------------------------
    def calculate_cluster_features(self, points):
        """
        Estimate robust geometric cluster features from sphere fitting.

        Returns
        -------
        dict or None
            Feature dictionary, or None if the cluster is invalid.
        """
        if len(points) == 0:
            return None

        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)

        try:
            a_mat = np.hstack((2 * xyz, np.ones((xyz.shape[0], 1))))
            b_vec = np.sum(xyz**2, axis=1).reshape(-1, 1)
            x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
            cx, cy, cz, c_val = x_sol.flatten()
            radius = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))

            # Reject clusters whose fitted radius is not plausible.
            if not (0.02 < radius < 0.08):
                return None

            diameter = 2.0 * radius

        except Exception:
            return None

        min_vals = np.min(xyz, axis=0)
        max_vals = np.max(xyz, axis=0)
        dimensions = max_vals - min_vals

        return {
            "centroid": centroid,
            "min": min_vals,
            "max": max_vals,
            "diameter": diameter,
            "point_count": len(points),
            "size": dimensions,
        }

    def match_clusters_across_frames(self, current_clusters):
        """
        Match current clusters against the most recent history frame
        to stabilize apple identities across time.
        """
        if not self.temporal_smoothing or not self.cluster_history:
            return current_clusters

        latest_history = self.cluster_history[-1] if self.cluster_history else []
        matched_clusters = []
        used_current_indices = set()

        for hist_cluster in latest_history:
            hist_features = self.calculate_cluster_features(hist_cluster["points"])
            if not hist_features:
                continue

            best_match_idx = -1
            best_match_score = 0.0

            for curr_idx, curr_cluster in enumerate(current_clusters):
                if curr_idx in used_current_indices:
                    continue

                curr_features = self.calculate_cluster_features(curr_cluster)
                if not curr_features:
                    continue

                position_diff = np.linalg.norm(
                    hist_features["centroid"] - curr_features["centroid"]
                )
                size_diff = abs(
                    hist_features["diameter"] - curr_features["diameter"]
                ) / max(hist_features["diameter"], 1e-6)

                if (
                    position_diff < self.position_tolerance
                    and size_diff < self.size_tolerance
                ):
                    score = (
                        1.0
                        - (position_diff / self.position_tolerance) * 0.5
                        - size_diff * 0.5
                    )
                    if score > best_match_score:
                        best_match_score = score
                        best_match_idx = curr_idx

            if best_match_idx >= 0 and best_match_score > self.stability_threshold:
                used_current_indices.add(best_match_idx)
                matched_clusters.append(current_clusters[best_match_idx])

        for curr_idx, curr_cluster in enumerate(current_clusters):
            if curr_idx not in used_current_indices:
                matched_clusters.append(curr_cluster)

        return matched_clusters

    def detect_apples_with_stability(self, points_array):
        """
        Detect apple clusters using DBSCAN and sphere-based validation.

        Returns
        -------
        list[np.ndarray]
            List of valid apple clusters.
        """
        if points_array.size == 0:
            return []

        xyz = points_array[:, :3]
        current_clusters = []

        try:
            clustering = DBSCAN(
                eps=self.dbscan_eps_seg,
                min_samples=self.dbscan_min_samples_seg,
            ).fit(xyz)
            labels = clustering.labels_

            cluster_dict = defaultdict(list)
            for idx, label in enumerate(labels):
                if label != -1:
                    cluster_dict[label].append(points_array[idx])

            for _, cluster_points in cluster_dict.items():
                points_np = np.array(cluster_points)

                if len(points_np) < self.min_cluster_points:
                    self.debug(
                        f"[DEBUG] cluster dropped: too few points "
                        f"({len(points_np)}) < min_cluster_points={self.min_cluster_points}"
                    )
                    continue

                if len(points_np) < self.min_cluster_size:
                    self.debug(
                        f"[DEBUG] cluster dropped: too small "
                        f"({len(points_np)}) < min_cluster_size={self.min_cluster_size}"
                    )
                    continue

                features = self.calculate_cluster_features(points_np)
                if not features:
                    continue

                if features["diameter"] < self.min_apple_diameter * 0.7:
                    self.debug(
                        f"[DEBUG] cluster dropped: diameter too small "
                        f"({features['diameter']:.4f})"
                    )
                    continue

                if features["diameter"] > self.max_apple_diameter * 1.3:
                    self.debug(
                        f"[DEBUG] cluster dropped: diameter too large "
                        f"({features['diameter']:.4f})"
                    )
                    continue

                current_clusters.append(points_np)

            if self.temporal_smoothing:
                current_clusters = self.match_clusters_across_frames(current_clusters)

            if len(current_clusters) > 0:
                cluster_info = []
                for cluster in current_clusters:
                    features = self.calculate_cluster_features(cluster)
                    if features:
                        cluster_info.append(
                            {
                                "points": cluster,
                                "features": features,
                            }
                        )
                self.cluster_history.append(cluster_info)

            return current_clusters

        except Exception as exc:
            rospy.logerr(f"Apple detection error: {exc}")
            return []

    # -------------------------------------------------------------------------
    # Stage 3: cluster splitting
    # -------------------------------------------------------------------------
    def _is_single_sphere(self, xyz):
        """
        Determine whether a cluster is likely to correspond to a single sphere.

        A large residual standard deviation suggests that two spheres may have
        been merged into one cluster.
        """
        a_mat = np.hstack((2 * xyz, np.ones((xyz.shape[0], 1))))
        b_vec = np.sum(xyz**2, axis=1).reshape(-1, 1)
        try:
            x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        except Exception:
            return False

        cx, cy, cz, c_val = x_sol.flatten()
        radius = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))

        dist = np.linalg.norm(xyz - np.array([cx, cy, cz]), axis=1)
        residual = np.abs(dist - radius)

        return residual.std() < (0.03 * max(radius, 1e-6))

    def _split_by_radius_direction(self, cluster):
        """
        Split a potentially merged cluster into two sub-clusters.

        Priority:
        1. Split by directional clustering on normalized radial vectors.
        2. Fall back to a median-radius split.
        """
        xyz = cluster[:, :3]

        a_mat = np.hstack((2 * xyz, np.ones((xyz.shape[0], 1))))
        b_vec = np.sum(xyz**2, axis=1).reshape(-1, 1)
        x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        cx, cy, cz, c_val = x_sol.flatten()
        center = np.array([cx, cy, cz])
        _ = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))

        vecs = xyz - center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / (norms + 1e-8)

        pcd_unit = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unit))
        labels = np.array(
            pcd_unit.cluster_dbscan(
                eps=0.15,
                min_points=20,
                print_progress=False,
            )
        )

        unique_labels = np.unique(labels)
        valid_clusters = []

        if len(unique_labels) >= 2:
            for label in unique_labels:
                if label == -1:
                    continue
                sub = cluster[labels == label]
                if sub.shape[0] >= self.min_cluster_points:
                    valid_clusters.append(sub)

            if len(valid_clusters) >= 2:
                return valid_clusters[:2]

        dist = norms.squeeze()
        median_r = np.median(dist)
        cluster_1 = cluster[dist < median_r]
        cluster_2 = cluster[dist >= median_r]
        return [cluster_1, cluster_2]

    def split_clusters(self, clusters):
        """
        Split ambiguous clusters so that each output cluster corresponds
        as closely as possible to a single apple.
        """
        final_clusters = []

        for cluster in clusters:
            xyz = cluster[:, :3]

            sub_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            sub_labels = np.array(
                sub_pcd.cluster_dbscan(
                    eps=self.dbscan_eps_seg * 0.7,
                    min_points=max(5, self.dbscan_min_samples_seg - 2),
                    print_progress=False,
                )
            )

            unique_sub = np.unique(sub_labels)

            # Case A: effectively one cluster, so directly test single-sphere validity.
            if len(unique_sub) == 1 or (len(unique_sub) == 2 and -1 in unique_sub):
                if self._is_single_sphere(xyz):
                    final_clusters.append(cluster)
                else:
                    spheres = self._split_by_radius_direction(cluster)
                    final_clusters.extend(spheres)
                continue

            # Case B: multiple sub-clusters, test each one separately.
            for label in unique_sub:
                if label == -1:
                    continue

                mask = sub_labels == label
                sub_cluster = cluster[mask]

                if sub_cluster.shape[0] < self.min_cluster_points:
                    continue

                if self._is_single_sphere(sub_cluster[:, :3]):
                    final_clusters.append(sub_cluster)
                else:
                    spheres = self._split_by_radius_direction(sub_cluster)
                    final_clusters.extend(spheres)

        return final_clusters

    # -------------------------------------------------------------------------
    # Stage 4: spherical completion
    # -------------------------------------------------------------------------
    def complete_missing_sphere_section(self, cluster_points, num_points=500):
        """
        Complete missing spherical regions for a single cluster.

        Returns
        -------
        np.ndarray
            Array of shape (N, 4) with columns [x, y, z, rgb].
        """
        if len(cluster_points) < 10:
            return np.array([])

        xyz = cluster_points[:, :3].astype(np.float64)

        a_mat = np.hstack((2 * xyz, np.ones((xyz.shape[0], 1))))
        b_vec = np.sum(xyz**2, axis=1).reshape(-1, 1)
        x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        cx, cy, cz, c_val = x_sol.flatten()
        radius = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))
        center = np.array([cx, cy, cz], dtype=np.float64)

        vecs = xyz - center
        dists = np.linalg.norm(vecs, axis=1)
        valid = dists > 1e-6
        if np.count_nonzero(valid) < 5:
            self.debug("[DEBUG] completion aborted: too few valid direction vectors")
            return np.array([])

        unit = (vecs[valid] / dists[valid][:, None]).astype(np.float64)

        phi = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
        theta = np.arctan2(unit[:, 1], unit[:, 0])
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)

        phi_edges = np.linspace(0.0, np.pi, self.phi_bins + 1)
        theta_edges = np.linspace(0.0, 2 * np.pi, self.theta_bins + 1)

        phi_idx = np.searchsorted(phi_edges, phi, side="right") - 1
        theta_idx = np.searchsorted(theta_edges, theta, side="right") - 1
        phi_idx = np.clip(phi_idx, 0, self.phi_bins - 1)
        theta_idx = np.clip(theta_idx, 0, self.theta_bins - 1)

        hist = np.zeros((self.phi_bins, self.theta_bins), dtype=np.int32)
        np.add.at(hist, (phi_idx, theta_idx), 1)

        min_region_points = max(1, int(self.min_region_pts_ratio * len(xyz)))
        missing_mask = hist < min_region_points
        if not np.any(missing_mask):
            return np.array([])

        orig_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        total_area = 4.0 * np.pi * (radius**2)

        gen_chunks = []
        missing_indices = np.array(np.where(missing_mask)).T
        if missing_indices.size == 0:
            return np.array([])

        for i_phi, i_theta in missing_indices:
            phi_min = phi_edges[i_phi]
            phi_max = phi_edges[i_phi + 1]
            theta_min = theta_edges[i_theta]
            theta_max = theta_edges[i_theta + 1]

            area = (
                (theta_max - theta_min)
                * (np.cos(phi_min) - np.cos(phi_max))
                * (radius**2)
            )
            region_ratio = max(0.0, area / (total_area + 1e-12))
            count = max(2, int(num_points * region_ratio))

            phi_r = np.random.uniform(phi_min, phi_max, count)
            theta_r = np.random.uniform(theta_min, theta_max, count)

            sin_phi = np.sin(phi_r)
            cos_phi = np.cos(phi_r)
            cos_theta = np.cos(theta_r)
            sin_theta = np.sin(theta_r)

            jitter = (
                (2.0 * np.random.rand(count) - 1.0)
                * (self.radius_jitter_ratio * radius)
            )
            r_samp = radius + jitter

            xs = cx + r_samp * sin_phi * cos_theta
            ys = cy + r_samp * sin_phi * sin_theta
            zs = cz + r_samp * cos_phi

            gen_chunks.append(np.column_stack((xs, ys, zs)))

        if not gen_chunks:
            return np.array([])

        all_gen = np.vstack(gen_chunks)

        gen_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_gen))
        dists_to_orig = np.asarray(gen_pcd.compute_point_cloud_distance(orig_pcd))
        mask_keep = dists_to_orig > self.merge_distance
        all_gen = all_gen[mask_keep]

        if all_gen.shape[0] == 0:
            self.debug("[DEBUG] completion aborted: merge-distance filtering removed all points")
            return np.array([])

        gen_pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_gen))
        dists_to_orig_2 = np.asarray(gen_pcd2.compute_point_cloud_distance(orig_pcd))
        final_mask = dists_to_orig_2 > self.keep_eps
        all_gen = all_gen[final_mask]

        if all_gen.shape[0] == 0:
            return np.array([])

        rgb_col = np.full((all_gen.shape[0], 1), self.completion_rgb, dtype=np.float32)
        filled = np.hstack((all_gen.astype(np.float32), rgb_col))
        return filled

    def perform_shape_completion(self, clusters):
        """
        Run spherical completion for each split cluster.

        Parameters
        ----------
        clusters : list[np.ndarray]
            List of split clusters, each with shape (N, 4).

        Returns
        -------
        list[np.ndarray]
            List of completed point sets. Empty arrays are kept for failed cases.
        """
        completed_list_per_cluster = []

        if not clusters:
            return completed_list_per_cluster

        num_clusters = len(clusters)
        per_cluster_points = max(
            200, self.shape_completion_points // max(num_clusters, 1)
        )

        for idx, cluster in enumerate(clusters):
            if cluster.shape[0] < self.min_cluster_points_comp:
                completed_list_per_cluster.append(np.array([]))
                continue

            self.debug(
                f"[DEBUG] spherical completion: cluster {idx}, size={len(cluster)}"
            )

            filled = self.complete_missing_sphere_section(
                cluster, per_cluster_points
            )
            if filled.size > 0:
                completed_list_per_cluster.append(filled)
            else:
                completed_list_per_cluster.append(np.array([]))

        return completed_list_per_cluster

    # -------------------------------------------------------------------------
    # TF helper
    # -------------------------------------------------------------------------
    def transform_point(self, point_xyz, from_frame, to_frame, stamp):
        """Transform a 3D point between frames using tf2."""
        point_stamped = PointStamped()
        point_stamped.header.stamp = stamp
        point_stamped.header.frame_id = from_frame
        point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = map(
            float, point_xyz
        )

        try:
            trans = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                stamp,
                rospy.Duration(0.2),
            )
            point_out = tf2_geometry_msgs.do_transform_point(point_stamped, trans)
            return np.array(
                [point_out.point.x, point_out.point.y, point_out.point.z],
                dtype=np.float64,
            )
        except Exception as exc:
            rospy.logwarn_once(f"TF transform failed: {exc}")
            return None

    # -------------------------------------------------------------------------
    # Main processing pipeline
    # -------------------------------------------------------------------------
    def process_pipeline(self, version):
        """Run the full Segmentation -> Detect -> Split -> Complete pipeline."""
        start_t = time.time()
        self.frame_count += 1

        if version != self.data_version:
            self.processing = False
            return

        with self.lock:
            pc_msg = self.current_pointcloud
            self.output_frame = pc_msg.header.frame_id
            mask_msg = self.current_mask

        try:
            pc_stamp = pc_msg.header.stamp.to_nsec() if hasattr(pc_msg, "header") else None
            if self.last_pc_stamp is not None and pc_stamp == self.last_pc_stamp:
                self.processing = False
                return
            self.last_pc_stamp = pc_stamp

            # -----------------------------------------------------------------
            # Stage 1: segmentation
            # -----------------------------------------------------------------
            segment_start = time.time()
            improved_mask = self.improve_mask_quality(mask_msg)
            segmented_points = self.extract_points_using_projection(
                improved_mask, pc_msg
            )
            self.debug(
                f"[DEBUG] Segmentation extracted {segmented_points.shape[0]} points"
            )
            segment_time = (time.time() - segment_start) * 1000.0

            if segmented_points.size == 0:
                self.publish_segmenter_status("No points extracted from mask.")
                return

            # -----------------------------------------------------------------
            # Stage 2: detection
            # -----------------------------------------------------------------
            clusters_detected = self.detect_apples_with_stability(segmented_points)
            self.debug(
                f"[DEBUG] detect_apples_with_stability: input points = "
                f"{segmented_points.shape[0]}"
            )

            if not clusters_detected:
                self.publish_segmented_pointcloud(segmented_points)
                self.publish_segmenter_status("No apple clusters detected.")
                return

            # -----------------------------------------------------------------
            # Stage 3: split
            # -----------------------------------------------------------------
            clusters_split = self.split_clusters(clusters_detected)
            if not clusters_split:
                # Fall back to the detected clusters if splitting fails entirely.
                clusters_split = clusters_detected

            if len(clusters_split) > self.max_clusters:
                clusters_split = clusters_split[: self.max_clusters]

            combined_points = np.vstack(clusters_split)
            self.publish_segmented_pointcloud(combined_points)
            self.debug(
                f"[DEBUG] combine_all_clusters: {len(clusters_split)} clusters combined"
            )

            total_points = sum(len(c) for c in clusters_split)
            segment_status = (
                f"Frame {self.frame_count}: {len(clusters_split)} apples (split), "
                f"{total_points} points"
            )
            self.publish_segmenter_status(segment_status)

            # -----------------------------------------------------------------
            # Stage 4: completion
            # -----------------------------------------------------------------
            completion_start = time.time()
            completed_points_list = self.perform_shape_completion(clusters_split)
            completion_time = (time.time() - completion_start) * 1000.0

            all_completed = [c for c in completed_points_list if c.size > 0]

            if all_completed:
                completed_points = np.vstack(all_completed)
                self.publish_completed_pointcloud(completed_points)
                completion_status = (
                    f"Completion: {completed_points.shape[0]} points generated"
                )
                self.publish_completion_status(completion_status)

                total_time = (time.time() - start_t) * 1000.0
                pipeline_status = (
                    f"Pipeline: {segment_time:.1f} ms + {completion_time:.1f} ms "
                    f"= {total_time:.1f} ms"
                )
                self.publish_pipeline_status(pipeline_status)
            else:
                self.publish_completion_status("No spherical missing regions detected.")
                self.publish_pipeline_status(
                    f"Segmentation only: {segment_time:.1f} ms"
                )

            # -----------------------------------------------------------------
            # Estimate apple centers and radii from observed + completed points
            # -----------------------------------------------------------------
            centers = []

            for i, cluster in enumerate(clusters_split):
                xyz_vis = cluster[:, :3].astype(np.float64)
                if xyz_vis.shape[0] < 10:
                    continue

                # Initial sphere fit from observed points.
                a_mat = np.hstack((2 * xyz_vis, np.ones((xyz_vis.shape[0], 1))))
                b_vec = np.sum(xyz_vis**2, axis=1).reshape(-1, 1)
                try:
                    x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
                    cx, cy, cz, c_val = x_sol.flatten()
                    center0 = np.array([cx, cy, cz])
                    radius0 = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))
                except Exception:
                    continue

                # Use only the completion points associated with the current apple.
                xyz_comp = None
                if i < len(completed_points_list) and completed_points_list[i].size > 0:
                    xyz_comp = completed_points_list[i][:, :3]

                if xyz_comp is not None:
                    dist_comp = np.linalg.norm(
                        xyz_comp - center0[None, :], axis=1
                    )
                    mask_comp = np.abs(dist_comp - radius0) < 0.01
                    xyz_all = np.vstack([xyz_vis, xyz_comp[mask_comp]])
                else:
                    xyz_all = xyz_vis

                if xyz_all.shape[0] < 10:
                    continue

                # Final sphere fit.
                a_mat = np.hstack((2 * xyz_all, np.ones((xyz_all.shape[0], 1))))
                b_vec = np.sum(xyz_all**2, axis=1).reshape(-1, 1)
                try:
                    x_sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
                    cx, cy, cz, c_val = x_sol.flatten()
                    radius = np.sqrt(max(0.0, c_val + cx * cx + cy * cy + cz * cz))

                    centers.append(
                        {
                            "center": np.array([cx, cy, cz]),
                            "radius": radius,
                        }
                    )
                except Exception:
                    continue

            # -----------------------------------------------------------------
            # Publish centers and radii in the requested output frame
            # -----------------------------------------------------------------
            stamp = pc_msg.header.stamp
            centers_base = []
            radii = []

            for item in centers:
                center_cam = item["center"]
                radius = item["radius"]

                center_base = self.transform_point(
                    center_cam,
                    from_frame=self.output_frame,
                    to_frame=self.center_output_frame,
                    stamp=stamp,
                )
                if center_base is not None:
                    centers_base.append(center_base)
                    radii.append(radius)

            self.publish_apple_centers(centers_base)
            self.publish_apple_radii(radii)

        except Exception as exc:
            rospy.logerr(f"Pipeline processing error: {exc}")
            self.publish_pipeline_status(f"Error: {str(exc)[:50]}")
        finally:
            try:
                self.processing = False
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # ROS publishing helpers
    # -------------------------------------------------------------------------
    def publish_segmented_pointcloud(self, points):
        """Publish the segmented apple point cloud."""
        if points.size == 0:
            return

        try:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.output_frame

            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.FLOAT32, 1),
            ]
            pointcloud_msg = pc2.create_cloud(header, fields, points)
            self.segmented_pc_pub.publish(pointcloud_msg)
        except Exception as exc:
            rospy.logerr_once(f"Segmented publish error: {exc}")

    def publish_completed_pointcloud(self, points):
        """Publish the completed apple point cloud."""
        if points.size == 0:
            return

        try:
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.output_frame

            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.FLOAT32, 1),
            ]
            pc_msg = pc2.create_cloud(header, fields, points)
            self.debug(
                f"[DEBUG] Publishing completed pointcloud with {points.shape[0]} points"
            )
            self.completed_pc_pub.publish(pc_msg)
        except Exception as exc:
            rospy.logerr(f"Completed publish error: {exc}")

    def publish_apple_centers(self, centers_xyz):
        """Publish all estimated apple centers."""
        for center in centers_xyz:
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.center_output_frame
            msg.point.x, msg.point.y, msg.point.z = map(float, center)
            self.apple_center_pub.publish(msg)

    def publish_apple_radii(self, radii):
        """Publish all estimated apple radii."""
        for radius in radii:
            msg = Float32()
            msg.data = float(radius)
            self.apple_radius_pub.publish(msg)

    def publish_segmenter_status(self, message):
        """Publish segmentation-stage status text."""
        try:
            status_msg = String()
            status_msg.data = f"Segmenter: {message}"
            self.segmenter_status_pub.publish(status_msg)
        except Exception:
            pass

    def publish_completion_status(self, message):
        """Publish completion-stage status text."""
        try:
            status_msg = String()
            status_msg.data = f"Completer: {message}"
            self.completion_status_pub.publish(status_msg)
        except Exception:
            pass

    def publish_pipeline_status(self, message):
        """Publish overall pipeline status text."""
        try:
            status_msg = String()
            status_msg.data = f"Pipeline: {message}"
            self.pipeline_status_pub.publish(status_msg)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------
    def run(self):
        """Start the ROS spin loop."""
        rospy.loginfo("Integrated Apple PointCloud Processor running...")
        rospy.loginfo("Published topics:")
        rospy.loginfo("  /segmented_apple_pointcloud  - segmented apple point cloud")
        rospy.loginfo("  /completed_apple_pointcloud  - completed apple point cloud")
        rospy.loginfo("  /segmenter_status            - segmentation-stage status")
        rospy.loginfo("  /shape_completion_status     - completion-stage status")
        rospy.loginfo("  /pipeline_status             - pipeline status")
        rospy.loginfo(f"Max apples to detect: {self.max_clusters}")
        rospy.spin()


if __name__ == "__main__":
    try:
        processor = IntegratedApplePointCloudProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Integrated processor shutting down.")