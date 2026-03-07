#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apple_segmentation_detector.py

ROS1 node for apple instance segmentation using a YOLOv8 segmentation model.

Overview
--------
This node performs multi-instance apple detection from the in-hand RGB image,
builds clean binary masks directly from polygon contours, estimates 3D target
positions using the aligned depth image and camera intrinsics, and publishes
both visualization and downstream perception results.

Main features
-------------
1. Multi-instance apple detection
   - Detects all valid apple instances in the current RGB frame.
   - Keeps a "best" target for backward-compatible single-target pipelines.
   - Optionally publishes all valid 3D targets as a PoseArray.

2. Clean mask generation
   - Builds masks from segmentation polygons in the original image space.
   - Avoids low-resolution mask upsampling artifacts.
   - Supports mask combination policies:
       * union   : logical union of all detected apple masks
       * largest : only the largest detected apple mask

3. 3D target estimation
   - Uses centroid-based depth lookup on the aligned depth image.
   - Converts image coordinates to 3D camera-frame coordinates.

4. Visualization and monitoring
   - Publishes a clean binary mask.
   - Publishes an RGB visualization image with bounding boxes and labels.
   - Publishes detector status messages.
   - Optionally shows a local OpenCV display window.

Subscribed topics
-----------------
/in_hand_camera/color/image_raw         : sensor_msgs/Image (bgr8)
/in_hand_camera/depth/image_rect_raw    : sensor_msgs/Image (16UC1, mm)
/in_hand_camera/color/camera_info       : sensor_msgs/CameraInfo

Published topics
----------------
/apple_positions_in_hand_camera_array       : geometry_msgs/PointStamped
    Best single target in camera frame (backward-compatible output).

/apple_positions_in_hand_camera_pose_array  : geometry_msgs/PoseArray
    All valid target positions in camera frame.

/apple_segmentation_mask                    : sensor_msgs/Image
    Clean binary mask (mono8).

/detection_visualization                    : sensor_msgs/Image
    RGB image with detection overlays.

/detector_status                            : std_msgs/String
    Detector status text.

Private parameters
------------------
~weights (str)
    Path to YOLO model weights. Default: <pkg>/weights/seg.pt

~confidence_threshold (float)
    YOLO inference confidence threshold. Default: 0.25

~iou_threshold (float)
    YOLO inference IoU threshold. Default: 0.4

~device (str)
    Inference device, e.g. "", "cpu", "0". Default: ""

~min_conf (float)
    Minimum confidence kept after inference. Default: 0.05

~imgsz (int)
    YOLO inference image size. Default: 960

~enable_mask_overlay (bool)
    Whether to render a green mask overlay on the visualization image.
    Default: False

~mask_alpha (float)
    Alpha value for the optional mask overlay. Default: 0.5

~max_depth (float)
    Maximum valid depth in meters. Default: 3.0

~publish_all (bool)
    Whether to publish all valid targets as a PoseArray. Default: True

~max_instances (int)
    Maximum number of detections retained after sorting by mask area.
    Default: 20

~mask_policy (str)
    Mask combination policy: "union" or "largest". Default: "union"

~print_interval (float)
    Terminal print interval in seconds. Default: 2.0

~show_window (bool)
    Whether to show a local OpenCV visualization window. Default: True
"""

import os
import time

import cv2
import numpy as np
import rospy
import rospkg
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except ImportError as exc:
    rospy.logerr("Failed to import ultralytics.YOLO: %s", exc)
    raise


class AppleSegmentationDetector(object):
    """ROS node for multi-instance apple segmentation and 3D position estimation."""

    def __init__(self):
        rospy.init_node("apple_segmentation_detector", anonymous=True)

        self.bridge = CvBridge()

        # ---------------------------------------------------------------------
        # Package paths and model parameters
        # ---------------------------------------------------------------------
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path("nbv_ros")

        default_weight = os.path.join(self.pkg_path, "weights", "seg.pt")
        self.weights_path = rospy.get_param("~weights", default_weight)
        self.conf_thres = float(rospy.get_param("~confidence_threshold", 0.25))
        self.iou_thres = float(rospy.get_param("~iou_threshold", 0.4))
        self.device = rospy.get_param("~device", "")
        self.min_conf = float(rospy.get_param("~min_conf", 0.05))
        self.imgsz = int(rospy.get_param("~imgsz", 960))

        # ---------------------------------------------------------------------
        # Visualization parameters
        # ---------------------------------------------------------------------
        self.enable_mask_overlay = bool(rospy.get_param("~enable_mask_overlay", False))
        self.mask_alpha = float(rospy.get_param("~mask_alpha", 0.5))
        self.show_window = bool(rospy.get_param("~show_window", True))

        # ---------------------------------------------------------------------
        # Depth and output behavior
        # ---------------------------------------------------------------------
        self.max_depth = float(rospy.get_param("~max_depth", 3.0))
        self.publish_all = bool(rospy.get_param("~publish_all", True))
        self.max_instances = int(rospy.get_param("~max_instances", 20))

        self.mask_policy = str(rospy.get_param("~mask_policy", "union")).strip().lower()
        if self.mask_policy not in ["union", "largest"]:
            rospy.logwarn("Invalid ~mask_policy=%s. Falling back to 'union'.", self.mask_policy)
            self.mask_policy = "union"

        # ---------------------------------------------------------------------
        # Logging and runtime state
        # ---------------------------------------------------------------------
        self.print_interval = float(rospy.get_param("~print_interval", 2.0))
        self.last_print_time = time.time()
        self.frame_count = 0

        self.fx = 0.0
        self.fy = 0.0
        self.ppx = 0.0
        self.ppy = 0.0

        self.color_image = None
        self.depth_image = None

        self.last_detection_time = time.time()
        self.detection_timeout = 2.0

        # ---------------------------------------------------------------------
        # Model initialization
        # ---------------------------------------------------------------------
        self.model = None
        self.names = None
        self.apple_class_id = None

        self.init_yolo()
        self.find_apple_class()

        # ---------------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------------
        self.best_pub = rospy.Publisher(
            "/apple_positions_in_hand_camera_array",
            PointStamped,
            queue_size=1
        )
        self.all_pub = rospy.Publisher(
            "/apple_positions_in_hand_camera_pose_array",
            PoseArray,
            queue_size=1
        )
        self.status_pub = rospy.Publisher("/detector_status", String, queue_size=10)
        self.segmentation_pub = rospy.Publisher("/apple_segmentation_mask", Image, queue_size=1)
        self.visualization_pub = rospy.Publisher("/detection_visualization", Image, queue_size=1)

        # ---------------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------------
        rospy.Subscriber(
            "/in_hand_camera/color/image_raw",
            Image,
            self.color_callback,
            queue_size=1
        )
        rospy.Subscriber(
            "/in_hand_camera/depth/image_rect_raw",
            Image,
            self.depth_callback,
            queue_size=1
        )
        rospy.Subscriber(
            "/in_hand_camera/color/camera_info",
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )

        rospy.loginfo("Apple Segmentation Detector initialized.")
        rospy.loginfo(
            "weights=%s conf=%.2f iou=%.2f device=%s imgsz=%d",
            self.weights_path,
            self.conf_thres,
            self.iou_thres,
            str(self.device),
            self.imgsz,
        )
        rospy.loginfo(
            "publish_all=%s max_instances=%d mask_policy=%s",
            str(self.publish_all),
            self.max_instances,
            self.mask_policy,
        )

    # -------------------------------------------------------------------------
    # Model utilities
    # -------------------------------------------------------------------------
    def init_yolo(self):
        """Load the YOLO segmentation model."""
        try:
            self.model = YOLO(self.weights_path)
            self.names = self.model.names
            rospy.loginfo("YOLOv8 segmentation model loaded: %s", self.weights_path)
            rospy.loginfo("Available classes: %s", str(self.names))
        except Exception as exc:
            rospy.logerr("Failed to load YOLOv8 segmentation model: %s", exc)
            raise

    def find_apple_class(self):
        """Find the apple class ID from the model class dictionary/list."""
        self.apple_class_id = None

        if isinstance(self.names, dict):
            items = self.names.items()
        else:
            items = enumerate(self.names)

        for class_id, class_name in items:
            if "apple" in str(class_name).lower():
                self.apple_class_id = int(class_id)
                rospy.loginfo(
                    "Apple class found: '%s' (ID=%d)",
                    str(class_name),
                    self.apple_class_id
                )
                return

        self.apple_class_id = 0
        fallback_name = (
            self.names[0]
            if not isinstance(self.names, dict)
            else self.names.get(0, "0")
        )
        rospy.logwarn(
            "No class containing 'apple' was found. Falling back to class 0: '%s'",
            str(fallback_name),
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def camera_info_callback(self, msg):
        """Store camera intrinsics from the CameraInfo message."""
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.ppx = float(msg.K[2])
        self.ppy = float(msg.K[5])

        rospy.loginfo_once(
            "Camera intrinsics received: fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f",
            self.fx,
            self.fy,
            self.ppx,
            self.ppy,
        )

    def depth_callback(self, msg):
        """Receive the aligned depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as exc:
            rospy.logwarn("Depth callback error: %s", exc)
            self.depth_image = None

    def color_callback(self, msg):
        """Receive the RGB image and trigger frame processing if depth is ready."""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.depth_image is not None:
                self.process_frame(msg.header)
        except Exception as exc:
            rospy.logwarn("Color callback error: %s", exc)

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def get_name(self, cls_id):
        """Return the class name for a given class ID."""
        try:
            if isinstance(self.names, dict):
                return str(self.names.get(cls_id, cls_id))
            return str(self.names[cls_id])
        except Exception:
            return str(cls_id)

    def get_xyz_from_depth(self, cx, cy):
        """
        Convert an image pixel location into a 3D point using the aligned depth image.

        Returns
        -------
        (x, y, z) in camera frame, or (None, None, None) if invalid.
        """
        if self.depth_image is None or self.fx <= 1e-6 or self.fy <= 1e-6:
            return None, None, None

        h, w = self.depth_image.shape
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        depth_m = float(self.depth_image[cy, cx]) / 1000.0
        if depth_m <= 0.0 or depth_m > self.max_depth:
            return None, None, None

        x3d = (cx - self.ppx) * depth_m / self.fx
        y3d = (cy - self.ppy) * depth_m / self.fy
        z3d = depth_m
        return x3d, y3d, z3d

    def mask_from_polygon(self, poly, height, width):
        """
        Build a binary mask from a polygon in image coordinates.

        Parameters
        ----------
        poly : array-like, shape (N, 2)
            Polygon vertices in original image coordinates.
        height, width : int
            Output image size.

        Returns
        -------
        np.ndarray
            Binary mask in uint8, with values 0 or 255.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if poly is None or len(poly) < 3:
            return mask

        pts = np.round(np.array(poly)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [pts], 255)
        return mask

    def centroid_from_mask(self, mask_u8):
        """Return the binary-mask centroid as integer pixel coordinates."""
        moments = cv2.moments((mask_u8 > 0).astype(np.uint8))
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy
        return None, None

    def print_detection_info(self, best_det, all_count, valid_3d_count):
        """Print throttled terminal diagnostics for the current frame."""
        now = time.time()
        if now - self.last_print_time < self.print_interval:
            return
        self.last_print_time = now

        if best_det is None:
            print("No apple detected.")
            return

        x_val, y_val, z_val, conf, cls_name = best_det
        print("\n" + "=" * 60)
        print(
            "Frame %d detections: total=%d valid3D=%d"
            % (self.frame_count, all_count, valid_3d_count)
        )
        print("Best target: %s conf=%.3f" % (cls_name, conf))
        if x_val is not None:
            print("XYZ = (%.4f, %.4f, %.4f) m" % (x_val, y_val, z_val))
        else:
            print("XYZ = n/a (invalid or missing depth)")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------
    def process_frame(self, header):
        """Run segmentation, visualization, and topic publishing for one RGB-D frame."""
        self.frame_count += 1

        try:
            results = self.model.predict(
                self.color_image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
                retina_masks=True,
            )
        except Exception as exc:
            rospy.logerr("YOLO inference failed: %s", exc)
            return

        vis_image = self.color_image.copy()
        height, width = vis_image.shape[:2]

        detections = []
        best = None
        best_area = -1

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None or len(boxes) == 0:
                continue
            if masks is None or masks.xy is None:
                continue

            polys = masks.xy
            n_items = min(len(boxes), len(polys))

            for i in range(n_items):
                box = boxes[i]
                poly = polys[i]

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if self.apple_class_id is not None and cls != self.apple_class_id:
                    continue
                if conf < self.min_conf:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                mask_bin = self.mask_from_polygon(poly, height, width)
                area = int((mask_bin > 0).sum())
                if area <= 0:
                    continue

                cx, cy = self.centroid_from_mask(mask_bin)
                if cx is None:
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                x3d, y3d, z3d = self.get_xyz_from_depth(cx, cy)
                cls_name = self.get_name(cls)

                det = {
                    "mask": mask_bin,
                    "area": area,
                    "box": (x1, y1, x2, y2),
                    "cxcy": (cx, cy),
                    "xyz": (x3d, y3d, z3d),
                    "conf": conf,
                    "cls": cls,
                    "name": cls_name,
                }
                detections.append(det)

                if area > best_area:
                    best_area = area
                    best = det

        detections.sort(key=lambda det: det["area"], reverse=True)
        if self.max_instances > 0:
            detections = detections[:max(1, self.max_instances)]

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        if len(detections) > 0:
            if self.mask_policy == "largest":
                combined_mask = detections[0]["mask"]
            else:
                for det in detections:
                    combined_mask = cv2.bitwise_or(combined_mask, det["mask"])

        if self.enable_mask_overlay and np.any(combined_mask > 0):
            overlay = np.zeros_like(vis_image)
            overlay[combined_mask > 0] = (0, 255, 0)
            vis_image = cv2.addWeighted(vis_image, 1.0, overlay, self.mask_alpha, 0)

        valid_3d_count = 0
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx, cy = det["cxcy"]
            x3d, y3d, z3d = det["xyz"]

            if x3d is not None:
                valid_3d_count += 1

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(vis_image, (cx, cy), 4, (0, 0, 255), -1)

            line1 = "%s (%.2f)" % (det["name"], det["conf"])
            line2 = "(%.2f, %.2f, %.2f) m" % (x3d, y3d, z3d) if x3d is not None else ""

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
            (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
            text_width = max(tw1, tw2)
            text_height = th1 + (th2 if line2 else 0) + 8

            text_x = x1
            text_y = max(0, y1 - text_height - 6)

            bg = vis_image.copy()
            cv2.rectangle(
                bg,
                (text_x - 4, text_y - 4),
                (text_x + text_width + 4, text_y + text_height),
                (0, 0, 0),
                -1,
            )
            alpha = 0.6
            vis_image = cv2.addWeighted(bg, alpha, vis_image, 1 - alpha, 0)

            cv2.putText(
                vis_image,
                line1,
                (text_x, text_y + th1),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            if line2:
                cv2.putText(
                    vis_image,
                    line2,
                    (text_x, text_y + th1 + th2 + 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

        best_det_for_print = None
        if best is not None:
            x3d, y3d, z3d = best["xyz"]
            best_msg = PointStamped()
            best_msg.header = header

            if x3d is not None:
                best_msg.point.x = x3d
                best_msg.point.y = y3d
                best_msg.point.z = z3d
                self.best_pub.publish(best_msg)
                best_det_for_print = (x3d, y3d, z3d, best["conf"], best["name"])
            else:
                best_det_for_print = (None, None, None, best["conf"], best["name"])

        if self.publish_all and len(detections) > 0:
            pose_array = PoseArray()
            pose_array.header = header

            for det in detections:
                x3d, y3d, z3d = det["xyz"]
                if x3d is None:
                    continue

                pose = Pose()
                pose.position.x = x3d
                pose.position.y = y3d
                pose.position.z = z3d
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)

            if len(pose_array.poses) > 0:
                self.all_pub.publish(pose_array)

        self.print_detection_info(best_det_for_print, len(detections), valid_3d_count)

        if np.any(combined_mask > 0):
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, "mono8")
            mask_msg.header = header
            self.segmentation_pub.publish(mask_msg)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
        vis_msg.header = header
        self.visualization_pub.publish(vis_msg)

        if len(detections) > 0:
            self.last_detection_time = time.time()
            self.status_pub.publish(String(data="Detected apples: %d" % len(detections)))
        else:
            if time.time() - self.last_detection_time > self.detection_timeout:
                self.status_pub.publish(String(data="Searching for apple..."))

        if self.show_window:
            cv2.imshow("Apple Detection", vis_image)
            cv2.waitKey(1)

    # -------------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------------
    def run(self):
        """Start the ROS spin loop."""
        rospy.loginfo("Starting Apple Segmentation Detector...")
        rospy.spin()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        detector = AppleSegmentationDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr("Segmentation detector failed: %s", exc)