#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mask_quality_quantifier.py

ROS1 node for mask-quality visualization and running-mean reporting.

Overview
--------
This node subscribes to a binary apple segmentation mask and an aligned depth
image, computes several mask-quality indicators, overlays the metrics on a
visualization window, and prints running mean statistics to the terminal.

The design goal is lightweight online inspection of segmentation quality in
RGB-D apple perception experiments, especially for comparing occlusion cases
and monitoring mask consistency over time.

Main functions
--------------
1. Mask quality analysis
   - Computes geometric quality indicators from the binary mask:
       * A : full mask area
       * n : number of connected components
       * r : largest-component area ratio
       * c : circularity of the largest component
       * s : solidity of the largest component

2. Depth-aware occlusion analysis
   - Optionally estimates depth-related quantities:
       * Z_obj(med)   : median object depth
       * occ_infront  : occlusion ratio inside the object mask
       * occ_ring     : occlusion ratio in a ring around the mask

3. Visualization
   - Displays either a white binary mask or a filled red mask.
   - Draws metrics as green text in a semi-transparent overlay box.
   - Attempts to avoid overlapping the metric box with the target bounding box.

4. Terminal running statistics
   - Continuously prints running mean values from cached frames.
   - Uses nanmean for depth-related terms so missing values do not bias results.
   - Supports single-line terminal refresh mode.

Subscribed topics
-----------------
/apple_segmentation_mask             : sensor_msgs/Image (mono8)
/in_hand_camera/depth/image_rect_raw : sensor_msgs/Image (16UC1 or 32FC1)

Private parameters
------------------
~mask_topic (str)
    Mask topic. Default: /apple_segmentation_mask

~depth_topic (str)
    Depth topic. Default: /in_hand_camera/depth/image_rect_raw

~style (str)
    Visualization style: "binary" or "red". Default: "binary"

~compute_interval (float)
    Minimum interval between metric updates in seconds. Default: 0.2

~enable_occluder (bool)
    Whether to compute depth-based occlusion terms. Default: True

~delta_z (float)
    Depth margin used to determine front occlusion in meters. Default: 0.015

~max_depth (float)
    Maximum valid depth in meters. Default: 3.0

~depth_unit (str)
    Depth encoding mode: "16UC1_mm" or "32FC1_m". Default: "16UC1_mm"

~enable_ring (bool)
    Whether to compute ring-based occlusion ratio. Default: True

~ring_width_px (int)
    Ring width in pixels. Default: 6

~font_scale (float)
    Overlay text font scale. Default: 0.55

~line_thickness (int)
    Overlay text thickness. Default: 1

~margin_px (int)
    Overlay layout margin in pixels. Default: 12

~box_alpha (float)
    Background box alpha for metric overlay. Default: 0.60

~show_paper_vector_only (bool)
    Reserved display option. Currently retained for compatibility. Default: True

~text_color_bgr (list[int])
    Text color in BGR format. Default: [80, 255, 180]

~enable_terminal_mean (bool)
    Whether to print running mean statistics. Default: True

~terminal_print_interval (float)
    Terminal print interval in seconds. Default: 0.5

~terminal_single_line (bool)
    Whether to refresh a single terminal line instead of printing new lines.
    Default: True

Usage examples
--------------
rosrun your_pkg mask_quality_quantifier.py _style:=binary
rosrun your_pkg mask_quality_quantifier.py _style:=red _box_alpha:=0.65 _font_scale:=0.50
rosrun your_pkg mask_quality_quantifier.py _terminal_print_interval:=0.2
"""

import math
import sys
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class MaskQualityOverlayViewerMean:
    """ROS node for online mask-quality analysis and running-mean reporting."""

    def __init__(self):
        rospy.init_node("mask_quality_overlay_viewer_mean", anonymous=True)

        self.bridge = CvBridge()

        # ---------------------------------------------------------------------
        # Topic configuration
        # ---------------------------------------------------------------------
        self.mask_topic = rospy.get_param("~mask_topic", "/apple_segmentation_mask")
        self.depth_topic = rospy.get_param("~depth_topic", "/in_hand_camera/depth/image_rect_raw")

        # ---------------------------------------------------------------------
        # Visualization settings
        # ---------------------------------------------------------------------
        self.style = rospy.get_param("~style", "binary")

        self.font_scale = float(rospy.get_param("~font_scale", 0.55))
        self.line_thickness = int(rospy.get_param("~line_thickness", 1))
        self.margin_px = int(rospy.get_param("~margin_px", 12))
        self.box_alpha = float(rospy.get_param("~box_alpha", 0.60))
        self.show_paper_vector_only = bool(rospy.get_param("~show_paper_vector_only", True))

        self.text_color = tuple(rospy.get_param("~text_color_bgr", [80, 255, 180]))

        # ---------------------------------------------------------------------
        # Processing rate control
        # ---------------------------------------------------------------------
        self.compute_interval = float(rospy.get_param("~compute_interval", 0.2))
        self.last_compute_t = 0.0

        # ---------------------------------------------------------------------
        # Depth-aware occlusion settings
        # ---------------------------------------------------------------------
        self.enable_occluder = bool(rospy.get_param("~enable_occluder", True))
        self.delta_z = float(rospy.get_param("~delta_z", 0.015))
        self.max_depth = float(rospy.get_param("~max_depth", 3.0))
        self.depth_unit = rospy.get_param("~depth_unit", "16UC1_mm")

        self.enable_ring = bool(rospy.get_param("~enable_ring", True))
        self.ring_width_px = int(rospy.get_param("~ring_width_px", 6))

        # ---------------------------------------------------------------------
        # Terminal reporting settings
        # ---------------------------------------------------------------------
        self.enable_terminal_mean = bool(rospy.get_param("~enable_terminal_mean", True))
        self.terminal_print_interval = float(rospy.get_param("~terminal_print_interval", 0.5))
        self.terminal_single_line = bool(rospy.get_param("~terminal_single_line", True))
        self._last_terminal_print_t = 0.0

        # ---------------------------------------------------------------------
        # Cached statistics
        # Each row: [A, n, r, c, s, z_obj, occ_infront, occ_ring]
        # ---------------------------------------------------------------------
        self.history = []

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------
        self.depth_image = None
        self.last_metrics = None
        self.last_metrics_t = 0.0

        # ---------------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------------
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        rospy.Subscriber(self.mask_topic, Image, self.mask_callback, queue_size=1)

        rospy.loginfo("[MaskQualityOverlayViewerMean] started")
        rospy.loginfo("  mask_topic: %s", self.mask_topic)
        rospy.loginfo("  depth_topic: %s", self.depth_topic)
        rospy.loginfo("  style: %s (binary|red)", self.style)
        rospy.loginfo("  compute_interval: %.3f s", self.compute_interval)
        rospy.loginfo(
            "  enable_occluder: %s  enable_ring: %s",
            str(self.enable_occluder),
            str(self.enable_ring),
        )
        rospy.loginfo("  delta_z: %.3f m, ring_width_px: %d", self.delta_z, self.ring_width_px)
        rospy.loginfo(
            "  terminal mean: %s (interval=%.3f, single_line=%s)",
            str(self.enable_terminal_mean),
            self.terminal_print_interval,
            str(self.terminal_single_line),
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------
    def depth_callback(self, msg):
        """Decode and store the latest depth image."""
        try:
            if self.depth_unit == "32FC1_m":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as exc:
            rospy.logwarn("[MaskQualityOverlayViewerMean] depth callback error: %s", exc)
            self.depth_image = None

    def mask_callback(self, msg):
        """
        Decode the binary mask, update metrics at the configured rate,
        refresh terminal output, and display the visualization window.
        """
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
            mask_bin = (mask > 0).astype(np.uint8)
        except Exception as exc:
            rospy.logwarn("[MaskQualityOverlayViewerMean] mask callback error: %s", exc)
            return

        now = time.time()
        if now - self.last_compute_t >= self.compute_interval:
            self.last_compute_t = now
            self.last_metrics = self.analyze_mask(mask_bin)
            self.last_metrics_t = now

            # Push valid metrics into the running cache without additional filtering.
            if self.last_metrics is not None and self.last_metrics.get("valid", False):
                self.push_cache(self.last_metrics)

            # Print running mean statistics at the configured terminal interval.
            if self.enable_terminal_mean and (now - self._last_terminal_print_t >= self.terminal_print_interval):
                self._last_terminal_print_t = now
                self.print_running_mean()

        vis = self.render_mask(mask_bin)
        vis = self.draw_metrics(vis, self.last_metrics)
        cv2.imshow("Apple Mask + Quality Metrics", vis)
        cv2.waitKey(1)

    # -------------------------------------------------------------------------
    # Cache and terminal output
    # -------------------------------------------------------------------------
    def push_cache(self, metrics):
        """Append one valid metric record to the history buffer."""
        area = float(metrics["A"])
        n_comp = float(metrics["n"])
        largest_ratio = float(metrics["r"])
        circularity = float(metrics["c"])
        solidity = float(metrics["s"])

        z_obj = metrics.get("z_obj", None)
        occ_in = metrics.get("occ_infront", None)
        occ_ring = metrics.get("occ_ring", None)

        row = [
            area,
            n_comp,
            largest_ratio,
            circularity,
            solidity,
            float(z_obj) if z_obj is not None else np.nan,
            float(occ_in) if occ_in is not None else np.nan,
            float(occ_ring) if occ_ring is not None else np.nan,
        ]
        self.history.append(row)

    def print_running_mean(self):
        """Print running mean statistics over the cached history."""
        if len(self.history) == 0:
            return

        data = np.array(self.history, dtype=np.float64)
        mean = np.nanmean(data, axis=0)

        line = (
            f"[mean] N={len(self.history):4d} | "
            f"A={mean[0]:8.1f} n={mean[1]:5.2f} "
            f"r={mean[2]:6.3f} c={mean[3]:6.3f} s={mean[4]:6.3f} | "
            f"Z_obj={mean[5]:6.3f} occ_in={mean[6]:6.3f} occ_ring={mean[7]:6.3f}"
        )

        if self.terminal_single_line:
            sys.stdout.write("\r" + line + " " * 10)
            sys.stdout.flush()
        else:
            print(line)

    # -------------------------------------------------------------------------
    # Metric computation
    # -------------------------------------------------------------------------
    def analyze_mask(self, mask):
        """
        Compute geometric and optional depth-aware quality metrics from a binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask with values in {0, 1}.

        Returns
        -------
        dict
            Metric dictionary including validity flag and overlay bounding box.
        """
        total_area = int(mask.sum())
        if total_area == 0:
            return {"valid": False, "msg": "No mask"}

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        n_components = int(len(component_areas))
        if n_components <= 0:
            return {"valid": False, "msg": "No components"}

        largest_area = int(component_areas.max())
        largest_ratio = float(largest_area) / float(max(total_area, 1))

        largest_idx = 1 + int(np.argmax(component_areas))
        largest_mask = (labels == largest_idx).astype(np.uint8)

        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            z_obj, occ_infront, occ_ring = self.compute_occluder_ratios(largest_mask=largest_mask)
            return {
                "valid": True,
                "A": total_area,
                "n": n_components,
                "r": largest_ratio,
                "c": 0.0,
                "s": 0.0,
                "z_obj": z_obj,
                "occ_infront": occ_infront,
                "occ_ring": occ_ring,
                "bbox": None,
            }

        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)

        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True))

        if perimeter > 1e-6:
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        else:
            circularity = 0.0

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1e-6)

        z_obj, occ_infront, occ_ring = self.compute_occluder_ratios(largest_mask=largest_mask)

        return {
            "valid": True,
            "A": total_area,
            "n": n_components,
            "r": largest_ratio,
            "c": float(circularity),
            "s": float(solidity),
            "z_obj": z_obj,
            "occ_infront": occ_infront,
            "occ_ring": occ_ring,
            "bbox": (x, y, bw, bh),
        }

    def depth_to_meters(self, depth):
        """Convert depth image values to meters according to the configured encoding."""
        if self.depth_unit == "32FC1_m":
            return depth.astype(np.float32)
        return depth.astype(np.float32) / 1000.0

    def compute_occluder_ratios(self, largest_mask):
        """
        Compute depth-based occlusion ratios for the dominant mask component.

        Returns
        -------
        (z_obj, occ_infront, occ_ring)
            Depth median and occlusion ratios. Returns (None, None, None) if
            occlusion analysis is disabled or insufficient valid depth exists.
        """
        if (not self.enable_occluder) or (self.depth_image is None):
            return None, None, None

        depth_m = self.depth_to_meters(self.depth_image)
        valid_depth = (depth_m > 0.05) & (depth_m < self.max_depth)

        obj_mask = (largest_mask > 0)
        obj_valid = obj_mask & valid_depth
        if int(obj_valid.sum()) < 50:
            return None, None, None

        z_obj = float(np.median(depth_m[obj_valid]))
        occ_cond = (depth_m < (z_obj - self.delta_z)) & valid_depth

        denom = float(obj_valid.sum())
        occ_infront = float((occ_cond & obj_mask).sum()) / max(denom, 1.0)

        occ_ring = None
        if self.enable_ring:
            k = int(max(1, self.ring_width_px))
            kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)

            dil = (cv2.dilate(largest_mask, kernel, iterations=1) > 0)
            ero = (cv2.erode(largest_mask, kernel, iterations=1) > 0)
            ring = dil & (~ero)

            ring_valid = ring & valid_depth
            if int(ring_valid.sum()) >= 50:
                occ_ring = float((occ_cond & ring).sum()) / float(ring_valid.sum())

        return z_obj, occ_infront, occ_ring

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def render_mask(self, mask_bin):
        """
        Render the mask as either a white binary image or a filled red image.

        Parameters
        ----------
        mask_bin : np.ndarray
            Binary mask with values in {0, 1}.

        Returns
        -------
        np.ndarray
            BGR visualization image.
        """
        h, w = mask_bin.shape
        if self.style.lower() == "red":
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[:, :, 2] = mask_bin * 255
        else:
            gray = (mask_bin * 255).astype(np.uint8)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return vis

    def draw_metrics(self, img_bgr, metrics):
        """
        Draw the metric text box on the visualization image.

        The text box is placed at the lower-left by default and then shifted if
        necessary to reduce overlap with the main mask bounding box.
        """
        if metrics is None:
            return img_bgr

        h, w = img_bgr.shape[:2]
        lines = []

        if not metrics.get("valid", False):
            lines = ["Mask: N/A"]
        else:
            lines.append(f"A: {metrics['A']}")
            lines.append(f"n: {metrics['n']}")
            lines.append(f"r: {metrics['r']:.3f}")
            lines.append(f"c: {metrics['c']:.3f}")
            lines.append(f"s: {metrics['s']:.3f}")

            if self.enable_occluder:
                if metrics.get("z_obj", None) is None:
                    lines.append("Z_obj: n/a")
                    lines.append("occ_infront: n/a")
                    if self.enable_ring:
                        lines.append("occ_ring: n/a")
                else:
                    lines.append(f"Z_obj(med): {metrics['z_obj']:.3f} m")
                    lines.append(f"occ_infront: {metrics['occ_infront']:.3f}")
                    if self.enable_ring:
                        if metrics.get("occ_ring", None) is None:
                            lines.append("occ_ring: n/a")
                        else:
                            lines.append(f"occ_ring: {metrics['occ_ring']:.3f}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = self.font_scale
        th = self.line_thickness
        pad = self.margin_px

        (_, text_h), _ = cv2.getTextSize("Ag", font, fs, th)
        line_h = text_h + int(8 * fs)

        max_w = 0
        for text in lines:
            (tw, _), _ = cv2.getTextSize(text, font, fs, th)
            max_w = max(max_w, tw)

        box_w = max_w + 2 * pad
        box_h = len(lines) * line_h + pad

        x1 = pad
        x2 = x1 + box_w
        y2 = h - pad
        y1 = max(pad, y2 - box_h)

        bbox = metrics.get("bbox", None)
        if bbox is not None:
            mx, my, mbw, mbh = bbox
            mx2, my2 = mx + mbw, my + mbh

            def intersects(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
                return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

            if intersects(x1, y1, x2, y2, mx, my, mx2, my2):
                max_shift = int(0.45 * w)
                step = max(20, int(0.08 * w))
                shifted = False

                for shift in range(step, max_shift + 1, step):
                    nx1 = x1 + shift
                    nx2 = nx1 + box_w
                    if nx2 > w - pad:
                        break
                    if not intersects(nx1, y1, nx2, y2, mx, my, mx2, my2):
                        x1, x2 = nx1, nx2
                        shifted = True
                        break

                if not shifted:
                    ny2 = y2 - step
                    ny1 = max(pad, ny2 - box_h)
                    if ny2 > pad and not intersects(x1, ny1, x2, ny2, mx, my, mx2, my2):
                        y1, y2 = ny1, ny2

        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        img_bgr = cv2.addWeighted(overlay, self.box_alpha, img_bgr, 1 - self.box_alpha, 0)

        y = y1 + pad + text_h
        for text in lines:
            cv2.putText(img_bgr, text, (x1 + pad, y), font, fs, self.text_color, th, cv2.LINE_AA)
            y += line_h

        return img_bgr


if __name__ == "__main__":
    try:
        MaskQualityOverlayViewerMean()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass