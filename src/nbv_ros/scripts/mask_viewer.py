#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mask_overlay_viewer.py

Overlay apple segmentation mask in red for visualization.
Best for paper screenshots.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MaskOverlayViewer:

    def __init__(self):
        rospy.init_node("mask_overlay_viewer")

        self.bridge = CvBridge()

        rospy.Subscriber(
            "/apple_segmentation_mask",
            Image,
            self.callback,
            queue_size=1
        )

        rospy.loginfo("Mask Overlay Viewer started.")
        rospy.spin()

    def callback(self, msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if len(mask.shape) != 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            mask_binary = (mask > 0).astype(np.uint8)

            # 创建黑底图
            h, w = mask_binary.shape
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # 红色通道叠加
            canvas[:, :, 2] = mask_binary * 255

            # 半透明效果
            alpha = 0.6
            overlay = (canvas * alpha).astype(np.uint8)

            cv2.imshow("Apple Mask Overlay", overlay)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(str(e))


if __name__ == "__main__":
    MaskOverlayViewer()