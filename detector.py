import os
import time
from datetime import datetime
from pathlib import Path

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from shapely import MultiPoint, Polygon
from ultralytics import YOLO

from camera import Camera
from utils import retry


class Detector:
    """
    Handles fetching of images from DOT cameras, running inference on images,
    filtering irrelevant detections, and annotating images.

    Attributes:
        today (str): Current date in YYYYMMDD format.
        vehicle_classes (list): List of vehicle class identifiers.
        cwd (Path): Current working directory.
        weights (str): Specifies to use model pre-trained on COCO or custom images.
        model (YOLO): YOLO model instance for object detection.
    """

    def __init__(self, weights: str):
        self.today = time.strftime("%Y%m%d")
        # these classes represent motor vehicles such as cars, trucks, buses, etc.
        self.vehicle_classes = [2.0, 3.0, 5.0, 6.0, 7.0, 8.0]
        self.cwd = Path.cwd()
        self.weights = weights
        if self.weights == "custom":
            self.model = YOLO("custom.pt")
        else:
            self.model = YOLO("best.pt")

    def get_detections(self, camera: Camera, update_refresh: bool = True):
        """
        Fetches an image from the camera, runs detection, and processes the results.

        Args:
            camera (Camera): The camera object to process.
            update_refresh (bool): Whether to update the camera's refresh timestamp.

        Returns:
            tuple: A tuple containing:
                - dict: Filtered detections of vehicles within ROI.
                - Path: Path to the processed image file.
        """

        image_path = retry(
            self._get_image, function_args=camera, error_msg="Failed to get image."
        )

        if update_refresh:
            camera.refresh = datetime.now()

        if image_path is None:
            return None, None

        print(f"\n{camera.name}")
        detections = self._get_model_detections(image_path)
        if detections is None:
            return None, None

        filtered_detections = self._filter_detections(detections, camera.roi)
        if filtered_detections is None:
            return None, None

        self._annotate(camera, image_path, filtered_detections)

        return filtered_detections, image_path

    def _get_image(self, camera: Camera):
        """
        Downloads an image from NYC DOT traffic camera and saves it.

        Args:
            camera (Camera): The camera object to fetch the image from.

        Returns:
            Path: Path to the saved image file.
        """

        url = camera.url
        response = requests.get(url, stream=True)
        response.raise_for_status()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filepath = (
            self.cwd
            / f"{self.today}_imgs"
            / camera.name
            / f"{camera.name}_{timestamp}.jpg"
        )
        os.makedirs(self.cwd / f"{self.today}_imgs" / camera.name, exist_ok=True)

        with open(image_filepath, "wb") as file:
            file.write(response.content)

        return image_filepath

    def _filter_detections(self, detections, roi: list):
        """
        Creates a mask to filter objects within the region of interest.

        Args:
            detections (YOLO results object): List of detections from the model.
            roi (list): Region of interest coordinates.

        Returns:
            dict: Filtered vehicle detections including bounding boxes, confidences, and class labels.
        """

        roi_polygon = Polygon(roi[0])
        boxes = detections[0].boxes.xyxy

        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_points = MultiPoint(list(zip(center_x.tolist(), center_y.tolist())))
        roi_mask = torch.tensor(
            [roi_polygon.intersects(point) for point in list(center_points.geoms)]
        )
        if len(roi) > 1:
            roi2_mask = self.within_roi(boxes, [roi[1]])
            roi_mask = torch.logical_or(roi_mask, roi2_mask)

        boxes_in_roi = detections[0].boxes.xyxy[roi_mask]
        cls_in_roi = detections[0].boxes.cls[roi_mask]
        confs_in_roi = detections[0].boxes.conf[roi_mask]
        if boxes_in_roi.numel() == 0:
            return None

        if self.weights != "custom":
            class_mask = torch.tensor(
                [cls in self.vehicle_classes for cls in cls_in_roi]
            )
            boxes_in_roi = boxes_in_roi[class_mask]
            confs_in_roi = confs_in_roi[class_mask]
            cls_in_roi = cls_in_roi[class_mask]
            if boxes_in_roi.numel() == 0:
                return None

        class_labels = []
        for label in cls_in_roi.tolist():
            class_labels.append(detections[0].names[label])

        output = {
            "bboxes": boxes_in_roi,
            "confs": confs_in_roi,
            "class_labels": class_labels,
        }

        return output

    def _annotate(self, camera: Camera, image_path: Path, filtered_detections: dict):
        """
        Annotates images with relevant detections, bounding boxes, class labels, and confidence scores.

        Args:
            camera (Camera): The camera object associated with the image.
            image_path (Path): Path to the image file.
            filtered_detections (dict): Dictionary of filtered detections to annotate.
        """

        boxes = filtered_detections["bboxes"]
        confidences = filtered_detections["confs"]
        class_labels = filtered_detections["class_labels"]

        # creates an annotated image for reference
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.truetype("Arial.ttf", size=11)
        opacity = 0.6

        if camera.roi is not None:
            draw.polygon(camera.roi[0], outline=(0, 220, 166), width=2)
            if len(camera.roi) > 1:
                draw.polygon(camera.roi[1], outline=(0, 220, 166), width=2)

        for box, confidence, class_label in zip(boxes, confidences, class_labels):
            box = box.tolist()
            xmin, ymin, xmax, ymax = box
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(250, 140, 0), width=2)

            text_x = xmin - 1
            text_y = ymin - 12

            text = f"{class_label} {confidence:.2f}"
            textbbox = draw.textbbox((text_x, text_y), text)
            text_xmin, text_ymin, text_xmax, text_ymax = textbbox
            draw.rounded_rectangle(
                (text_xmin - 3, text_ymin - 3, text_xmax + 3, text_ymax + 3),
                fill=(250, 140, 0, int(255 * opacity)),
                width=3,
                radius=7,
            )
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        os.makedirs(
            self.cwd / f"{self.today}_imgs" / camera.name / "_annotated", exist_ok=True
        )
        image.save(
            self.cwd
            / f"{self.today}_imgs"
            / camera.name
            / "_annotated"
            / image_path.name
        )

    def _get_model_detections(self, image_filepath: Path):
        """
        Runs inference on camera image with a minimum confidence score of 0.5.

        Args:
            image_filepath (Path): Path to the image file.

        Returns:
            list: List of results objects from the model.
        """

        image = Image.open(image_filepath)
        detections = self.model(source=image, conf=0.5)
        if detections[0].boxes.cls.numel() == 0:
            return None

        return detections
