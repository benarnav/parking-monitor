import ast
import csv
from datetime import datetime, timedelta
from pathlib import Path

import torch
from torchvision import ops

from vehicle import Vehicle


class Camera:
    """
    Represents a NYC Department of Transportation camera used for monitoring illegal
    parking and information relevant for filing a service request with 311.

    Attributes:
        name (str): The name of the camera.
        url (str): The URL to access the camera feed.
        address (str): The physical address where the region of interest is located.
        descriptor (str): Category of illegal parking complaint in the 311 system.
        description_txt (str): A short description of the parking violation.
        roi (list): Region of interest for the camera, parsed from a string.
        open_request (bool): Flag indicating if there's an open service request.
        refresh (datetime): Timestamp of the last refresh.
        vehicles (list[Vehicle]): List of vehicles detected by the camera.
        sr_num (str): Service request number, if there is an open service request for the camera.
        sr_vehicles (list[torch.tensor]): List of bounding boxes of vehicles associated with the service request.
    """

    def __init__(self, camera_name, url, address, descriptor, description_txt, roi):
        self.name: str = camera_name
        self.url: str = url
        self.address: str = address
        self.descriptor: str = descriptor
        self.description_txt: str = description_txt
        self.roi = ast.literal_eval(roi)
        self.open_request: bool = False
        self.refresh: datetime = datetime.now()
        self.vehicles: list[Vehicle] = []
        self.sr_num: str = ""
        self.sr_vehicles: list[torch.tensor] = []

    def write_to_csv(
        self,
        csv_path: Path,
        created_date: datetime,
        closed_date: datetime,
        resolution_description: str,
        persistent_vehicles: int,
    ):
        """
        Writes service request information to a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            created_date (datetime): Date and time when the service request was created.
            closed_date (datetime): Date and time when the service request was closed.
            resolution_description (str): Description provided by NYPD of how the service request was resolved.
            persistent_vehicles (int): Number of vehicles illegally parked when the service request was closed.
        """

        with open(csv_path, "a", newline="") as csv_file:
            row = [
                self.sr_num,
                self.name,
                self.sr_vehicle.first_seen,
                created_date,
                closed_date,
                persistent_vehicles,
                resolution_description,
                self.sr_vehicle.image_names,
                self.sr_vehicle.bbox,
            ]
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row)

        self.vehicles = []
        self.open_request = False

    def assign_vehicles(self, bboxes: list[torch.Tensor]):
        """
        Compares bounding boxes for consecutive images and assigns vehicles to either a new or existing vehicle.

        Args:
            bboxes (list[torch.tensor]): List of bounding boxes for detected vehicles.

        Returns:
            tuple: A tuple containing:
                - dict: Assignments of existing vehicles to new bounding boxes.
                - list: New vehicles that couldn't be matched to existing ones.
        """

        assignments, new_vehicles = {}, []

        for bbox in bboxes:
            max_iou, max_vehicle = 0.0, None

            for vehicle in self.vehicles:
                iou = ops.box_iou(vehicle.bbox.unsqueeze(0), bbox.unsqueeze(0))
                if iou > max_iou and iou > 0.88:
                    max_iou = iou
                    max_vehicle = vehicle.id

            if max_iou == 0.0:
                new_vehicles.append(bbox)
            else:
                assignments[max_vehicle] = bbox

        return assignments, new_vehicles

    def update_camera_vehicles(self, filtered_detections: dict, image_path: Path):
        """
        Updates the list of vehicles detected by the camera based on new detections.

        Args:
            filtered_detections (dict): Dictionary containing filtered detection results.
            image_path (Path): Path to the image file where detections were made.
        """

        assignments, new_vehicles = self.assign_vehicles(filtered_detections["bboxes"])

        for v in self.vehicles:
            if v.id in assignments:
                v.bbox = assignments[v.id]
                v.last_seen = datetime.now()
                v.image_names.append(image_path.name)

        for bbox in new_vehicles:
            self.vehicles.append(
                Vehicle(
                    bbox=bbox,
                    image_names=[image_path.name],
                )
            )

        # Remove vehicles that have not been seen in last 3 minutes
        self.vehicles = [
            v
            for v in self.vehicles
            if (datetime.now() - v.last_seen) < timedelta(minutes=3)
        ]

    def check_illegal_parking(self):
        """
        Checks for vehicles that have been parked for more than 3 minutes.

        Returns:
            bool: True if illegal parking is detected, False otherwise.
        """

        for vehicle in self.vehicles:
            if (vehicle.last_seen - vehicle.first_seen) > timedelta(minutes=3):
                self.sr_vehicle = vehicle
                return True

        return False
