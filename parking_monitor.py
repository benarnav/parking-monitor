import csv
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from camera import Camera
from config import config
from detector import Detector
from service_request import ServiceRequest, ServiceRequestStatusCodes
from utils import retry


def create_camera_list(filename=config["cameras"]):
    """
    Creates a list of Camera objects from a CSV file.

    Args:
        filename (str): Path to the CSV file containing camera information.

    Returns:
        list: List of Camera objects.
    """

    camera_list = []
    with open(filename, mode="r") as camera_file:
        reader = csv.DictReader(camera_file, delimiter=",")
        for row in reader:
            cam = Camera(
                row["camera_name"],
                row["url"],
                row["address"],
                row["descriptor"],
                row["description_txt"],
                row["roi"],
            )
            camera_list.append(cam)
    return camera_list


def wait_for_start_time(start_time: datetime):
    """
    Waits until the specified start time is reached.

    Args:
        start_time (datetime): The time to start the monitoring process.
    """

    while datetime.now() < start_time:
        print("WAITING for start time")
        time.sleep(60)


def check_sr_status(camera: Camera, det: Detector, sr: ServiceRequest, csv_path: Path):
    """
    Checks the status of a service request for a given camera.

    Args:
        camera (Camera): The camera object associated with the service request.
        det (Detector): The detector object for processing images.
        sr (ServiceRequest): The service request object for checking status.
        csv_path (Path): Path to the CSV file for logging results.
    """

    sr_num = camera.sr_num
    sr_status = retry(
        sr.check_status,
        function_args=sr_num,
        error_msg="Failed to check service request status",
    )

    camera.refresh = datetime.now()

    # Fix for edge case where status returned is an ERROR (not represented by a code)
    if sr_status is None:
        return

    if sr_status["Status"] == ServiceRequestStatusCodes.CLOSED:
        filtered_detections, _ = det.get_detections(camera, update_refresh=False)
        if filtered_detections is None:
            persistent_vehicles = 0
        else:
            assignments, _ = camera.assign_vehicles(filtered_detections["bboxes"])
            persistent_vehicles = len(assignments)

        camera.write_to_csv(
            csv_path,
            created_date=sr_status["DateTimeSubmitted"],
            closed_date=sr_status["ResolutionActionUpdatedDate"],
            resolution_description=sr_status["ResolutionAction"],
            persistent_vehicles=persistent_vehicles,
        )

        if "summons" in sr_status["ResolutionAction"]:
            camera.refresh = datetime.now() + timedelta(
                hours=config["post_summons_grace_hours"]
            )


def main():
    """
    Main function to run the parking monitoring system.

    This function initializes the system, creates camera list, sets up logging,
    and runs the monitoring loop for the specified duration.
    """

    # Selects between custom trained model and COCO trained model
    weights = config["model"]

    camera_list: list[Camera] = create_camera_list()

    sr = ServiceRequest(
        email=config["email"], password=config["password"], status_key=config["311_key"]
    )
    detector = Detector(weights)

    cwd = Path.cwd()
    today = time.strftime("%Y%m%d")
    logging.basicConfig(filename=f"errors_{today}.log", level=logging.ERROR)
    os.makedirs(cwd / "service_requests", exist_ok=True)
    csv_path = cwd / "service_requests" / f"{today}.csv"

    with open(csv_path, "a", newline="") as csv_file:
        fieldnames = [
            "sr_num",
            "camera_name",
            "first_seen",
            "created_date",
            "closed_date",
            "persistent_vehicles",
            "resolution_description",
            "ref_images",
            "bboxes",
        ]
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)

    start_time = datetime.now().replace(
        hour=config["start_time"]["hour"],
        minute=config["start_time"]["minute"],
        second=0,
    )
    end_time = start_time + timedelta(hours=config["runtime_hours"])

    wait_for_start_time(start_time=start_time)

    print("STARTING MONITORING")
    try:
        while datetime.now() < end_time:
            for camera in camera_list:
                if camera.open_request is False and (
                    datetime.now() - camera.refresh
                ) > timedelta(minutes=1):

                    filtered_detections, image_path = detector.get_detections(camera)

                    if filtered_detections is None:
                        continue

                    camera.update_camera_vehicles(
                        filtered_detections=filtered_detections, image_path=image_path
                    )

                    illegal_parking_exists = camera.check_illegal_parking()

                    if illegal_parking_exists:
                        # Files a service request on the 311 website
                        sr_num = retry(
                            sr.submit_service_request,
                            function_args=camera,
                            error_msg="Failed to submit service request",
                        )
                        if sr_num is None:
                            continue

                        print(sr_num)
                        camera.open_request = True
                        camera.sr_num = sr_num

                elif camera.open_request and (
                    datetime.now() - camera.refresh
                ) > timedelta(minutes=1):
                    check_sr_status(
                        camera=camera, det=detector, sr=sr, csv_path=csv_path
                    )

    except KeyboardInterrupt:
        print("\nStopping camera checks.\n")

    sr.driver.quit()

    print("Checking remaining open service requests.")

    while True:
        open_requests = 0
        for camera in camera_list:
            if camera.open_request:
                open_requests += 1

            if camera.open_request and (datetime.now() - camera.refresh) > timedelta(
                minutes=1
            ):
                check_sr_status(camera=camera, det=detector, sr=sr, csv_path=csv_path)

        if open_requests == 0:
            break


if __name__ == "__main__":
    main()
