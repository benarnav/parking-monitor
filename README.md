# Eye in the Sky: Harnessing AI to Monitor Police Response to Illegal Parking Complaints

## Overview

This project implements an AI-driven system to systematically analyze police response to illegal parking complaints in New York City. The code was used to conduct original research and resulted in the paper, [Eye in the Sky: Harnessing AI to Monitor Police Response to Illegal Parking Complaints](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4974275). The study leverages publicly available traffic camera feeds and computer vision to detect illegal parking incidents, automatically file complaints through the [NYC 311](https://portal.311.nyc.gov/) system and quantify police response.

### Key findings from the study include:

- 52.15% of complaints were closed while vehicles were still illegally parked
- Only 2.87% of complaints resulted in ticket issuance
- Significant discrepancies between official police resolutions and observed ground truth
- Identification of persistent illegal parking hotspots across the city

The code in this repository forms the backbone of the data collection and processing pipeline used in the study. It is also stored at [NYU's library](https://ultraviolet.library.nyu.edu/records/1vs56-e3h85).

## Features

1. **Camera Management (`camera.py`)**: Handles interaction with NYC Department of Transportation (DOT) cameras, including region of interest definition and vehicle tracking.

2. **Object Detection (`detector.py`)**: Implements YOLO-based object detection to identify vehicles in camera feeds and produces custom annotated images.

3. **Service Request Automation (`service_request.py`)**: Automates the process of submitting and tracking 311 service requests for illegal parking.

4. **Parking Monitor (`parking_monitor.py`)**: Coordinates the overall monitoring process, including illegal parking detection, complaint submission, and response tracking.

## Limitations

- Limited by the resolution and placement of existing DOT cameras
- Potential for false positives in challenging lighting conditions or complex urban scenes
- Does not account for potential variations in enforcement due to special events or emergencies

## Citation

If you use this code, please cite it:

Arnav, B., & Ensari, E. (2024). Automated Illegal Parking Monitor for 311 Based on Publicly Available Data. New York University. https://doi.org/10.58153/1vs56-e3h85

## License & Distribution

This code is licensed under the [AGPL license](https://www.gnu.org/licenses/agpl-3.0.en.html).
