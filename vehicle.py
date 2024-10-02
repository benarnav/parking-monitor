import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

import torch

from utils import CentralIDGenerator


@dataclass
class Vehicle:
    """
    Represents a vehicle detected by a camera.

    Attributes:
        id_gen (ClassVar[CentralIDGenerator]): Class-level ID generator.
        id (uuid.UUID): Unique identifier for the vehicle.
        bbox (torch.Tensor): Bounding box coordinates of the vehicle.
        image_names (list): List of image names where the vehicle was detected.
        last_seen (datetime): Timestamp of when the vehicle was last detected.
        first_seen (datetime): Timestamp of when the vehicle was first detected.
    """

    id_gen: ClassVar = CentralIDGenerator()
    id: uuid.UUID = field(default_factory=id_gen.generate_id)
    bbox: torch.Tensor = field(default_factory=list)
    image_names: list = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    first_seen: datetime = field(default_factory=datetime.now)
