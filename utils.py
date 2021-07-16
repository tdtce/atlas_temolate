from typing import Optional, Union, Tuple, Sequence, List, Dict, Any
from pathlib import Path
import numpy as np
import cv2


BboxType = Union[
    Tuple[int, int, int, int], Tuple[int, ...], List[int]
]  # bbox is presented with 4 values: left, top, right, bottom
BboxListType = List[BboxType]
PointType = Union[Tuple[int, int], Tuple[int, ...], List[int]]  # point is presented as (x, y)
PointListType = List[PointType]


def frame_id_from_filename(name: Path):
    try:
        splits = name.stem.split("_")
        return int(splits[-1])
    except ValueError:
        return 0


class VideoWriter:
    """
    Class for video frame by frame writing with ability to use in context manager
    """

    def __init__(self, video_path: Path, fps: int = 25):
        self.video_path = video_path
        self.fps = fps
        self.cv2_video_writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cv2_video_writer:
            self.cv2_video_writer.release()
        self.cv2_video_writer = None
        if exc_val:
            raise

    def write(self, image: np.ndarray, to_bgr: bool = True):
        if to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        if not self.cv2_video_writer:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.cv2_video_writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )

        self.cv2_video_writer.write(image)


def open_image(filename: Path, to_gray: bool = False) -> np.ndarray:
    """
    Open and return only RGB or GRAY image. NOT BGR!!!
    """
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    assert image is not None, f"Image not found: {filename}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.array(image)


class DetectionContainer:
    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        bboxes: Optional[BboxListType] = None,
        confidences: Optional[Sequence[float]] = None,
    ):
        self.image = image
        self.bboxes = bboxes
        self.confidences = confidences

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> "DetectionContainer":
        assert isinstance(input_dict, dict)
        return cls(**input_dict)

    @classmethod
    def _get_dict_from_image_file(cls, filename: Path, to_gray: bool = False) -> Dict[str, Any]:
        image = open_image(filename, to_gray)
        init_dict = {"image": np.array(image), }
        return init_dict

    @classmethod
    def from_image_file(cls, filename: Path, to_gray: bool = False) -> "DetectionContainer":
        init_dict = cls._get_dict_from_image_file(filename, to_gray)
        return cls.from_dict(init_dict)


class SabotageConfig:
    label_to_class = {0: "No sabotage", 1: "Risk of sabotage", 2: "Sabotage"}
    class_to_label = {"No sabotage": 0, "Risk of sabotage": 1, "Sabotage": 2}
    label_to_color = {0: (0, 255, 0), 1: (255, 165, 0), 2: (255, 0, 0)}  # RGB
