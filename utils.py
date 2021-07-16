from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Sequence, List, Dict, Any
from pathlib import Path
from torch import cuda
from PIL import Image

import numpy as np
import cv2


SizeHW = Tuple[int, int]  # Size of images in (height, width) order
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


class CommonConfig:
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    data_processed_dir = data_dir / "processed"
    result_dir = project_dir / "artifacts"
    model_dir = project_dir / "models"
    test_data_dir = project_dir / "data" / "test"
    video_chunk_size = 10

    device = "cuda" if cuda.is_available() else "cpu"

class DetectionConfig:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size_hw = (448, 768)
    batch_size = 10
    num_classes = 1
    score_threshold = 0.5
    iou_threshold = 0.5
    max_detections = 50
    data_split_random_seed = 22814886969321
    checkpoint_path = None


class EfficientDetConfig(DetectionConfig):
    size_hw = (512, 768)
    name = "tf_efficientdet_d3"
    checkpoint_path = "release/efficientdet_d3.pth.tar"
    result_dir = "data"


class DetectionABC(ABC):
    @abstractmethod
    def predict(self, x: DetectionContainer) -> DetectionContainer:
        """
        The method expects a DetectionContainer that contains at least an image.
        After completing the method, you must fill in the bboxes and confidences as well as classes.
        """
        raise NotImplementedError


class ChangeResolution:
    def __init__(self, size_hw: SizeHW):
        self.size_hw = size_hw
        self._resize_func = lambda x: x

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
        if isinstance(image, Image.Image):
            image = np.array(image)
            type_pilimage = True
        else:
            type_pilimage = False

        image = self._resize_func(image)

        if type_pilimage:
            image = Image.fromarray(image)

        return image


def get_letterboxing_param(src_hw: SizeHW, new_hw: SizeHW) -> Tuple[float, List[int], List[int]]:
    scale = min(new_hw[0] / src_hw[0], new_hw[1] / src_hw[1])
    resized_hw = [int(scale * src) for src in src_hw]
    pad = [(new - resized) // 2 for new, resized in zip(new_hw, resized_hw)]
    return scale, resized_hw, pad


def letterbox_image(image: np.ndarray, size_hw: SizeHW) -> np.ndarray:
    """
    Function resizes image to desired resolution with keeping aspect ratio
    """
    dtype = image.dtype
    src_hw = image.shape[:2]

    if size_hw == src_hw:
        return image

    _, (resized_h, resized_w), (pad_h, pad_w) = get_letterboxing_param(src_hw, size_hw)

    image_resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR).astype(dtype)

    if size_hw == (resized_h, resized_w):
        return image_resized

    if image.ndim == 2:
        image_new = np.zeros((size_hw[0], size_hw[1]), dtype=dtype)
    else:
        image_new = np.zeros((size_hw[0], size_hw[1], image.shape[2]), dtype=dtype)

    image_new[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w, ...] = image_resized.copy()

    return image_new


class LetterboxImage(ChangeResolution):
    def __init__(self, size_hw: SizeHW):
        super().__init__(size_hw)
        self._resize_func = lambda x: letterbox_image(x, self.size_hw)


def deletterbox_image(curr_image: np.ndarray, src_hw: SizeHW) -> np.ndarray:
    """
    Function resizes letterboxed image to resoultion of source image. Extra black borders are removing.
    """
    curr_h, curr_w = curr_image.shape[:2]
    curr_hw = (curr_h, curr_w)
    dtype = curr_image.dtype

    _, (resized_h, resized_w), (pad_h, pad_w) = get_letterboxing_param(src_hw, curr_hw)

    image = curr_image[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w, ...]

    src_image = cv2.resize(image, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_LANCZOS4).astype(dtype)

    return src_image


def to_int(x: Any):
    if isinstance(x, list):
        return list(map(to_int, x))
    elif isinstance(x, tuple):
        return tuple(map(to_int, x))
    elif isinstance(x, (float, int)):
        return int(x)
    elif isinstance(x, np.ndarray):
        return x.astype(int)
    else:
        try:
            return int(x)
        except Exception:
            raise TypeError(f'Not implemented to "int" converting for this type {type(x)}')


def bbox_fourpoint2array(bbox_4point: BboxType) -> np.ndarray:
    """
    Convert bbox from [xmin, ymin, xmax, ymax] to array with 4 point (LT, RT, RB, LB) as (x_k, y_k)
    """
    assert isinstance(bbox_4point, (tuple, list))
    assert len(bbox_4point) == 4

    bbox_array = np.zeros((4, 2))
    bbox_array[0, :] = (bbox_4point[0], bbox_4point[1])
    bbox_array[1, :] = (bbox_4point[2], bbox_4point[1])
    bbox_array[2, :] = (bbox_4point[2], bbox_4point[3])
    bbox_array[3, :] = (bbox_4point[0], bbox_4point[3])

    return bbox_array


def deletterbox_pts_array(curr_pts: np.ndarray, curr_hw: SizeHW, src_hw: SizeHW) -> np.ndarray:
    """
    Recalculate arbitrary array of points (presented as [[x_1, y1], [x_2, y_2], , ...]) from
    letterboxed image to source image
    """
    scale, resized_hw, pad = get_letterboxing_param(src_hw, curr_hw)

    src_pts_array = (curr_pts - np.array(pad)[np.newaxis, ::-1]) / scale
    src_pts_array[:, 0] = np.clip(src_pts_array[:, 0], 0, src_hw[1])
    src_pts_array[:, 1] = np.clip(src_pts_array[:, 1], 0, src_hw[0])

    return src_pts_array


def bbox_array2fourpoint(bbox_array: np.ndarray) -> BboxType:
    """
    Convert bbox array with 4 point (LT, RT, RB, LB) as (x_k, y_k) to [xmin, ymin, xmax, ymax] format
    """
    assert isinstance(bbox_array, np.ndarray)
    assert bbox_array.shape == (4, 2)

    bbox = bbox_array[0, 0], bbox_array[0, 1], bbox_array[2, 0], bbox_array[2, 1]

    return bbox


def deletterbox_bbox(curr_bbox: BboxType, curr_hw: SizeHW, src_hw: SizeHW) -> BboxType:
    """
    Resize of bbox: Bbox in letterboxed image (curr_hw) -> Bbox in source image (src_hw)
    Bbox presented as 4 point.
    """
    bbox_array = bbox_fourpoint2array(curr_bbox)
    bbox_array = deletterbox_pts_array(bbox_array, curr_hw, src_hw)
    bbox_4point = bbox_array2fourpoint(bbox_array)
    return bbox_4point