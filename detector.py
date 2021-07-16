from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

from runner import Runner
from utils import EfficientDetConfig, DetectionABC, DetectionConfig, DetectionContainer, SizeHW, BboxListType, \
    CommonConfig, LetterboxImage, to_int, deletterbox_bbox, frame_id_from_filename
import numpy as np
import torch
from effdet.factory import create_model
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


class BatchedDetectorABC(DetectionABC):
    def __init__(
        self,
        num_classes: int,
        score_threshold: float,
        iou_threshold: float,
        max_detections: int,
        batch_size: int,
        progressbar: bool,
        config: DetectionConfig,
    ):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.batch_size = batch_size
        self.progressbar = progressbar
        self.config = config

    def predict_from_path(self, file_name: Path) -> DetectionContainer:
        container = DetectionContainer.from_image_file(file_name)
        return self.predict(container)

    def predict_from_numpy(self, image: np.ndarray) -> DetectionContainer:
        container = DetectionContainer.from_dict({"image": image})
        return self.predict(container)

    def predict(self, container: DetectionContainer) -> DetectionContainer:
        container.confidences, container.classes_num, container.bboxes = self.predict_ndarray_list([container.image])[0]
        return container

    def predict_containers_list(self, containers_list: List[DetectionContainer]) -> List[DetectionContainer]:
        images = [cont.image for cont in containers_list]
        res_list = self.predict_ndarray_list(images)

        for (scores, labels, boxes), cont in zip(res_list, containers_list):
            cont.confidences = scores
            cont.classes_num = labels
            cont.bboxes = boxes

        return containers_list

    @staticmethod
    def collate_fn_inference(data: List[Tuple[torch.Tensor, SizeHW]]) -> Tuple[torch.Tensor, List[SizeHW]]:
        tensors, src_hw_list = zip(*data)
        tensors = torch.cat(tensors, dim=0)
        return tensors, src_hw_list

    def predict_ndarray_list(self, images_list: List[np.ndarray]) -> List[Tuple[List[float], List[int], BboxListType]]:
        if self.progressbar:
            progressbar = tqdm(desc="Detection", total=len(images_list))

        runner = Runner("thread", 2, progressbar=False)
        dataset = runner.run(self.preprocess, images_list)

        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn_inference,
        )

        results_list = []

        for tensors_batch, src_hw_batch in dataloader:
            scores_batch, labels_batch, boxes_batch = self.batched_inference(tensors_batch.to(CommonConfig.device))

            for scores, labels, boxes, src_hw in zip(scores_batch, labels_batch, boxes_batch, src_hw_batch):
                scores, labels, boxes = self.postprocess(scores, labels, boxes, src_hw)
                results_list.append((scores.tolist(), labels.tolist(), to_int(boxes).tolist()))
                if self.progressbar:
                    progressbar.update(1)

        if self.progressbar:
            progressbar.close()
        return results_list

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, SizeHW]:
        assert image.dtype == np.uint8
        tensor_image = self.transforms(image).unsqueeze(0)
        return tensor_image, image.shape[:2]

    def postprocess(
        self, scores: np.ndarray, labels: np.ndarray, boxes: np.ndarray, src_hw: SizeHW
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = to_int(np.array([deletterbox_bbox(bbox.tolist(), self.config.size_hw, src_hw) for bbox in boxes]))

        # select indices which have a score above the threshold
        indices = np.where(scores > self.score_threshold)[0]
        if indices.shape[0] > 0:
            # select those scores
            scores = scores[indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[: self.max_detections]

            # select detections
            image_boxes = boxes[indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[indices[scores_sort]]
            return image_scores, image_labels, image_boxes
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0, 4))

    @abstractmethod
    def batched_inference(self, inputs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class EfficientDetDetector(BatchedDetectorABC):
    def __init__(
        self,
        num_classes: int = EfficientDetConfig.num_classes,
        score_threshold: float = EfficientDetConfig.score_threshold,
        iou_threshold: float = EfficientDetConfig.iou_threshold,
        max_detections: int = EfficientDetConfig.max_detections,
        checkpoint_path: Path = EfficientDetConfig.checkpoint_path,
        batch_size: int = EfficientDetConfig.batch_size,
        progressbar: bool = True,
    ):
        self.net = create_model(
            EfficientDetConfig.name,
            bench_task="predict",
            checkpoint_path=checkpoint_path,
            image_size=EfficientDetConfig.size_hw,
            num_classes=num_classes,
        ).to(CommonConfig.device)
        self.net.eval()
        self.transforms = Compose(
            [
                LetterboxImage(EfficientDetConfig.size_hw),
                ToTensor(),
                Normalize(mean=EfficientDetConfig.mean, std=EfficientDetConfig.std),
            ]
        )
        super().__init__(
            num_classes, score_threshold, iou_threshold, max_detections, batch_size, progressbar, EfficientDetConfig
        )

    @torch.no_grad()
    def batched_inference(self, inputs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        det = self.net(inputs)
        scores_list = []
        labels_list = []
        boxes_list = []

        for batch_idx in range(inputs.shape[0]):
            boxes = det[batch_idx][:, :4]
            scores = det[batch_idx][:, 4]
            labels = det[batch_idx][:, 5] - 1

            scores_over_thresh = scores > self.score_threshold
            if scores_over_thresh.sum() == 0:
                nms_idx = []
            else:
                scores = scores[scores_over_thresh]
                boxes = boxes[scores_over_thresh]
                labels = labels[scores_over_thresh]
                nms_idx = nms(boxes, scores, self.iou_threshold)

            scores_list.append(scores[nms_idx].cpu().numpy())
            labels_list.append(labels[nms_idx].cpu().numpy())
            boxes_list.append(boxes[nms_idx].cpu().numpy())

        return scores_list, labels_list, boxes_list


def demo_detector(
    detector: BatchedDetectorABC,
    dirpath: Path = Path("data/vid/"),
    chunk_size: int = CommonConfig.video_chunk_size,
):
    all_frames = sorted(dirpath.glob("*.jpg"), key=frame_id_from_filename)
    for file_id in tqdm(range(0, len(all_frames), chunk_size)):
        containers = [
            DetectionContainer.from_image_file(file) for file in all_frames[file_id : file_id + chunk_size]
        ]
        containers = detector.predict_containers_list(containers)
        for frame_id, container in enumerate(containers):
            print(len(container.bboxes))

        del containers


if __name__ == "__main__":
    demo_detector(EfficientDetDetector())