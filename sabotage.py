from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from utils import frame_id_from_filename, VideoWriter, SabotageConfig, DetectionContainer


class DescriptorHistory:
    """Keeps track of mean distances and descriptors
    Args:
        iters_to_keep: how many iters to keep in history
    """

    def __init__(self, iters_to_keep: int = 50):
        self.iters_to_keep = iters_to_keep
        self.score_history = []  # mean dists
        self.des_history = []  # descriptor history
        self.counter = 0

    def update_history(self, mean_dist, des):
        if self.counter >= self.iters_to_keep:
            self.score_history.pop(0)
            self.des_history.pop(0)
        self.score_history.append(mean_dist)
        self.des_history.append(des)
        self.counter += 1

    def clear_history(self):
        self.score_history = []
        self.des_history = []
        self.counter = 0

    def get_score_history_mean(self):
        return np.mean(self.score_history)

    """
    All methods below implement logic for adopting to new environment.
    We can either adopt to descriptor with max distance, or with min distance, or with closest to mean
    """

    def get_index_with_lowest_score(self):
        return self.score_history.index(min(self.score_history))

    def get_index_with_highest_score(self):
        return self.score_history.index(max(self.score_history))

    def get_index_with_closest_to_mean_score(self):
        closest_to_mean_val = min(self.score_history, key=lambda x: abs(x - self.get_score_history_mean()))
        return self.score_history.index(closest_to_mean_val)

    def get_lowest_score(self):
        return self.score_history[self.get_index_with_lowest_score()]

    def get_highest_score(self):
        return self.score_history[self.get_index_with_highest_score()]

    def get_closest_to_mean_score(self):
        return self.score_history[self.get_index_with_closest_to_mean_score()]

    def get_lowest_des(self):
        return self.des_history[self.get_index_with_lowest_score()]

    def get_closest_to_mean_des(self):
        return self.des_history[self.get_index_with_closest_to_mean_score()]

    def get_binarized_score(self, thresh: float) -> List[int]:
        mask = np.array(self.score_history) > thresh
        return list(mask.astype(int))


class SabotageAlarmUnit:
    """Responsible for making decision regarding presence of sabotage
    Args:
        max_risk_percent: max percent of risks in history to call it sabotage
        min_size_to_decide: min size of history to be able to call it sabotage
    """

    def __init__(self, max_risk_percent: float = 0.8, min_size_to_decide: int = 10):
        self.max_risk_percent = max_risk_percent
        self.min_size_to_decide = min_size_to_decide

    def is_sabotage(self, binarized_history: List[int]):
        """
        binarized_history: binarized history of mean dists where 0 is below threshold, 1 is above.
        """
        if len(binarized_history) < self.min_size_to_decide:
            return False
        max_absolute_risks = int(self.max_risk_percent * len(binarized_history))
        num_risks = sum(binarized_history)
        return True if num_risks >= max_absolute_risks else False


class Sabotage:
    """Implements entire logic for deciding if current state is corrupted
    Args:
        top_n_matches: number of matched points to calculate distance
        dist_thresh: threshold above which we call Risk of sabotage
        lower_boundary: threshold which tells us we have to change reference
        min_kp_to_match: min number of key points in an image to start comparing it with reference
        unmatched_value: value to fill in if min_kp_to_match is not reached
        soft_thresh: temporary distance threshold when current status is sabotage (introduced to adapt to new changes)
        percent_to_calc_soft_thresh: which percent of history is taken to calculate if soft_thresh can be applied
        write_risk: if True, Risk of sabotage is displayed. No sabotage otherwise
    """

    def __init__(
        self,
        top_n_matches: int = 50,
        dist_thresh: float = 5.0,
        lower_boundary: float = 0.1,
        min_kp_to_match: int = 1,
        unmatched_value: float = 100.0,
        soft_thresh: float = 7.0,
        percent_to_calc_soft_thresh: float = 0.40,
        iters_to_keep: int = 25,
        max_risk_percent: float = 0.8,
        min_size_to_decide: int = 1,
        write_risk: bool = False,
    ):
        assert dist_thresh > lower_boundary, "threshold must be greater than lower boundary"
        assert soft_thresh > dist_thresh, "soft threshold must be greater than basic one"
        self.top_n_matches = top_n_matches
        self.dist_thresh = dist_thresh
        self.lower_boundary = lower_boundary
        self.min_kp_to_match = min_kp_to_match
        self.unmatched_value = unmatched_value
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_des = None
        self.soft_thresh = soft_thresh
        self.percent_to_calc_soft_thresh = percent_to_calc_soft_thresh
        self.write_risk = write_risk
        self.history = DescriptorHistory(iters_to_keep=iters_to_keep)
        self.alarm_unit = SabotageAlarmUnit(max_risk_percent=max_risk_percent, min_size_to_decide=min_size_to_decide)

    def calc_reference_des(self, container: DetectionContainer):
        ref_img = cv2.cvtColor(container.image, code=cv2.COLOR_RGB2GRAY)
        try:
            ref_kp, self.reference_des = self.orb.detectAndCompute(ref_img, None)
        except cv2.error:
            print("Probably, no keypoints were found in the reference image")
            raise ValueError
        if len(ref_kp) < self.min_kp_to_match:
            print(
                "Attention: number of keypoint in reference image is lower than the required number of key points. \
                 The results might be incorrect"
            )

    def mean_dist(self, matches: List[cv2.DMatch]) -> float:
        """ Calculates mean distance between images and reference"""
        top_n = len(matches) if self.top_n_matches >= len(matches) else self.top_n_matches
        if len(matches) < self.min_kp_to_match:
            return self.unmatched_value
        av = sum([matches[i].distance for i in range(top_n)]) / top_n
        return av

    def check_for_soft_thresh(self):
        """ Checks if we can apply soft threshold """
        last_iters = int(len(self.history.score_history) * self.percent_to_calc_soft_thresh)
        recent_history_mean = np.mean(self.history.score_history[last_iters:])
        old_history_mean = np.mean(self.history.score_history[:last_iters])
        return True if recent_history_mean < old_history_mean else False

    def process_container(self, container: DetectionContainer) -> DetectionContainer:
        thresh_to_comp = self.dist_thresh
        current_img = cv2.cvtColor(container.image, code=cv2.COLOR_RGB2GRAY)
        cur_kp, cur_des = self.orb.detectAndCompute(current_img, None)
        if len(cur_kp) == 0:
            mean_top_matches = self.unmatched_value
        else:
            matches = self.bf.match(self.reference_des, cur_des)
            matches = sorted(matches, key=lambda x: x.distance)
            mean_top_matches = self.mean_dist(matches)
        self.history.update_history(mean_dist=mean_top_matches, des=cur_des)
        bin_score = self.history.get_binarized_score(thresh=self.dist_thresh)
        is_sabotage = self.alarm_unit.is_sabotage(bin_score)
        if is_sabotage:
            container.sabotage_status = SabotageConfig.class_to_label["Sabotage"]
            if self.check_for_soft_thresh():
                # if curr status is sabotage but we're contemplating positive changes, make new temporary threshold
                thresh_to_comp = self.soft_thresh
        elif mean_top_matches > self.dist_thresh:
            # if mean dists exceeded threshold but not enough to be sabotage, call it "Risk"
            container.sabotage_status = (
                SabotageConfig.class_to_label["Risk of sabotage"]
                if self.write_risk
                else SabotageConfig.class_to_label["No sabotage"]
            )
        else:
            container.sabotage_status = SabotageConfig.class_to_label["No sabotage"]
        if self.lower_boundary < self.history.get_score_history_mean() < thresh_to_comp:
            # if current mean dists are between lower and upper boundaries, adopt to new changes by taking new reference
            self.reference_des = self.history.get_lowest_des()
            self.history.clear_history()
        return container

    def process_container_list(self, cont_list: List[DetectionContainer]) -> List[DetectionContainer]:
        for container in cont_list:
            self.process_container(container)
        return cont_list


def demo_sabotage(
    dirpath: Path = Path("data/vid/"),
    ref_img_path: Path = Path("data/vid/vid_1.jpg"),
    start: int = 0,
    end: int = 5000,
    step: int = 5,
    chunk_size: int = 50,
):
    sabotage = Sabotage(
        top_n_matches=50,
        dist_thresh=6.5,
        lower_boundary=4.5,
        min_kp_to_match=1,
        unmatched_value=100.0,
        soft_thresh=7.5,
        percent_to_calc_soft_thresh=0.45,
        iters_to_keep=50,
        write_risk=False,
    )
    ref_cont = DetectionContainer.from_image_file(ref_img_path)
    sabotage.calc_reference_des(ref_cont)
    all_frames = sorted(dirpath.glob("*.jpg"), key=frame_id_from_filename)[start:end:step]

    for file_id in tqdm(range(0, len(all_frames), chunk_size)):
        containers = [
            DetectionContainer.from_image_file(file) for file in all_frames[file_id : file_id + chunk_size]
        ]
        containers = sabotage.process_container_list(containers)
        for frame_id, container in enumerate(containers):
            if container.sabotage_status == 2:
                print(frame_id)

        del containers


if __name__ == "__main__":
    demo_sabotage(
        dirpath=Path("data/vid/"),
        ref_img_path=Path("data/vid/vid_1.jpg"),
        start=0,
        end=10,
        step=1,
        chunk_size=50,
    )