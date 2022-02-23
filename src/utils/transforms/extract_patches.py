from albumentations.core.transforms_interface import DualTransform, to_tuple
from albumentations.augmentations.crops import functional as F
import random
import numpy as np
from typing import Tuple, Dict, Any, Union, List

random.seed(42)
# import Aimaug

BboxType = Union[List[int], List[float], Tuple[int, ...], Tuple[float, ...], np.ndarray]

from albumentations.augmentations.crops.functional import crop_bbox_by_coords


def bbox_crop_adj(
    bbox: BboxType,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    rows: int,
    cols: int,
    target_height: int,
    target_width: int,
):
    """Crop a bounding box.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        x_min (int):
        y_min (int):
        x_max (int):
        y_max (int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.

    """
    crop_coords = x_min, y_min, x_max, y_max
    crop_height = min(y_max, rows) - y_min
    crop_width = min(x_max, cols) - x_min
    # return crop_bbox_by_coords(bbox, crop_coords, target_height, target_width, rows, cols)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


class RandomCropNearBBoxFixed(DualTransform):
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        cropping_box_key: str = "cropping_bbox",
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super(RandomCropNearBBoxFixed, self).__init__(always_apply, p)
        if type(target_size) == int:
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.cropping_bbox_key = cropping_box_key
        self.random_crop = False

    def apply(
        self,
        img: np.ndarray,
        x_min: int = 0,
        x_max: int = 0,
        y_min: int = 0,
        y_max: int = 0,
        h_start=0,
        w_start=0,
        **params
    ) -> np.ndarray:
        if self.random_crop:
            return F.random_crop(img, self.target_size[0], self.target_size[1], h_start, w_start)
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        bboxes = params[self.cropping_bbox_key]
        self.radnom_crop = False
        if len(bboxes) == 0:
            self.random_crop = True
            return {"h_start": random.random(), "w_start": random.random()}
        # FIXME tackle bboxes greater than target size
        bbox = random.choice(bboxes)
        target_height, target_width = self.target_size
        height = round(bbox[3] - bbox[1])
        width = round(bbox[2] - bbox[0])

        width_dif = target_width - width
        height_dif = target_height - height
        # max_dim = max(height, width)
        left_pad = random.randint(0, width_dif)
        right_pad = width_dif - left_pad
        top_pad = random.randint(0, height_dif)
        bottom_pad = height_dif - top_pad

        x_min = bbox[0] - left_pad
        x_max = bbox[2] + right_pad

        y_min = bbox[1] - top_pad
        y_max = bbox[3] + bottom_pad

        if x_min < 0:
            x_max -= x_min
        if y_min < 0:
            y_max -= y_min

        # if y_max >

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply_to_bbox(
        self, bbox: Tuple[float, float, float, float], **params
    ) -> Tuple[float, float, float, float]:
        # if self.random_crop:
        #     return F.bbox_random_crop(bbox, self.target_size[0], self.target_size[1], **params)
        return bbox_crop_adj(
            bbox, **params, target_height=self.target_size[0], target_width=self.target_size[1]
        )

    @property
    def targets_as_params(self) -> List[str]:
        return [self.cropping_bbox_key]

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("target_size",)
