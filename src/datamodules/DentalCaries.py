from icevision.all import *
import pandas as pd
import icevision
from icevision.data.data_splitter import RandomSplitter
import albumentations as A

import torch


@dataclass
class CollectOp:
    fn: Callable
    order: float = 0.5


# return image width height only (this information is lost after transforms)
class WidthHeightDataset:
    def __init__(
        self, records: List[dict],
    ):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i].load()
        return record.img_size.height, record.img_size.width

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"


def setup_bboxes_extended(self, record_component):
    self.adapter._compose_kwargs["bbox_params"] = A.BboxParams(
        format="pascal_voc", label_fields=["labels"]
    )

    self.adapter._albu_in["bboxes"] = [o.xyxy for o in record_component.bboxes]
    self.adapter._albu_in["cropping_bbox"] = [o.xyxy for o in record_component.bboxes]

    self.adapter._collect_ops.append(CollectOp(self.collect))


class RandomDatasetSampler(DataSplitter):
    def __init__(self, n, seed: int = None) -> None:
        self.n = n
        self.seed = seed

    def split(self, records: Sequence[BaseRecord]):

        with np_local_seed(self.seed):
            shuffled = np.random.permutation([record.record_id for record in records])
        return np.split(shuffled, [self.n])[0]


class RandomSplitterLimited(DataSplitter):
    def __init__(self, probs: Sequence[int], seed: int = None, n: int = None):
        self.probs = probs
        self.seed = seed
        self.n = n

    def split(self, records: Sequence[BaseRecord]):
        p = np.array(self.probs) * len(records)  # convert percentage to absolute
        p = np.ceil(p).astype(int)  # round up, so each split has at least one example
        p[p.argmax()] -= sum(p) - len(records)  # removes excess from split with most items
        p = np.cumsum(p)

        with np_local_seed(self.seed):
            shuffled = np.random.permutation([record.record_id for record in records])

        train, val = np.split(shuffled, p.tolist())[:-1]
        if self.n is not None and len(train) > self.n:
            train = train[: self.n]
        return [train, val]


class DentalCariesParser(Parser):
    def __init__(self, template_record, data_dir):
        super().__init__(template_record=template_record)
        self.data_dir = data_dir
        # TODO change this if needed
        self.df = pd.read_csv(data_dir / "annotations2.csv")
        self.class_map = ClassMap(["decay"])

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o

    def __len__(self) -> int:
        return len(self.df)

    def record_id(self, o) -> Hashable:
        return o.image

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.data_dir / "images" / o.image)
            record.set_img_size(ImgSize(width=o.image_width, height=o.image_height))
            record.detection.set_class_map(self.class_map)
        if not o.noobject:
            record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
            record.detection.add_labels(["decay"])


class DentalCariesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        image_size: int,
        model_type,
        ann_file: str = "annotations.json",
        batch_size: int = 4,
        num_workers: int = 4,
        seed=777,
        limit_train_samples=None,
        transforms=None,
        preprocess_val=0,
        train_shuffle=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_type", "trasnforms"])
        self.ds_normalize = {"mean": (0.3669, 0.3669, 0.3669), "std": (0.2768, 0.2768, 0.2768)}
        if preprocess_val == 0:
            self.valid_tfms = tfms.A.Adapter(self.default_transforms(image_size))
        else:
            self.valid_tfms = tfms.A.Adapter(
                transforms[:preprocess_val] + self.default_transforms(image_size)
            )

        if transforms is not None:
            for transform in self.default_transforms(image_size):
                transforms.append(transform)
            self.train_tfms = tfms.A.Adapter(transforms)
        else:
            self.train_tfms = self.valid_tfms

        # hack to be able to pass module as argument - python converts it to string
        m = icevision
        for mod in model_type.split("."):
            m = getattr(m, mod)
        self.model_type = m

    def setup(self, stage: Optional[str] = None):
        template_record = ObjectDetectionRecord()
        parser = DentalCariesParser(template_record, Path(self.hparams.data_root))
        train_record, valid_record = parser.parse(
            data_splitter=RandomSplitterLimited(
                [0.8, 0.2], seed=self.hparams.seed, n=self.hparams.limit_train_samples
            )
        )

        icevision.tfms.A.AlbumentationsBBoxesComponent.setup_bboxes = setup_bboxes_extended
        self.train_ds = Dataset(train_record, self.train_tfms)
        self.valid_ds = Dataset(valid_record, self.valid_tfms)
        self.debug_ds = Dataset(valid_record, None)

        # Dataset returning width and height for image necessary for inverse transformation
        self.train_hw_info = WidthHeightDataset(train_record)
        self.valid_hw_info = WidthHeightDataset(valid_record)

    def train_dataloader(self):
        return self.model_type.train_dl(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True if self.hparams.train_shuffle else False,
        )

    def val_dataloader(self: Optional[str] = None):
        return self.model_type.valid_dl(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self, stage="val"):
        if stage == "val":
            return self.val_dataloader()
        else:
            return self.train_dataloader()

    def default_transforms(self, image_size):
        return [
            *tfms.A.resize_and_pad(image_size),
            A.Normalize(mean=self.ds_normalize["mean"], std=self.ds_normalize["std"]),
        ]

