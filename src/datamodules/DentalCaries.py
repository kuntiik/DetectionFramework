from icevision.all import *
import pandas as pd
import icevision
from icevision.data.data_splitter import RandomSplitter
import albumentations as A


class DentalCariesParser(Parser):
    def __init__(self, template_record, data_dir):
        super().__init__(template_record=template_record)
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir / "annotations.csv")
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
        batch_size: int = 4,
        num_workers: int = 4,
        seed=777,
        transforms=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_type", "trasnforms"])
        self.ds_normalize = {"mean": (0.3669, 0.3669, 0.3669), "std": (0.2768, 0.2768, 0.2768)}
        self.valid_tfms = tfms.A.Adapter(self.default_transforms(image_size))

        # t = tfms.A.Adapter(
        #     [
        #     ]
        # )
        if transforms is not None:
            for transform in self.default_transforms(image_size):
                transforms.append(transform)
            self.train_tfms = tfms.A.Adapter(transforms)
        else:
            self.train_tfms = self.valid_tfms

        m = icevision
        for mod in model_type.split("."):
            m = getattr(m, mod)
        self.model_type = m

    def setup(self, stage: Optional[str] = None):
        template_record = ObjectDetectionRecord()
        parser = DentalCariesParser(template_record, Path(self.hparams.data_root))
        train_record, valid_record = parser.parse(
            data_splitter=RandomSplitter([0.8, 0.2], seed=self.hparams.seed)
        )

        self.train_ds = Dataset(train_record, self.train_tfms)
        self.valid_ds = Dataset(valid_record, self.valid_tfms)
        # self.train_ds_valid_tfms = Dataset(train_record, self.valid_tfms)

    def train_dataloader(self):
        return self.model_type.train_dl(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self: Optional[str] = None):
        return self.model_type.valid_dl(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def default_transforms(self, image_size):
        return [
            *tfms.A.resize_and_pad(image_size),
            A.Normalize(mean=self.ds_normalize["mean"], std=self.ds_normalize["std"]),
        ]

    # def test_dataloader(self):
    #     return self.model_type.valid_dl(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
