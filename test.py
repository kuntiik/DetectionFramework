from src.datamodules.CarsDataModule import CarsDataset,CarsDataModule
import pandas as pd
from pathlib import Path
from icevision.all import *
import pandas as pd
import pytorch_lightning as pl




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
            record.set_filepath(self.data_dir / 'images' / o.image)
            record.set_img_size(ImgSize(width=o.image_width, height=o.image_height))
            record.detection.set_class_map(self.class_map)
        if not o.noobject:
            record.detection.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
            record.detection.add_labels(["decay"])

template_record = ObjectDetectionRecord()
parser = DentalCariesParser(template_record, Path("/home/kuntik/carries_dataset"))
train_record, valid_record = parser.parse()
show_record(train_record[2], figsize=(14,10))


image_size = 512
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=600), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
train_ds = Dataset(train_record, train_tfms)
valid_ds = Dataset(valid_record, valid_tfms)

model_type = models.ross.efficientdet
backbone = model_type.backbones.tf_lite0(pretrained=True)
x_args = {'img_size' : image_size}
det_model = model_type.model(backbone=backbone, num_classes=2, **x_args)
train_dl = model_type.train_dl(train_ds, batch_size=8, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=8, num_workers=4, shuffle=False)


class CariesDataModule(pl.LightningDataModule):
    def __init__(self, data_root : str, image_size : int, model_type, batch_size : int = 4, num_workers : int = 4):
        # self.save_hyperparameters(ignore=['model_type'])
        self.train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=600), tfms.A.Normalize()])
        self.valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
        self.model_type = model_type

        self.data_root = data_root
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None):
        template_record = ObjectDetectionRecord()
        # parser = DentalCariesParser(template_record, Path(self.hparams.data_root))
        parser = DentalCariesParser(template_record, Path(self.data_root))
        train_record, valid_record = parser.parse()
        self.train_ds = Dataset(train_record, self.train_tfms)
        self.valid_ds = Dataset(valid_record, self.valid_tfms)

    def train_dataloader(self):
        # return self.model_type.train_dl(self.train_ds, self.hparams.batch_size,num_workers=self.hparams.num_workers, shuffle=True)
        return self.model_type.train_dl(self.train_ds, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # return self.model_type.valid_dl(self.valid_ds, self.hparams.batch_size,num_workers=self.hparams.num_workers, shuffle=False)
        return self.model_type.valid_dl(self.valid_ds, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=False)
# %%
from src.modules.EfficientDetModule import EfficientDetModule
model = EfficientDetModule(det_model, 1e-3)
trainer = pl.Trainer(accelerator='cpu', max_epochs=30, overfit_batches = 1)
dm = CariesDataModule("/home/kuntik/carries_dataset", 512, model_type)
dm.setup()
t=dm.train_dataloader()
trainer.fit(model, dm.train_dataloader())
print("foo")