from icevision.all import *
import pandas as pd
import icevision


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

class DentalCariesDataModule(pl.LightningDataModule):
    def __init__(self, data_root : str, image_size : int, model_type, batch_size : int = 4, num_workers : int = 4):
        super().__init__()
        self.save_hyperparameters(ignore=['model_type'])
        self.train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=image_size), tfms.A.Normalize()])
        self.valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
        m = icevision
        for mod in model_type.split('.'):
            m = getattr(m, mod)
        self.model_type = m
    
    def setup(self, stage : Optional[str] = None):
        template_record = ObjectDetectionRecord()
        parser = DentalCariesParser(template_record, Path(self.hparams.data_root))
        train_record, valid_record = parser.parse()
        self.train_ds = Dataset(train_record, self.train_tfms)
        self.valid_ds = Dataset(valid_record, self.valid_tfms)

    def train_dataloader(self):
        return self.model_type.train_dl(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self : Optional[str] = None):
        return self.model_type.valid_dl(self.valid_ds, batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers, shuffle=False)
