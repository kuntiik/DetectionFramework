import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Tuple, Optional

from src.datamodules.coco_datamodule import COCODataset

class DentalCarriesDataModule(LightningDataModule):
    def __init__(self, data_root: str, ann_file: str, batch_size: int = 8, num_workers: int = 8, seed: int = 42, pin_memory : bool = True,
                 data_split: Tuple[int, int, int] = [1400, 226, 0], train_transforms=None, val_transforms=None, test_transforms=None,
                 skip_transforms : bool = False):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.ds_normalize = {"mean" : (0.3669, 0.3669, 0.3669), "std" : (0.2768, 0.2768, 0.2768)}

        if self.hparams.train_transforms == None and not skip_transforms:
            self.hparams.train_transforms = self.default_transforms()
        if self.hparams.val_transforms == None and not skip_transforms:
            self.hparams.val_transforms = self.default_transforms()
        if self.hparams.test_transforms == None and not skip_transforms:
            self.hparams.test_transforms = self.default_transforms()
        

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def name(self) -> str:
        return "dental_caries" 
    
    @staticmethod
    def collate(batch):
        return tuple(zip(*batch))

    def setup(self, stage: Optional[str] = None):
        self.train_dataset, _, _ = random_split(COCODataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.train_transforms),
                                                lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
        _, self.val_dataset, _ = random_split(COCODataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.val_transforms),
                                              lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
        _, _, self.test_dataset = random_split(COCODataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.test_transforms),
                                               lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate)
    
    def default_transforms(self):
        cars_transforms = A.Compose([
            A.Resize(width=1014, height=768),
            A.Normalize(mean=self.ds_normalize['mean'], std=self.ds_normalize['std']),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.4, min_area=5))
        return cars_transforms