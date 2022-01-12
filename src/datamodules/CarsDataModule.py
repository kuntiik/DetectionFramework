import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Tuple, Optional
from pathlib import Path



class CarsDataset(Dataset):
    def __init__(self, img_root : str ="/home.stud/kuntluka/dataset/cars/data/training_images", ann_file : str= "/home.stud/kuntluka/dataset/cars/data/train_solution_bounding_boxes.csv", transforms=None):
        super().__init__()
        self.root = Path(img_root)
        self.df = pd.read_csv(ann_file, index_col='image')
        self.ids = [_.name for _ in self.root.iterdir()]
        self.ids_with_box = self.df.index.unique().values
        self.transforms = transforms
        self.num_cat = 1

        self.imgs = self.df.index.unique().tolist()

    def __getitem__(self, index):
        # img = self.imgs[index]
        img = Image.open(self.root / self.imgs[index])
        if self.ids[index] in self.ids_with_box:
            target = self.df.loc[self.ids[index]].values
        else:
            target = np.array([])

        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)
        n_boxes = np.size(target, 0)
        if np.size(target, 1) == 0:
            n_boxes = 0
            target = []

        if self.transforms != None:
            labels = [1 for _ in range(n_boxes)]
            transformed = self.transforms(image=np.array(
                img), bboxes=target, class_labels=labels)
            target = {}
            img = transformed['image']
            target["boxes"] = torch.tensor(transformed["bboxes"]) if len(
                transformed["bboxes"]) != 0 else torch.empty((0, 4))
            target["labels"] = torch.tensor(
                transformed["class_labels"]).type(torch.int64)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)

class CarsDataModule(LightningDataModule):

    def __init__(self, data_root: str =  "/home.stud/kuntluka/dataset/cars/data/training_images",
              ann_file: str= "/home.stud/kuntluka/dataset/cars/data/train_solution_bounding_boxes.csv",
              batch_size: int = 4, num_workers: int = 8, seed: int = 42, pin_memory : bool = True,
              data_split: Tuple[int, int, int] = [851, 150, 0], train_transforms=None, val_transforms=None, test_transforms=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.ds_normalize = {"mean" : (0.2432, 0.3255, 0.3522), "std" : (0.2186, 0.2592, 0.3083)}
        if self.hparams.train_transforms == None:
            self.hparams.train_transforms = self.default_transforms()
        if self.hparams.val_transforms == None:
            self.hparams.val_transforms = self.default_transforms()
        if self.hparams.test_transforms == None:
            self.hparams.test_transforms = self.default_transforms()
        

    @property
    def num_classes(self) -> int:
        return 1
    
    @staticmethod
    def collate(batch):
        return tuple(zip(*batch))
    
    @staticmethod
    def collate_effdet(batch):
        images, targets = tuple(zip(*batch))
        images = torch.stack(images).float()
        image_size = []
        for image in images:
            _, y, x = image.shape
            image_size.append((y,x))

        boxes = [target['boxes'][:,[1,0,3,2]].float() for target in targets]

        labels = [target['labels'].float() for target in targets]
        img_scale = torch.ones((1, len(labels))).float().flatten()


        annotations = {
            "bbox" : boxes,
            "cls" : labels,
            "img_size" : torch.tensor(image_size).float(),
            "img_scale" : img_scale
        }

        return images, annotations


    def setup(self, stage: Optional[str] = None):
        self.train_dataset, _, _ = random_split(CarsDataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.train_transforms),
                                                lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
        _, self.val_dataset, _ = random_split(CarsDataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.val_transforms),
                                              lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
        _, _, self.test_dataset = random_split(CarsDataset(self.hparams.data_root, self.hparams.ann_file, self.hparams.test_transforms),
                                               lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate_effdet)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate_effdet)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate)
    
    def default_transforms(self):
        cars_transforms = A.Compose([
            # A.Resize(width = 512, height=512),
            # A.Normalize(mean=self.ds_normalize['mean'], std=self.ds_normalize['std']),
            # ToTensorV2()
            A.Resize(height=512, width=512, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.0, min_area=0))
        return cars_transforms