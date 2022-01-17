import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from pycocotools.coco import COCO
from pytorch_lightning import LightningDataModule
from typing import List, Any, Tuple, Optional
from PIL import Image
from pathlib import Path

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

class COCODataset(Dataset):
    def __init__(self, root, annFile, transforms):
        super().__init__()
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transfroms = transforms
        self.category_names = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            self.category_names[cat['id']] = cat['name']

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(Path(self.root)/path).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        targets = self.coco.loadAnns(self.coco.getAnnIds(id))
        boxes = []
        labels = []
        for box in targets:
            # if box["bbox"][2] == 0 or box["bbox"][3] == 0:
            #     #TODO change to log warn
            #     print(f"Warn! bbox with id:{box['id']} is invalid")
            #     continue
            boxes.append(xywh_to_xyxy(box["bbox"]))
            labels.append(box["category_id"])

        return {"boxes": torch.tensor(boxes) if len(boxes) != 0 else torch.empty((0, 4)),
                "labels": torch.tensor(labels).type(torch.int64)}

    def __getitem__(self, index):
        id = self.ids[index]
        img = self._load_image(id)
        target = self._load_target(id)
        if self.transfroms is not None:
            transformed = self.transfroms(image=np.array(
                img), bboxes=target["boxes"], class_labels=target["labels"])
            img = transformed['image']
            target["boxes"] = torch.tensor(transformed["bboxes"]) if len(
                transformed["bboxes"]) != 0 else torch.empty((0, 4))
            target["labels"] = torch.tensor(
                transformed["class_labels"]).type(torch.int64)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


class COCODataModule(LightningDataModule):
    def __init__(self, data_root_train: str, ann_file_train: str, data_root_val : str = None, ann_file_val : str = None,
                    batch_size: int = 8, num_workers: int = 8, seed: int = 42, pin_memory : bool = True,
                    data_split: Tuple[int, int, int] = [113_287, 5000, 0], train_transforms=None, val_transforms=None, test_transforms=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.ds_normalize = {"mean" : (0.485, 0.456, 0.406), "std" : (0.229, 0.224, 0.225)}

        if self.hparams.train_transforms == None:
            self.hparams.train_transforms = self.default_transforms()
        if self.hparams.val_transforms == None:
            self.hparams.val_transforms = self.default_transforms()
        if self.hparams.test_transforms == None:
            self.hparams.test_transforms = self.default_transforms()
        

    @property
    def num_classes(self) -> int:
        return 90
    
    @staticmethod
    def collate(batch):
        return tuple(zip(*batch))

    def setup(self, stage: Optional[str] = None):
        #TODO modify for val_set size different from 5000 and nonzero test set 
        if self.hparams.data_root_val != None and self.hparams.ann_file_val != None:
            self.train_dataset = COCODataset(self.hparams.data_root_train, self.hparams.ann_file_train, self.hparams.train_transforms)
            self.val_dataset = COCODataset(self.hparams.data_root_val, self.hparams.ann_file_val, self.hparams.val_transforms)

        else:
            self.train_dataset, _, _ = random_split(COCODataset(self.hparams.data_root_train, self.hparams.ann_file_train, self.hparams.train_transforms),
                                                    lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
            _, self.val_dataset, _ = random_split(COCODataset(self.hparams.data_root_train, self.hparams.ann_file_train, self.hparams.val_transforms),
                                                    lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))
            _, _, self.test_dataset = random_split(COCODataset(self.hparams.data_root_train, self.hparams.ann_file_train, self.hparams.test_transforms),
                                                lengths=self.hparams.data_split, generator=torch.Generator().manual_seed(self.hparams.seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=True, collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size,num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, shuffle=False, collate_fn=self.collate)
    
    def default_transforms(self):
        default_t = A.Compose([
            A.Resize(width=480, height=480),
            A.Normalize(mean=self.ds_normalize['mean'], std=self.ds_normalize['std']),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.4, min_area=5))
        return default_t