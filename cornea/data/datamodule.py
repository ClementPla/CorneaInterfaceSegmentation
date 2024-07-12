from pytorch_lightning import LightningDataModule
import os
import torch
import nntools.dataset as D
from nntools.dataset.utils.ops import split

import albumentations as A

from albumentations.pytorch import ToTensorV2
import numpy as np


red = np.asarray([230, 62, 85])
red = red[np.newaxis, np.newaxis, :]


@D.nntools_wrapper
def extract_label(image, mask):
    if mask.shape[-1] == 4:
        mask = mask[..., :3]
    if image.shape[-1] == 4:
        image = image[..., :3]

    gt = np.zeros_like(mask)
    gt = np.abs(mask - red)
    gt = gt < 50
    gt = gt.all(axis=-1)
    argmax = np.argmax(gt, axis=0, keepdims=True)
    _, yy = np.meshgrid(np.arange(gt.shape[1]), np.arange(gt.shape[0]))
    gt = yy < argmax
    return {"image": image, "mask": gt.astype(np.uint8)}


class CorneaDatamodule(LightningDataModule):
    def __init__(self, root_folder, img_size, batch_size, num_workers="auto"):
        super().__init__()
        self.root_folder = root_folder
        self.img_size = img_size
        self.batch_size = batch_size

        if num_workers == "auto":
            self.num_workers = os.cpu_count() // torch.cuda.device_count()

    def prepare_data(self):
        root_img = os.path.join(self.root_folder, "Images")

        root_labels = os.path.join(self.root_folder, "Labels")

        dataset = D.MultiImageDataset(
            {"image": root_img, "mask": root_labels},
            keep_size_ratio=True,
            shape=self.img_size,
        )

        dataset.composer = D.Composition()
        dataset.composer.add(extract_label)

        filenames = dataset.filenames["image"]
        unique_names = ["_".join(f.split("_")[:2]) for f in filenames]

        all_patients = np.unique(unique_names)
        index = np.arange(len(all_patients))
        np.random.shuffle(index)
        train = index[: int(0.8 * len(index))]
        val = index[int(0.8 * len(index)) :]

        train_patients = all_patients[train]
        val_patients = all_patients[val]

        train_idx = [i for i, f in enumerate(unique_names) if f in train_patients]
        val_idx = [i for i, f in enumerate(unique_names) if f in val_patients]

        self.train, self.val = split(dataset, [train_idx, val_idx])

    def setup(self, stage=None):
        self.train.composer.add(
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                ]
            )
        )
        self.train.composer.add(A.Normalize())
        self.train.composer.add(ToTensorV2())

        self.val.composer.add(A.Normalize())
        self.val.composer.add(ToTensorV2())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
