from typing import Any, Mapping
from pytorch_lightning import LightningModule
from torch import Tensor
import torchmetrics
import torch
from torch.optim import AdamW


class CorneaTrainerModule(LightningModule):
    def __init__(self, model, criterion, lr):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr

        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "val_IoU": torchmetrics.JaccardIndex(
                    task="binary",
                    num_classes=1,
                ),
                # "val_Dice": torchmetrics.Dice(
                #     num_classes=1,
                # ),
            }
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor | Mapping[str, Any] | None:
        x = batch["image"]
        y = batch["mask"].long()

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        mask = batch["mask"].long()

        logits = self(x)

        loss = self.criterion(logits, mask)
        self.log(
            "val_loss",
            loss.to(self.device),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        logits = torch.sigmoid(logits).squeeze(1)
        self.valid_metrics.update(logits, mask)
        return logits

    def on_validation_epoch_end(self):
        score = self.valid_metrics.compute()
        self.log_dict(score, sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
