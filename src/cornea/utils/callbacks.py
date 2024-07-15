from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import wandb


class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, n_images=8, frequency=10):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency
        self.__call = 0

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            batch_idx < 1
            and trainer.is_global_zero
            and (self.__call % self.frequency == 0)
        ):
            n = self.n_images
            x = batch["image"][:n].float()
            y = batch["mask"][:n]
            prob = outputs
            pred = prob > 0.5

            columns = ["image"]
            class_labels = {i: name for i, name in enumerate(["Lower", "Upper"])}

            data = [
                [
                    wandb.Image(
                        x_i,
                        masks={
                            "Prediction": {
                                "mask_data": p_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                            "Groundtruth": {
                                "mask_data": y_i.cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    )
                ]
                for x_i, y_i, p_i in list(zip(x, y, pred))
            ]
            self.wandb_logger.log_table(
                data=data, key=f"Validation Batch {batch_idx}", columns=columns
            )
        self.__call += 1
