from torchseg import create_model
from torchseg.losses import DiceLoss
from cornea.data.datamodule import CorneaDatamodule
from cornea.trainer_module import CorneaTrainerModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from cornea.utils.callbacks import LogPredictionSamplesCallback
import os
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    config = {
        "model": "unet",
        "num_classes": 1,
        "batch_size": 24,
        "img_size": (416, 1280),
    }

    root_folder = "/home/clement/Documents/data/Cornea/"
    project_name = "Cornea Interface Segmentation"
    model = create_model(config["model"], classes=config["num_classes"])

    datamodule = CorneaDatamodule(
        root_folder=root_folder,
        img_size=config["img_size"],
        batch_size=config["batch_size"],
    )
    datamodule.prepare_data()

    logger = WandbLogger(name=project_name, config=config)

    if os.environ.get("LOCAL_RANK", None) is None:
        os.environ["WANDB_RUN_NAME"] = logger.experiment.name

    checkpoint_callback = ModelCheckpoint(
        monitor="val_IoU",
        mode="max",
        save_last=True,
        auto_insert_metric_name=True,
        save_top_k=1,
        dirpath=os.path.join("checkpoints", project_name, os.environ["WANDB_RUN_NAME"]),
    )

    loss = DiceLoss(mode="binary")

    module = CorneaTrainerModule(model, loss, lr=0.001)

    trainer = Trainer(
        logger=logger,
        max_epochs=100,
        callbacks=[
            LogPredictionSamplesCallback(
                logger,
            ),
            checkpoint_callback,
        ],
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
