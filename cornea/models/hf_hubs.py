from huggingface_hub import PyTorchModelHubMixin
import torch.nn as nn
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi
import torchseg

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


ROOT_HF = "ClementP/cornea-interface-segmentation"


class HuggingFaceModel(PyTorchModelHubMixin, nn.Module):
    def __init__(self, model=None, architecture=None, encoder=None):
        super().__init__()
        if model is not None:
            self.model = model
        else:
            print(f"Creating model {architecture} with encoder {encoder}")
            self.model = torchseg.create_model(
                architecture, encoder, encoder_weights=None, classes=1
            )


def download_model(architecture, encoder):
    model = HuggingFaceModel.from_pretrained(
        f"{ROOT_HF}-{architecture}",
        revision=encoder,
        architecture=architecture,
        encoder=encoder,
    ).model
    return model


def submit_model(model, architecture, encoder):
    model = HuggingFaceModel(model)

    hfapi = HfApi()
    hfapi.create_repo(
        f"{ROOT_HF}-{architecture}", token=HF_TOKEN, exist_ok=True, repo_type="model"
    )

    hfapi.create_branch(
        f"{ROOT_HF}-{architecture}", branch=encoder, token=HF_TOKEN, exist_ok=True
    )

    model.push_to_hub(f"{ROOT_HF}-{architecture}", branch=encoder, token=HF_TOKEN)

    return model
