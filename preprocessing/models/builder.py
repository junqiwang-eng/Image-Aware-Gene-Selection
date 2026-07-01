# ------------------------------------------------------------------------------
# Copyright (c) 2007 Free Software Foundation, Inc. https://fsf.org/
# Licensed under the GPL-3.0 License.
# The code is based on CLAM.
# (https://github.com/mahmoodlab/CLAM)
# ------------------------------------------------------------------------------


import os
from functools import partial
import timm

from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import transforms


def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ""
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained

        # check if CONCH_CKPT_PATH is set
        if "CONCH_CKPT_PATH" not in os.environ:
            raise ValueError("CONCH_CKPT_PATH not set")
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ["CONCH_CKPT_PATH"]
    except Exception as e:
        print(e)
        print("CONCH not installed or CONCH_CKPT_PATH not set")
    return HAS_CONCH, CONCH_CKPT_PATH


def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ""
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if "UNI_CKPT_PATH" not in os.environ:
            raise ValueError("UNI_CKPT_PATH not set")
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ["UNI_CKPT_PATH"]
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH


def get_env_path(env_name):
    path = os.environ.get(env_name)
    if path:
        print(f"loading local checkpoint from ${env_name}: {path}")
    return path


def get_state_dict(checkpoint):
    for key in ["state_dict", "model", "model_state_dict"]:
        if isinstance(checkpoint, dict) and key in checkpoint:
            return checkpoint[key]
    return checkpoint


class MidnightEncoder(torch.nn.Module):
    def __init__(self, model_name="kaiko-ai/midnight"):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if os.path.isfile(model_name):
            config_source = os.environ.get("MIDNIGHT_CONFIG_PATH", "kaiko-ai/midnight")
            config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            self.model = AutoModel.from_config(config, trust_remote_code=True)
            if model_name.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(model_name)
            else:
                state_dict = torch.load(model_name, map_location="cpu")
            self.model.load_state_dict(get_state_dict(state_dict), strict=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, x):
        output = self.model(pixel_values=x, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        patch_tokens = output.last_hidden_state[:, 1:].mean(dim=1)
        return torch.cat([cls_token, patch_tokens], dim=-1)


def get_encoder(model_name, target_img_size=224):
    img_transforms = None
    print("loading model checkpoint")
    if model_name == "resnet50_trunc":
        checkpoint_path = get_env_path("RESNET50_TRUNC_CKPT_PATH")
        if checkpoint_path is None:
            print("RESNET50_TRUNC_CKPT_PATH not set; loading timm pretrained weights")
        model = TimmCNNEncoder(checkpoint_path=checkpoint_path)
    elif model_name == "uni_v1":
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, "UNI is not available"
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        checkpoint = torch.load(UNI_CKPT_PATH, map_location="cpu")
        model.load_state_dict(get_state_dict(checkpoint), strict=True)
    elif model_name == "conch_v1":
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        # assert HAS_CONCH, "CONCH is not available"
        from conch.open_clip_custom import create_model_from_pretrained

        if HAS_CONCH:
            model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        else:
            print("CONCH_CKPT_PATH not set; loading CONCH from HuggingFace")
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
            )
        model.forward = partial(
            model.encode_image, proj_contrast=False, normalize=False
        )
    elif model_name == "midnight":
        midnight_model_name = get_env_path("MIDNIGHT_CKPT_PATH")
        if midnight_model_name is None:
            midnight_model_name = os.environ.get("MIDNIGHT_MODEL_NAME", "kaiko-ai/midnight")
            print(f"MIDNIGHT_CKPT_PATH not set; loading Midnight from {midnight_model_name}")
        model = MidnightEncoder(midnight_model_name)
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=constants["mean"],
                    std=constants["std"],
                ),
            ]
        )
    elif model_name == "hopt":
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
    elif model_name == "dinov2":
        model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        data_config = timm.data.resolve_model_data_config(model)
        img_transforms = timm.data.create_transform(**data_config, is_training=False)
    else:
        raise NotImplementedError("model {} not implemented".format(model_name))

    # print(model)  # Uncomment to visualize model structure
    if img_transforms is None:
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(
            mean=constants["mean"],
            std=constants["std"],
            target_img_size=target_img_size,
        )

    return model, img_transforms
