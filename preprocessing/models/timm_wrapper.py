# ------------------------------------------------------------------------------
# Copyright (c) 2007 Free Software Foundation, Inc. https://fsf.org/
# Licensed under the GPL-3.0 License.
# The code is based on CLAM.
# (https://github.com/mahmoodlab/CLAM)
# ------------------------------------------------------------------------------

import torch
import timm


class TimmCNNEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50.tv_in1k",
        kwargs: dict = {
            "features_only": True,
            "out_indices": (3,),
            "pretrained": True,
            "num_classes": 0,
        },
        pool: bool = True,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        if checkpoint_path is not None:
            kwargs = {**kwargs, "pretrained": False}
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint
            for key in ["state_dict", "model", "model_state_dict"]:
                if isinstance(checkpoint, dict) and key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            state_dict = {
                key.removeprefix("module.").removeprefix("model."): value
                for key, value in state_dict.items()
            }
            self.model.load_state_dict(state_dict, strict=False)
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
