import pytorch_lightning as pl
import torch
import torch.nn as nn


# 1. 独立 Pearson 损失类：先算每个基因在 batch 上的 pcc，再全基因求平均
class PearsonLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 沿 batch 维度（dim=0）做中心化
        y_pred_centered = y_pred - torch.mean(y_pred, dim=0, keepdim=True)
        y_true_centered = y_true - torch.mean(y_true, dim=0, keepdim=True)

        # 分子与分母计算 (dim=0 确保顺着纵向 Batch 轴计算)
        r_num = torch.sum(y_pred_centered * y_true_centered, dim=0)
        r_den = torch.sqrt(
            torch.sum(y_pred_centered**2, dim=0)
            * torch.sum(y_true_centered**2, dim=0)
            + 1e-8
        )

        r = r_num / r_den
        r = torch.nan_to_num(r, nan=0.0)  # 安全替换 NaN 值为 0

        # 返回 (1 - 平均PCC) 作为需要最小化的 Loss 标量
        return torch.mean(1.0 - r)


# 2. 针对 512 维优化的稳健 1 层隐藏层 MLP 模型
class FFlayer(nn.Module):

    def __init__(
        self, in_features=512, out_features=250, hidden_dims=[384], dropout_p=0.2
    ):
        super().__init__()
        layers = []
        current_dim = in_features
        for layer_idx, h_dim in enumerate(hidden_dims):
            if current_dim is None and layer_idx == 0:
                layers.append(nn.LazyLinear(h_dim))
            else:
                layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_p))
            current_dim = h_dim
        if current_dim is None:
            layers.append(nn.LazyLinear(out_features))
        else:
            layers.append(nn.Linear(current_dim, out_features))
        self.f = nn.Sequential(*layers)

    def forward(self, inputs):
        return {"gene_pred": self.f(inputs["img_feat"])}


class LinearModel(nn.Module):

    def __init__(self, in_features=512, out_features=250):
        super().__init__()
        if in_features is None:
            self.fc = nn.LazyLinear(out_features)
        else:
            self.fc = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        return {"gene_pred": self.fc(inputs["img_feat"])}


_model_dict = {"linear": LinearModel, "ff": FFlayer}


# 3. Lightning Module 主体：完全移除 val_step 干净清爽
class STPred(pl.LightningModule):

    def __init__(self, model_key, model_params={}, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = _model_dict[model_key](**model_params)
        self.lr = lr

        # 仅保留纯 PCC 损失
        self.pcc_fn = PearsonLoss()

        self.test_outputs = []  # 缓存测试集结果的容器

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        y_pred = out["gene_pred"]
        y_true = batch["exp"]

        # 直接计算纯 PCC 损失
        pcc_loss = self.pcc_fn(y_pred, y_true)

        # 记录日志
        self.log("train_loss", pcc_loss, on_epoch=True, prog_bar=True)
        return pcc_loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        y_pred = out["gene_pred"]
        y_true = batch["exp"]

        pcc_loss = self.pcc_fn(y_pred, y_true)

        self.log("test_pcc", 1 - pcc_loss)
        # 缓存给外部 train.py 做最后的多折平均值排序
        self.test_outputs.append(
            {"pred": y_pred.detach().cpu(), "true": y_true.detach().cpu()}
        )
        return pcc_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
