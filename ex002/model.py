"""
ex002/model.py
2026年 電子情報通信学会ソサイエティ大会 論文 (N-1-43)
「EPICにおけるチャネル分離度と分類精度の関係に関する検討」（室岡・代・神野, 2026）
の再現実験用モデル定義モジュール．

ImageNet事前学習済みResNet34のImagenetteファインチューニングモデル，
EPIC Disentanglement Module，および対象群以外のチャネル出力を0にマスクして
分類精度を評価する機構を定義する．
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImagenetteResNet34(nn.Module):
    """
    ImageNet で事前学習された ResNet-34 を Imagenette (10クラス) 用に構成した分類モデル．
    特徴マップ抽出部 Φ_Θ（layer4 出力, 512チャネル）と最終線形分類層 fc を分離して扱えるようにする．
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet34(weights=weights)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """最終畳み込みブロック出力 Z = Φ_Θ(x) ∈ R^{B×512×H×W} を返す．"""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通常の順伝播により 10 クラスのロジットを返す．"""
        z = self.backbone(x)
        v = torch.flatten(self.avgpool(z), 1)
        return self.fc(v)


class DisentanglementModule(nn.Module):
    """
    EPIC の特徴チャネル分離モジュール（U ∈ R^{512×512}）．
    初期値を単位行列 I_512 とし，直交制約（Cayley変換）または一般可逆行列として学習する．
    """

    def __init__(
        self,
        num_channels: int = 512,
        matrix_type: Literal["orthogonal", "invertible"] = "orthogonal",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.matrix_type = matrix_type

        if matrix_type == "orthogonal":
            linear = nn.Linear(num_channels, num_channels, bias=False)
            nn.init.eye_(linear.weight)
            self.ortho_layer = nn.utils.parametrizations.orthogonal(
                linear, orthogonal_map="cayley"
            )
        elif matrix_type == "invertible":
            self.weight = nn.Parameter(torch.eye(num_channels))
        else:
            raise ValueError(f"未対応の matrix_type です: {matrix_type}")

    def get_matrix(self) -> torch.Tensor:
        """変換行列 U ∈ R^{D×D} を返す．"""
        if self.matrix_type == "orthogonal":
            return self.ortho_layer.weight
        return self.weight

    def get_inverse_matrix(self) -> torch.Tensor:
        """逆変換行列 U^{-1} ∈ R^{D×D} を返す．"""
        U = self.get_matrix()
        if self.matrix_type == "orthogonal":
            return U.t()
        return torch.linalg.inv(U)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """各空間座標の特徴ベクトルに対して U を作用させる（Z_hat = U ⊛ Z）．"""
        U = self.get_matrix()
        return torch.einsum("dc,bchw->bdhw", U, z)


class ChannelMaskedEPICResNet34(nn.Module):
    """
    ファインチューニング済み ImagenetteResNet34 に EPIC を適用し，
    任意のチャネル部分集合（Purity 群 G1〜G5 等）のみを有効化して分類精度を評価可能なモデル．

    論文 N-1-43 2節：
    「分類時には，対象群以外のチャネル出力を 0 にして分類精度を評価した．」
    Global Average Pooling はチャネルごとに線形であるため，Z_hat の対象外チャネルを 0 にすることと
    プーリング後特徴 v = avg_pool(Z_hat) の対象外チャネルを 0 にすることは数学的に厳密に一致する．
    """

    def __init__(
        self,
        finetuned_model: ImagenetteResNet34,
        matrix_type: Literal["orthogonal", "invertible"] = "orthogonal",
    ) -> None:
        super().__init__()
        self.backbone = finetuned_model.backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_channels = finetuned_model.fc.in_features
        self.num_classes = finetuned_model.fc.out_features

        self.register_buffer("orig_weight", finetuned_model.fc.weight.detach().clone())
        self.register_buffer("orig_bias", finetuned_model.fc.bias.detach().clone())

        self.disentangle = DisentanglementModule(
            num_channels=self.num_channels,
            matrix_type=matrix_type,
        )

    def get_modified_fc_weight(self) -> torch.Tensor:
        """逆変換を適用した分類層重み A' = A U^{-1} ∈ R^{10×512} を返す．"""
        return self.orig_weight @ self.disentangle.get_inverse_matrix()

    def forward_from_pooled(
        self,
        v_disentangled: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EPIC 分離・プーリング後の特徴ベクトル v ∈ R^{B×512} に対し，
        channel_mask ∈ {0, 1}^{512} で対象群以外のチャネル出力を 0 にしてロジットを計算する．
        """
        if channel_mask is not None:
            mask = channel_mask.to(device=v_disentangled.device, dtype=v_disentangled.dtype)
            v_masked = v_disentangled * mask.unsqueeze(0)
        else:
            v_masked = v_disentangled

        a_prime = self.get_modified_fc_weight()
        return F.linear(v_masked, a_prime, self.orig_bias)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力画像 x から特徴マップ抽出・EPIC分離・チャネルマスク適用を行い，(logits, Z_hat) を返す．
        """
        with torch.no_grad():
            z = self.backbone(x)
        z_hat = self.disentangle(z)
        if channel_mask is not None:
            mask = channel_mask.to(device=z_hat.device, dtype=z_hat.dtype).view(1, -1, 1, 1)
            z_hat_masked = z_hat * mask
        else:
            z_hat_masked = z_hat

        v = z_hat_masked.mean(dim=(2, 3))
        a_prime = self.get_modified_fc_weight()
        logits = F.linear(v, a_prime, self.orig_bias)
        return logits, z_hat
