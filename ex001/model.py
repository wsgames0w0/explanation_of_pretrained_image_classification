"""
ex001/model.py
AAAI 2026 論文「EPIC: Explanation of Pretrained Image Classification Networks via Prototypes」
の再現実験用モデル定義モジュール．

事前学習済み画像分類ネットワークを「凍結バックボーン Φ_Θ」「Disentanglement Module (変換行列 U)」
「逆変換を適用した分類層 A' = A U^{-1}」に分解・統合するクラス群を定義する．
"""

from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DenseNetBackbone(nn.Module):
    """DenseNet系の特徴マップ抽出器（最終Global Average Pooling直前のReLU出力までを返す）．"""

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return F.relu(out, inplace=False)


def build_pretrained_backbone_and_head(
    arch_name: str,
) -> Tuple[nn.Module, nn.Module, nn.Linear, int]:
    """
    指定されたアーキテクチャのImageNet事前学習済みモデルを読み込み，
    (1) 凍結バックボーン Φ_Θ, (2) 元モデル全体, (3) 最終線形分類層, (4) チャネル数 D を返す．

    Parameters
    ----------
    arch_name : str
        'resnet18', 'resnet34', 'resnet50', 'densenet121' のいずれか．
    """
    arch = arch_name.lower()
    if arch == "resnet18":
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        fc_layer = base_model.fc
    elif arch == "resnet34":
        base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        fc_layer = base_model.fc
    elif arch == "resnet50":
        # 論文 Table 1 の ResNet-50 (ImageNet ACC 80.8%) に対応する V2 重みを使用
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        fc_layer = base_model.fc
    elif arch == "densenet121":
        base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        backbone = DenseNetBackbone(base_model.features)
        fc_layer = base_model.classifier
    else:
        raise ValueError(f"未対応のアーキテクチャ名です: {arch_name}")

    # 事前学習済みモデルのパラメータはすべて凍結する（EPICはバックボーンを再学習しない）
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    num_channels = fc_layer.in_features
    return backbone, base_model, fc_layer, num_channels


class DisentanglementModule(nn.Module):
    """
    EPICの核となる特徴チャネル分離モジュール（Disentanglement Module）．
    特徴マップ Z ∈ R^{B×D×H×W} の各空間位置 (h, w) に対して可逆変換行列 U ∈ R^{D×D} を作用させる：
        Z_hat = U ⊛ Z
    U = I（単位行列）から学習を開始することで，初期状態では元の事前学習済みモデルと完全に一致する．
    """

    def __init__(
        self,
        num_channels: int,
        matrix_type: Literal["orthogonal", "invertible"] = "orthogonal",
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.matrix_type = matrix_type

        if matrix_type == "orthogonal":
            # Cayley変換による直交行列パラメータ化（初期値 U = I_D，常に U^T U = I_D を満たす）
            linear = nn.Linear(num_channels, num_channels, bias=False)
            nn.init.eye_(linear.weight)
            self.ortho_layer = nn.utils.parametrizations.orthogonal(
                linear, orthogonal_map="cayley"
            )
        elif matrix_type == "invertible":
            # 一般の正則（可逆）行列パラメータ化（初期値 U = I_D）
            self.weight = nn.Parameter(torch.eye(num_channels))
        else:
            raise ValueError(f"matrix_type は 'orthogonal' または 'invertible' を指定してください: {matrix_type}")

    def get_matrix(self) -> torch.Tensor:
        """現在の変換行列 U ∈ R^{D×D} を返す．"""
        if self.matrix_type == "orthogonal":
            return self.ortho_layer.weight
        return self.weight

    def get_inverse_matrix(self) -> torch.Tensor:
        """逆行列 U^{-1} ∈ R^{D×D} を返す．"""
        U = self.get_matrix()
        if self.matrix_type == "orthogonal":
            # 直交行列では U^{-1} = U^T が厳密に成り立つが，汎用性確認のため solve も利用可能
            return U.t()
        return torch.linalg.inv(U)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """各空間座標 (h, w) のチャネルベクトルに対して線形変換 U を適用する（U ⊛ Z）．"""
        U = self.get_matrix()
        return torch.einsum("dc,bchw->bdhw", U, z)


class EPICModel(nn.Module):
    """
    事前学習済みバックボーン Φ_Θ に Disentanglement Module を挿入し，
    分類層重みを A' = A U^{-1} に置換した EPIC モデル（論文 式(1)〜(5) および Remark 3.1）．
    """

    def __init__(
        self,
        backbone: nn.Module,
        fc_layer: nn.Linear,
        matrix_type: Literal["orthogonal", "invertible"] = "orthogonal",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.num_channels = fc_layer.in_features
        self.num_classes = fc_layer.out_features

        # 元の分類層の重み A とバイアス b を凍結バッファとして保持
        self.register_buffer("orig_weight", fc_layer.weight.detach().clone())
        if fc_layer.bias is not None:
            self.register_buffer("orig_bias", fc_layer.bias.detach().clone())
        else:
            self.orig_bias = None

        # 学習対象となる Disentanglement Module のみを保持
        self.disentangle = DisentanglementModule(
            num_channels=self.num_channels,
            matrix_type=matrix_type,
        )

    def get_modified_fc_weight(self) -> torch.Tensor:
        """逆変換を適用した分類層重み A' = A U^{-1} を計算する．"""
        U_inv = self.disentangle.get_inverse_matrix()
        return self.orig_weight @ U_inv

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        入力画像 x に対して (logits, Z, Z_hat, v) を返す．
        - Z     : 元バックボーンの特徴マップ Φ_Θ(x) ∈ R^{B×D×H×W}
        - Z_hat : 分離後の特徴マップ U ⊛ Z ∈ R^{B×D×H×W}
        - v     : Global Average Pooling 後の特徴ベクトル avg_pool(Z_hat) ∈ R^{B×D}
        - logits: A' v + b ∈ R^{B×C}
        """
        with torch.no_grad():
            z = self.backbone(x)
        z_hat = self.disentangle(z)
        v = z_hat.mean(dim=(2, 3))
        a_prime = self.get_modified_fc_weight()
        logits = F.linear(v, a_prime, self.orig_bias)
        return logits, z, z_hat, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """分類ロジット w = (A U^{-1}) avg_pool(U ⊛ Φ_Θ(x)) + b を返す．"""
        logits, _, _, _ = self.forward_features(x)
        return logits

    @torch.no_grad()
    def top_k_contributing_channels(
        self, x: torch.Tensor, k: int = 5, target_class: int | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """
        論文 Algorithm 1 (Top-k Contributing Channels) の実装．
        単一入力画像 x (1×3×H×W) に対し，予測クラス（または指定クラス）への正の寄与度が
        最も大きい上位 k 個のチャネルインデックスとその寄与スコアを返す．

        Returns
        -------
        top_channels : Tensor (k,)
            寄与度上位 k 個のチャネル番号
        top_scores : Tensor (k,)
            対応する寄与スコア (w_pred ⊙ ReLU(v))
        pred_cls : int
            対象クラスインデックス
        z_hat : Tensor (1, D, H, W)
            分離後の特徴マップ
        """
        self.eval()
        logits, _, z_hat, v = self.forward_features(x)
        pred_cls = int(logits[0].argmax().item()) if target_class is None else int(target_class)
        a_prime = self.get_modified_fc_weight()
        w_pred = a_prime[pred_cls]  # (D,)
        # 正のプロトタイプに基づく説明のため ReLU(v) とのアダマール積を計算（Algorithm 1 行9）
        scores = w_pred * F.relu(v[0])  # (D,)
        top_scores, top_channels = torch.topk(scores, k=k)
        return top_channels, top_scores, pred_cls, z_hat
