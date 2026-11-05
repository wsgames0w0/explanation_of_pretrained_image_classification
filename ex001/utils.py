"""
ex001/utils.py
AAAI 2026 論文「EPIC: Explanation of Pretrained Image Classification Networks via Prototypes」
の再現実験に使用するデータ読み込み・特徴抽出・EPIC学習・可視化・評価ユーティリティ関数群．
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm

from model import EPICModel


# Imagenette の10クラスと ImageNet-1K クラスインデックスの厳密な対応表
IMAGENETTE_WNID_TO_IMAGENET1K: Dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENETTE_CLASS_NAMES_EN: List[str] = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

IMAGENETTE_CLASS_NAMES_JA: List[str] = [
    "テンチ(淡水魚)",
    "スプリンガー犬",
    "カセットプレーヤー",
    "チェーンソー",
    "教会",
    "フレンチホルン",
    "ごみ収集車",
    "ガソリン計量機",
    "ゴルフボール",
    "パラシュート",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def setup_japanese_matplotlib() -> None:
    """Matplotlib で日本語フォントとマイナス記号が正しく描画されるよう設定する．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def get_eval_transforms() -> Tuple[T.Compose, T.Compose]:
    """
    (1) モデル入力用の正規化付き変換 と (2) 可視化用の正規化なし(0〜1)変換 を返す．
    """
    norm_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    vis_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )
    return norm_tf, vis_tf


def load_imagenette_datasets(
    dataset_root: str | Path,
) -> Tuple[Dataset, Dataset, Dataset, Dataset, List[int]]:
    """
    Imagenette (320px) データセットを読み込み，
    (train_norm, val_norm, train_vis, val_vis, imagenet1k_indices) を返す．
    """
    root = Path(dataset_root)
    norm_tf, vis_tf = get_eval_transforms()

    train_norm = datasets.Imagenette(root, split="train", size="320px", transform=norm_tf, download=False)
    val_norm = datasets.Imagenette(root, split="val", size="320px", transform=norm_tf, download=False)
    train_vis = datasets.Imagenette(root, split="train", size="320px", transform=vis_tf, download=False)
    val_vis = datasets.Imagenette(root, split="val", size="320px", transform=vis_tf, download=False)

    # wnids の順序から ImageNet-1K インデックスリストを構築
    wnids = getattr(train_norm, "wnids", list(IMAGENETTE_WNID_TO_IMAGENET1K.keys()))
    imagenet1k_indices = [IMAGENETTE_WNID_TO_IMAGENET1K[w] for w in wnids]
    return train_norm, val_norm, train_vis, val_vis, imagenet1k_indices


@torch.no_grad()
def extract_backbone_features(
    backbone: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float32,
    desc: str = "特徴マップ抽出",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    凍結されたバックボーン Φ_Θ を用いて全画像の特徴マップ Z = Φ_Θ(I) を一括抽出する．
    EPIC ではバックボーンが完全に固定されるため，特徴マップを事前抽出することで
    Disentanglement Module の学習を大幅に高速化できる．
    """
    backbone.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(dtype == torch.float16)):
            z = backbone(x)
        feats_list.append(z.to(dtype=dtype).cpu())
        labels_list.append(y.cpu())

    return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def select_positive_prototypes(
    spatial_sums_orig: torch.Tensor,
    U: torch.Tensor,
    m_proto: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    論文 3節「Prototypes of a Feature Map Channel」に基づき，各チャネル k について
    空間和 activ(Z_hat; k) = Σ_{h,w} Z_hat[k, h, w] が上位 m_proto 件となる訓練画像のインデックスを選出する．

    線形性 Σ_{h,w} (U Z)[k, h, w] = (U Σ_{h,w} Z)_k を利用して厳密かつ高速に算出する．

    Parameters
    ----------
    spatial_sums_orig : Tensor (N, D)
        元特徴マップの空間和 Σ_{h,w} Z_I[:, h, w]
    U : Tensor (D, D)
        現在の変換行列
    m_proto : int
        各チャネルあたり選出する正のプロトタイプ枚数 m

    Returns
    -------
    top_vals : Tensor (m_proto, D)
    top_indices : Tensor (m_proto, D)
    """
    activ = spatial_sums_orig @ U.t()  # (N, D)
    top_vals, top_indices = torch.topk(activ, k=m_proto, dim=0)
    return top_vals, top_indices


@torch.no_grad()
def compute_all_channel_purities(
    z_train: torch.Tensor,
    U: torch.Tensor,
    m_proto: int = 5,
    device: torch.device = torch.device("cuda"),
    chunk_channels: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    全 D チャネルについて，上位 m_proto 枚の正のプロトタイプ画像における平均 Purity を算出する．
    論文 3節「Purity of Prototype」の定義：
        (h*, w*) = argmax_{x, y} Z_hat[k, x, y]
        p = ReLU(Z_hat[:, h*, w*]) ∈ R^D_+
        purity(I, k) = p_k / ||p||_2 ∈ [0, 1]

    Returns
    -------
    mean_purity_per_channel : Tensor (D,)
        各チャネルの平均 Purity
    top_indices : Tensor (m_proto, D)
        各チャネルの上位 m_proto プロトタイプ画像インデックス
    """
    D = U.shape[0]
    U_dev = U.to(device=device, dtype=torch.float32)

    # 空間和 (N, D) を計算（大次元モデルでもメモリ負荷を抑えるためバッチ処理）
    s_list = []
    for start in range(0, z_train.shape[0], 1024):
        z_chunk = z_train[start : start + 1024].to(device=device, dtype=torch.float32)
        s_list.append(z_chunk.sum(dim=(2, 3)))
    s_orig = torch.cat(s_list, dim=0)  # (N, D)

    _, top_indices = select_positive_prototypes(s_orig, U_dev, m_proto=m_proto)  # (m_proto, D)
    purities = torch.zeros(m_proto, D, device=device, dtype=torch.float32)

    for r in range(m_proto):
        for c_start in range(0, D, chunk_channels):
            c_end = min(c_start + chunk_channels, D)
            B_c = c_end - c_start
            ch_ids = torch.arange(c_start, c_end, device=device)
            img_ids_cpu = top_indices[r, c_start:c_end].cpu()

            z_sub = z_train[img_ids_cpu].to(device=device, dtype=torch.float32)  # (B_c, D, H, W)
            z_flat = z_sub.reshape(B_c, D, -1)  # (B_c, D, H*W)

            # 対象チャネル ch_ids の空間マップのみを計算して最大活性画素位置 flat_pos を特定（49倍高速化）
            u_rows = U_dev[ch_ids]  # (B_c, D)
            target_maps = torch.einsum("bd,bdp->bp", u_rows, z_flat)  # (B_c, H*W)
            flat_pos = target_maps.argmax(dim=1)  # (B_c,)

            # 最大活性位置の画素特徴ベクトルに対してのみ U を作用させる（厳密に同値）
            z_pix = z_flat[torch.arange(B_c, device=device), :, flat_pos]  # (B_c, D)
            p_raw = z_pix @ U_dev.t()  # (B_c, D)
            p_pos = F.relu(p_raw)
            pk = p_pos[torch.arange(B_c, device=device), ch_ids]
            norm_p = p_pos.norm(dim=1).clamp_min(1e-8)
            purities[r, c_start:c_end] = pk / norm_p

    return purities.mean(dim=0).cpu(), top_indices.cpu()


def train_epic_disentanglement(
    epic_model: EPICModel,
    z_train: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    recalc_every: int = 2,
    m_start: int = 100,
    m_end: int = 5,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 42,
    track_purity_every_epoch: bool = True,
) -> pd.DataFrame:
    """
    論文 3節「Training」（p.5）の設定に準拠して Disentanglement Module の変換行列 U を最適化する．
    - エポック数: 20
    - プロトタイプ再計算間隔: 2エポックごと
    - 各チャネルのプロトタイプ数 m: 初期 100 から終了時 5 まで線形減少
    - 最適化手法: Adam (β1=0.9, β2=0.999, lr=0.001, weight_decay=1e-5), batch_size=512
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    epic_model.to(device)
    D = epic_model.num_channels

    # z_train が GPU メモリに余裕で収まるサイズ（例: D <= 1024）なら GPU に常駐させて高速化
    keep_on_gpu = (z_train.numel() * z_train.element_size()) <= (2.5 * 1024**3)
    z_store = z_train.to(device) if keep_on_gpu else z_train

    # 元特徴マップの空間和 S_orig ∈ R^{N×D} を事前計算
    with torch.no_grad():
        s_chunks = []
        for start in range(0, z_train.shape[0], 1024):
            s_chunks.append(z_store[start : start + 1024].to(device=device, dtype=torch.float32).sum(dim=(2, 3)))
        s_orig = torch.cat(s_chunks, dim=0)  # (N, D)

    optimizer = torch.optim.Adam(
        epic_model.disentangle.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    history_rows = []
    if track_purity_every_epoch:
        # 学習前 (epoch 0 時点) の Purity (m=5) を記録
        pur_init, _ = compute_all_channel_purities(z_train, epic_model.disentangle.get_matrix(), m_proto=5, device=device)
        history_rows.append(
            {
                "epoch": 0,
                "m_prototypes": m_start,
                "loss": float(1.0 - pur_init.mean().item()),
                "mean_purity_m5": float(pur_init.mean().item()),
                "median_purity_m5": float(pur_init.median().item()),
                "p90_purity_m5": float(torch.quantile(pur_init, 0.90).item()),
            }
        )

    img_indices = torch.empty(0, dtype=torch.long, device=device)
    ch_indices = torch.empty(0, dtype=torch.long, device=device)
    m_cur = m_start
    last_recalc_ep = max(epochs - recalc_every, 1)

    pbar = tqdm(range(epochs), desc=f"EPIC学習 ({epic_model.disentangle.matrix_type}, D={D})")
    for ep in pbar:
        if ep % recalc_every == 0:
            # 100枚から5枚へ線形減少
            ratio = min(ep / last_recalc_ep, 1.0)
            m_cur = max(int(round(m_start - (m_start - m_end) * ratio)), m_end)
            with torch.no_grad():
                U_cur = epic_model.disentangle.get_matrix().detach()
                _, top_idx = select_positive_prototypes(s_orig, U_cur, m_proto=m_cur)
                img_indices = top_idx.reshape(-1)  # (m_cur * D,)
                ch_indices = torch.arange(D, device=device).unsqueeze(0).expand(m_cur, D).reshape(-1)

        perm = torch.randperm(img_indices.numel(), device=device)
        epoch_loss_sum = 0.0
        num_samples = 0

        for start in range(0, perm.numel(), batch_size):
            b_sel = perm[start : start + batch_size]
            b_imgs = img_indices[b_sel]
            b_chs = ch_indices[b_sel]
            B = b_imgs.numel()

            optimizer.zero_grad()
            U = epic_model.disentangle.get_matrix()
            if keep_on_gpu:
                z_b = z_store[b_imgs].to(dtype=torch.float32)
            else:
                z_b = z_store[b_imgs.cpu()].to(device=device, dtype=torch.float32)

            z_flat = z_b.reshape(B, D, -1)  # (B, D, H*W)
            with torch.no_grad():
                target_map = torch.einsum("bd,bdp->bp", U.detach()[b_chs], z_flat)
                flat_pos = target_map.argmax(dim=1)  # (B,)

            # 最大活性画素の特徴ベクトル z_pix に対して U を作用させる（(U ⊛ Z)[:,:,flat_pos] と数学的に完全一致）
            z_pix = z_flat[torch.arange(B, device=device), :, flat_pos]  # (B, D)
            p_raw = z_pix @ U.t()  # (B, D)
            p_pos = F.relu(p_raw)
            pk = p_pos[torch.arange(B, device=device), b_chs]
            purity = pk / p_pos.norm(dim=1).clamp_min(1e-8)

            loss = 1.0 - purity.mean()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * B
            num_samples += B

        avg_loss = epoch_loss_sum / max(num_samples, 1)
        if track_purity_every_epoch or (ep == epochs - 1):
            pur_ep, _ = compute_all_channel_purities(
                z_train, epic_model.disentangle.get_matrix(), m_proto=5, device=device
            )
            mean_pur = float(pur_ep.mean().item())
            history_rows.append(
                {
                    "epoch": ep + 1,
                    "m_prototypes": m_cur,
                    "loss": avg_loss,
                    "mean_purity_m5": mean_pur,
                    "median_purity_m5": float(pur_ep.median().item()),
                    "p90_purity_m5": float(torch.quantile(pur_ep, 0.90).item()),
                }
            )
            pbar.set_postfix({"m": m_cur, "loss": f"{avg_loss:.4f}", "purity(m=5)": f"{mean_pur:.4f}"})
        else:
            pbar.set_postfix({"m": m_cur, "loss": f"{avg_loss:.4f}"})

    return pd.DataFrame(history_rows)


@torch.no_grad()
def evaluate_prediction_equivalence(
    base_model: nn.Module,
    epic_model: EPICModel,
    val_dataset: Dataset,
    imagenet1k_indices: List[int],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Remark 3.1 および 論文 Table 1 に対応し，元の事前学習済みモデルと EPIC 適用後モデルの
    分類精度・予測一致率・ロジット数値誤差を定量評価する．
    """
    base_model.eval().to(device)
    epic_model.eval().to(device)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    idx_map = torch.tensor(imagenet1k_indices, device=device, dtype=torch.long)

    total = 0
    base_top1_1k = 0
    epic_top1_1k = 0
    base_top5_1k = 0
    epic_top5_1k = 0
    base_acc_10 = 0
    epic_acc_10 = 0
    pred_agree_1k = 0
    max_logit_diff = 0.0
    sum_logit_diff = 0.0

    for x, y_local in loader:
        x = x.to(device)
        y_local = y_local.to(device)
        y_1k = idx_map[y_local]

        logits_base = base_model(x)
        logits_epic = epic_model(x)

        diff = (logits_base - logits_epic).abs()
        max_logit_diff = max(max_logit_diff, float(diff.max().item()))
        sum_logit_diff += float(diff.mean(dim=1).sum().item())

        pred_b = logits_base.argmax(dim=1)
        pred_e = logits_epic.argmax(dim=1)
        pred_agree_1k += int((pred_b == pred_e).sum().item())

        base_top1_1k += int((pred_b == y_1k).sum().item())
        epic_top1_1k += int((pred_e == y_1k).sum().item())

        top5_b = logits_base.topk(5, dim=1).indices
        top5_e = logits_epic.topk(5, dim=1).indices
        base_top5_1k += int((top5_b == y_1k.unsqueeze(1)).any(dim=1).sum().item())
        epic_top5_1k += int((top5_e == y_1k.unsqueeze(1)).any(dim=1).sum().item())

        # Imagenette 10クラスに制限した場合の精度
        base_acc_10 += int((logits_base[:, idx_map].argmax(dim=1) == y_local).sum().item())
        epic_acc_10 += int((logits_epic[:, idx_map].argmax(dim=1) == y_local).sum().item())

        total += x.size(0)

    U = epic_model.disentangle.get_matrix().detach()
    eye = torch.eye(U.shape[0], device=U.device)
    ortho_err = float(torch.linalg.norm(U.t() @ U - eye, ord="fro").item())
    cond_num = float(torch.linalg.cond(U).item())

    return {
        "Base_ImageNet1K_Top1(%)": 100.0 * base_top1_1k / total,
        "EPIC_ImageNet1K_Top1(%)": 100.0 * epic_top1_1k / total,
        "Base_ImageNet1K_Top5(%)": 100.0 * base_top5_1k / total,
        "EPIC_ImageNet1K_Top5(%)": 100.0 * epic_top5_1k / total,
        "Base_Imagenette10_Acc(%)": 100.0 * base_acc_10 / total,
        "EPIC_Imagenette10_Acc(%)": 100.0 * epic_acc_10 / total,
        "Prediction_Agreement(%)": 100.0 * pred_agree_1k / total,
        "Max_Abs_Logit_Diff": max_logit_diff,
        "Mean_Abs_Logit_Diff": sum_logit_diff / total,
        "Cond_U": cond_num,
        "Ortho_Error_Frob": ortho_err,
    }


def find_activation_bbox(
    channel_map_2d: torch.Tensor,
    img_size: int = 224,
    rel_threshold: float = 0.50,
) -> Tuple[int, int, int, int]:
    """
    特定チャネルの 2D 活性化マップ (H×W) を入力画像サイズ (224×224) に双線形補間し，
    最大活性化位置を含む高活性領域のバウンディングボックス (x_min, y_min, x_max, y_max) を求める．
    論文 Figure 1, 3, 5 の黄色枠（yellow box）によるプロトタイプ部位可視化に対応する．
    """
    act = F.relu(channel_map_2d.detach().float()).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    up = F.interpolate(act, size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0]
    max_val = float(up.max().item())
    if max_val <= 1e-8:
        return (img_size // 4, img_size // 4, 3 * img_size // 4, 3 * img_size // 4)

    mask = (up >= rel_threshold * max_val).cpu().numpy()

    # ピーク画素を含む連結成分のみを抽出して局所パーツを正確に囲む
    peak_idx = int(up.argmax().item())
    py, px = divmod(peak_idx, img_size)
    from scipy.ndimage import label

    labeled, num_feat = label(mask)
    peak_label = labeled[py, px]
    if num_feat > 0 and peak_label > 0:
        ys, xs = np.where(labeled == peak_label)
    else:
        ys, xs = np.where(mask)

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    # ボックスが極端に小さすぎないよう最小パディングを確保
    min_half = 14
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    x_min = max(0, min(x_min, cx - min_half))
    x_max = min(img_size - 1, max(x_max, cx + min_half))
    y_min = max(0, min(y_min, cy - min_half))
    y_max = min(img_size - 1, max(y_max, cy + min_half))
    return (x_min, y_min, x_max, y_max)


def draw_yellow_box(ax: plt.Axes, bbox: Tuple[int, int, int, int], lw: float = 2.5) -> None:
    """画像 Axes 上に論文同様の黄色バウンディングボックスを描画する．"""
    x0, y0, x1, y1 = bbox
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=lw,
        edgecolor="#FFD700",
        facecolor="none",
    )
    ax.add_patch(rect)


def compute_gradcam(
    backbone: nn.Module,
    fc_layer: nn.Linear,
    x: torch.Tensor,
    target_class: int | None = None,
    img_size: int = 224,
) -> Tuple[np.ndarray, int]:
    """
    古典的 Post-hoc XAI 手法である Grad-CAM ヒートマップを計算する（論文 Figure 1, 5 の比較対象）．
    """
    dev = x.device
    backbone = backbone.to(dev).eval()
    fc_layer = fc_layer.to(dev).eval()
    with torch.enable_grad():
        z = backbone(x)
        z = z.detach().requires_grad_(True)
        v = z.mean(dim=(2, 3))
        logits = fc_layer(v)
        pred_cls = int(logits[0].argmax().item()) if target_class is None else int(target_class)
        score = logits[0, pred_cls]
        grads = torch.autograd.grad(score, z)[0]  # (1, D, H, W)

    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, D, 1, 1)
    cam = F.relu((weights * z).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam_up = F.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam_up.detach().cpu().numpy()
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np, pred_cls


@torch.no_grad()
def evaluate_channel_concentration_and_deletion(
    base_epic_identity: EPICModel,
    trained_epic: EPICModel,
    val_dataset: Dataset,
    device: torch.device,
    max_samples: int = 512,
    k_list: Tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> pd.DataFrame:
    """
    EPIC 最適化前後において，予測クラスへの正の寄与上位 k チャネルのみを残した際の
    予測維持率（Keep Top-k Faithfulness）および上位 k チャネルを除去した際の予測低下率
    （Delete Top-k Drop）を比較評価する．
    チャネルが分離（高 Purity 化）されることで，少数のプロトタイプチャネルでモデルの判断根拠を
    より高い忠実度で説明できるかを定量検証する．
    """
    base_epic_identity.eval().to(device)
    trained_epic.eval().to(device)

    loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    xs, _ = [], []
    count = 0
    for x, _ in loader:
        xs.append(x)
        count += x.size(0)
        if count >= max_samples:
            break
    x_all = torch.cat(xs, dim=0)[:max_samples].to(device)

    rows = []
    for label_name, model in [("EPIC適用前 (U = I)", base_epic_identity), ("EPIC適用後 (学習済み U)", trained_epic)]:
        logits_full, _, _, v = model.forward_features(x_all)
        preds_full = logits_full.argmax(dim=1)  # (N,)
        probs_full = F.softmax(logits_full, dim=1)[torch.arange(x_all.size(0), device=device), preds_full]

        a_prime = model.get_modified_fc_weight()  # (C, D)
        w_preds = a_prime[preds_full]  # (N, D)
        pos_contrib = w_preds * F.relu(v)  # (N, D)
        total_pos = pos_contrib.clamp_min(0.0).sum(dim=1).clamp_min(1e-8)

        sorted_idx = pos_contrib.argsort(dim=1, descending=True)  # (N, D)

        for k in k_list:
            topk_idx = sorted_idx[:, :k]
            mask_keep = torch.zeros_like(v)
            mask_keep.scatter_(1, topk_idx, 1.0)

            # 上位 k チャネルのみ保持
            logits_keep = F.linear(v * mask_keep, a_prime, model.orig_bias)
            keep_agree = float((logits_keep.argmax(dim=1) == preds_full).float().mean().item() * 100.0)
            contrib_share = float(
                (pos_contrib.gather(1, topk_idx).clamp_min(0.0).sum(dim=1) / total_pos).mean().item() * 100.0
            )

            # 上位 k チャネルを除去 (Deletion)
            logits_del = F.linear(v * (1.0 - mask_keep), a_prime, model.orig_bias)
            probs_del = F.softmax(logits_del, dim=1)[torch.arange(x_all.size(0), device=device), preds_full]
            prob_drop = float((probs_full - probs_del).mean().item() * 100.0)

            rows.append(
                {
                    "モデル状態": label_name,
                    "上位チャネル数 k": k,
                    "正寄与度カバー率(%)": contrib_share,
                    "Top-kのみでの予測維持率(%)": keep_agree,
                    "Top-k除去時の確信度低下幅(%pt)": prob_drop,
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_spatial_part_faithfulness(
    base_model: nn.Module,
    trained_epic: EPICModel,
    val_dataset: Dataset,
    device: torch.device,
    top_k_channels: int = 3,
    max_samples: int = 256,
    ratios: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """
    画像空間における Deletion / Insertion 評価（論文 4節 Multi-dimensional Analysis の忠実度検証に対応）．
    Algorithm 1 で選ばれた上位 k 個の正プロトタイプチャネルの空間寄与マップに基づき，
    重要画素から順に遮蔽（Deletion）または露出（Insertion）した際の予測クラス確信度変化を
    ランダム遮蔽ベースラインと比較する．
    """
    base_model.eval().to(device)
    trained_epic.eval().to(device)

    loader = DataLoader(val_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    xs = []
    for x, _ in loader:
        xs.append(x)
        if sum(t.size(0) for t in xs) >= max_samples:
            break
    x_all = torch.cat(xs, dim=0)[:max_samples].to(device)  # (N, 3, 224, 224)
    N, _, H_img, W_img = x_all.shape
    num_pix = H_img * W_img

    # EPIC 上位 k プロトタイプによる空間重要度マップを計算
    logits_orig, z_orig, z_hat, v = trained_epic.forward_features(x_all)
    preds = logits_orig.argmax(dim=1)  # (N,)
    a_prime = trained_epic.get_modified_fc_weight()  # (C, D)
    w_preds = a_prime[preds]  # (N, D)
    scores = w_preds * F.relu(v)  # (N, D)
    topk_idx = scores.topk(top_k_channels, dim=1).indices  # (N, k)

    # 上位 k チャネルの重み付き空間マップを合成 (N, 1, 7, 7) -> (N, 224*224)
    batch_arange = torch.arange(N, device=device).unsqueeze(1)
    z_topk = F.relu(z_hat[batch_arange, topk_idx])  # (N, k, 7, 7)
    w_topk = w_preds.gather(1, topk_idx).clamp_min(0.0).unsqueeze(-1).unsqueeze(-1)  # (N, k, 1, 1)
    epic_map_low = (w_topk * z_topk).sum(dim=1, keepdim=True)  # (N, 1, 7, 7)
    epic_map = F.interpolate(epic_map_low, size=(H_img, W_img), mode="bilinear", align_corners=False).reshape(N, -1)

    # ランダム順序ベースライン（塩胡椒ノイズによる高周波アーティファクトを防ぐため，同じ 7×7 パッチ解像度で生成）
    gen = torch.Generator(device=device).manual_seed(42)
    rand_low = torch.rand((N, 1, z_hat.shape[2], z_hat.shape[3]), generator=gen, device=device)
    rand_map = F.interpolate(rand_low, size=(H_img, W_img), mode="bilinear", align_corners=False).reshape(N, -1)

    epic_order = epic_map.argsort(dim=1, descending=True)
    rand_order = rand_map.argsort(dim=1, descending=True)

    # ぼかし背景（Insertion の初期キャンバス：正規化後ゼロ＝ImageNet平均色）
    canvas = torch.zeros_like(x_all)

    records = []
    for method_name, order in [("EPIC Top-3 プロトタイプ部位", epic_order), ("ランダム領域 (Baseline)", rand_order)]:
        for r in ratios:
            m_pix = int(round(r * num_pix))
            mask_flat = torch.zeros((N, num_pix), device=device)
            if m_pix > 0:
                mask_flat.scatter_(1, order[:, :m_pix], 1.0)
            mask_img = mask_flat.reshape(N, 1, H_img, W_img)

            # Deletion: 重要部位を平均色でマスク
            x_del = x_all * (1.0 - mask_img) + canvas * mask_img
            prob_del = F.softmax(base_model(x_del), dim=1)[torch.arange(N, device=device), preds].mean().item()

            # Insertion: 平均色キャンバスに重要部位のみを挿入
            x_ins = canvas * (1.0 - mask_img) + x_all * mask_img
            prob_ins = F.softmax(base_model(x_ins), dim=1)[torch.arange(N, device=device), preds].mean().item()

            records.append(
                {
                    "手法": method_name,
                    "画素割合(%)": r * 100.0,
                    "Deletion確信度": prob_del,
                    "Insertion確信度": prob_ins,
                }
            )

    return pd.DataFrame(records)
