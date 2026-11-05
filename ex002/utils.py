"""
ex002/utils.py
2026年 電子情報通信学会ソサイエティ大会 論文 (N-1-43)
「EPICにおけるチャネル分離度と分類精度の関係に関する検討」（室岡・代・神野, 2026）
の再現実験および交絡要因（チャネル数・分類層重み・対応クラス偏り・冗長性）解析用ユーティリティ関数群．
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label as connected_components
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as T
from tqdm import tqdm

from model import ImagenetteResNet34, ChannelMaskedEPICResNet34


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGENETTE_CLASSES_JA: List[str] = [
    "テンチ(魚)",
    "スプリンガー犬",
    "カセット機",
    "チェーンソー",
    "教会",
    "ホルン",
    "ごみ収集車",
    "給油計量機",
    "ゴルフボール",
    "パラシュート",
]

# 論文 N-1-43 表1 の参照値（全3,925枚の検証データに対する精度）
PAPER_TABLE1_REF: Dict[str, Tuple[str, int, float]] = {
    "G1": (">= 0.60", 115, 0.7778),
    "G2": ("0.55-0.60", 157, 0.8713),
    "G3": ("0.50-0.55", 126, 0.8471),
    "G4": ("0.45-0.50", 80, 0.7763),
    "G5": ("< 0.45", 34, 0.6189),
    "Baseline": ("All (512)", 512, 0.8736),
}


def setup_japanese_matplotlib() -> None:
    """Matplotlib の日本語フォント表示を設定する．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120


def preload_imagenette_uint8(
    dataset_root: str | Path,
    resize_shorter: int = 256,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    高速化のため Imagenette (train=9,469枚, val=3,925枚) をマルチスレッドで読み込み，
    256×256 の uint8 テンソル (N, 3, 256, 256) としてメモリ上に一括保持する．
    ディスク I/O ボトルネックを解消し，GPU 上で高速にランダムクロップ・正規化を行うために用いる．
    """
    root = Path(dataset_root)
    tf_256 = T.Compose(
        [
            T.Resize(resize_shorter),
            T.CenterCrop(resize_shorter),
            T.PILToTensor(),  # uint8 (3, 256, 256)
        ]
    )

    def _load_split(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = datasets.Imagenette(root, split=split, size="320px", download=False)
        samples = ds._samples

        def _read_one(item: Tuple[str, int]) -> Tuple[torch.Tensor, int]:
            path_str, label_idx = item
            with Image.open(path_str) as img:
                t = tf_256(img.convert("RGB"))
            return t, label_idx

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(
                tqdm(
                    ex.map(_read_one, samples),
                    total=len(samples),
                    desc=f"Imagenette ({split}) 画像読み込み",
                )
            )
        xs = torch.stack([r[0] for r in results], dim=0)
        ys = torch.tensor([r[1] for r in results], dtype=torch.long)
        return xs, ys

    x_train_256, y_train = _load_split("train")
    x_val_256, y_val = _load_split("val")
    return x_train_256, y_train, x_val_256, y_val


def normalize_uint8_batch(
    x_uint8: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """uint8 画像バッチ (B, 3, H, W) を GPU 上で [0, 1] 変換および ImageNet 正規化する．"""
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    x = x_uint8.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
    return (x - mean) / std


def center_crop_uint8(x_uint8: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    """(N, 3, H, W) の uint8 テンソルの中央 crop_size×crop_size 領域を切り出す．"""
    _, _, h, w = x_uint8.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return x_uint8[:, :, top : top + crop_size, left : left + crop_size]


def finetune_resnet34_on_imagenette(
    model: ImagenetteResNet34,
    x_train_256: torch.Tensor,
    y_train: torch.Tensor,
    x_val_256: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    crop_size: int = 224,
    seed: int = 42,
) -> pd.DataFrame:
    """
    ImageNet 事前学習済み ResNet-34 を Imagenette (10クラス) でファインチューニングする．
    GPU 上でのランダムクロップ (256→224) とランダム水平反転によりデータ拡張を行いながら高速に学習する．
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)

    n_train = x_train_256.size(0)
    x_val_224 = center_crop_uint8(x_val_256, crop_size=crop_size)
    val_loader = DataLoader(TensorDataset(x_val_224, y_val), batch_size=256, shuffle=False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    max_offset = x_train_256.size(2) - crop_size

    rows = []
    pbar = tqdm(range(epochs), desc="ResNet-34 ファインチューニング")
    for ep in pbar:
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        running_loss = 0.0
        running_correct = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb_256 = x_train_256[idx]
            yb = y_train[idx].to(device, non_blocking=True)

            # バッチ単位のランダムクロップ (256 -> 224) と水平反転
            top = int(torch.randint(0, max_offset + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, max_offset + 1, (1,), generator=gen).item())
            xb_crop = xb_256[:, :, top : top + crop_size, left : left + crop_size]
            xb = normalize_uint8_batch(xb_crop, device=device)
            if float(torch.rand(1, generator=gen).item()) < 0.5:
                xb = xb.flip(-1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            running_correct += int((logits.argmax(dim=1) == yb).sum().item())

        scheduler.step()
        train_loss = running_loss / n_train
        train_acc = running_correct / n_train

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        with torch.no_grad():
            for xv_u8, yv in val_loader:
                xv = normalize_uint8_batch(xv_u8, device=device)
                yv = yv.to(device, non_blocking=True)
                logits = model(xv)
                val_loss_sum += criterion(logits, yv).item() * xv.size(0)
                val_correct += int((logits.argmax(dim=1) == yv).sum().item())

        val_loss = val_loss_sum / len(y_val)
        val_acc = val_correct / len(y_val)
        rows.append(
            {
                "epoch": ep + 1,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        pbar.set_postfix({"train_acc": f"{train_acc:.4f}", "val_acc": f"{val_acc:.4f}"})

    return pd.DataFrame(rows)


@torch.no_grad()
def extract_feature_maps(
    model: ImagenetteResNet34,
    x_256: torch.Tensor,
    device: torch.device,
    crop_size: int = 224,
    batch_size: int = 256,
    desc: str = "特徴マップ抽出",
) -> torch.Tensor:
    """中央クロップ (224×224) 画像から layer4 特徴マップ Z ∈ R^{N×512×7×7} を一括抽出する．"""
    model.eval().to(device)
    x_224 = center_crop_uint8(x_256, crop_size=crop_size)
    z_list = []
    for start in tqdm(range(0, x_224.size(0), batch_size), desc=desc):
        xb = normalize_uint8_batch(x_224[start : start + batch_size], device=device)
        z = model.extract_feature_map(xb)
        z_list.append(z.cpu())
    return torch.cat(z_list, dim=0)


@torch.no_grad()
def verify_epic_equivalence_10cls(
    ft_model: ImagenetteResNet34,
    epic_model: ChannelMaskedEPICResNet34,
    x_val_256: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    ファインチューニング後の10クラスモデルと EPIC 適用後モデルの出力ロジットおよび予測一致率を検証する．
    """
    ft_model.eval().to(device)
    epic_model.eval().to(device)
    x_val_224 = center_crop_uint8(x_val_256, crop_size=224)

    base_correct = 0
    epic_correct = 0
    agree_count = 0
    max_diff = 0.0
    n_total = x_val_224.size(0)

    for start in range(0, n_total, batch_size):
        xb = normalize_uint8_batch(x_val_224[start : start + batch_size], device=device)
        yb = y_val[start : start + batch_size].to(device)

        logits_base = ft_model(xb)
        logits_epic, _ = epic_model(xb, channel_mask=None)

        diff = (logits_base - logits_epic).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        pred_b = logits_base.argmax(dim=1)
        pred_e = logits_epic.argmax(dim=1)

        base_correct += int((pred_b == yb).sum().item())
        epic_correct += int((pred_e == yb).sum().item())
        agree_count += int((pred_b == pred_e).sum().item())

    U = epic_model.disentangle.get_matrix().detach()
    ortho_err = float(torch.linalg.norm(U.t() @ U - torch.eye(U.size(0), device=U.device)).item())

    return {
        "EPIC適用前_Val_Accuracy": round(base_correct / n_total, 4),
        "EPIC適用後_Val_Accuracy": round(epic_correct / n_total, 4),
        "予測一致率(%)": round(100.0 * agree_count / n_total, 2),
        "最大ロジット絶対誤差": max_diff,
        "直交誤差_||U^T U - I||_F": ortho_err,
    }


@torch.no_grad()
def compute_channel_purities(
    z_train: torch.Tensor,
    U: torch.Tensor,
    m_proto: int = 5,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    EPIC 変換行列 U 適用後の特徴空間において，全512チャネルの平均 Purity（上位 m_proto 枚平均）を計算する．
    """
    _, D, _, _ = z_train.shape
    U_dev = U.to(device=device, dtype=torch.float32)
    z_gpu = z_train.to(device=device, dtype=torch.float32)

    s_orig = z_gpu.sum(dim=(2, 3))  # (N, D)
    activ = s_orig @ U_dev.t()  # (N, D)
    _, top_indices = torch.topk(activ, k=m_proto, dim=0)  # (m_proto, D)

    purities = torch.zeros(m_proto, D, device=device, dtype=torch.float32)
    ch_ids = torch.arange(D, device=device)

    for r in range(m_proto):
        img_ids = top_indices[r]  # (D,)
        z_sub = z_gpu[img_ids].reshape(D, D, -1)  # (D, D, H*W)
        target_maps = torch.einsum("bd,bdp->bp", U_dev, z_sub)  # (D, H*W)
        flat_pos = target_maps.argmax(dim=1)  # (D,)

        z_pix = z_sub[ch_ids, :, flat_pos]  # (D, D)
        p_pos = F.relu(z_pix @ U_dev.t())  # (D, D)
        pk = p_pos[ch_ids, ch_ids]
        purities[r] = pk / p_pos.norm(dim=1).clamp_min(1e-8)

    return purities.mean(dim=0).cpu(), top_indices.cpu()


def train_epic_on_features(
    epic_model: ChannelMaskedEPICResNet34,
    z_train: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    recalc_every: int = 2,
    m_start: int = 100,
    m_end: int = 5,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    m_eval_purity: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    EPIC 論文の設定に従い Disentanglement Module の変換行列 U を20エポック学習する．
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    epic_model.to(device)

    D = epic_model.num_channels
    z_gpu = z_train.to(device=device, dtype=torch.float32)
    s_orig = z_gpu.sum(dim=(2, 3))  # (N, D)

    optimizer = torch.optim.Adam(
        epic_model.disentangle.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    history = []
    pur0, _ = compute_channel_purities(z_train, epic_model.disentangle.get_matrix(), m_proto=m_eval_purity, device=device)
    history.append(
        {
            "epoch": 0,
            "m_prototypes": m_start,
            "loss": round(float(1.0 - pur0.mean()), 4),
            "mean_purity": round(float(pur0.mean()), 4),
        }
    )

    img_indices = torch.empty(0, dtype=torch.long, device=device)
    ch_indices = torch.empty(0, dtype=torch.long, device=device)
    m_cur = m_start
    last_recalc_ep = max(epochs - recalc_every, 1)

    pbar = tqdm(range(epochs), desc="EPIC Disentanglement 学習")
    for ep in pbar:
        if ep % recalc_every == 0:
            ratio = min(ep / last_recalc_ep, 1.0)
            m_cur = max(int(round(m_start - (m_start - m_end) * ratio)), m_end)
            with torch.no_grad():
                U_cur = epic_model.disentangle.get_matrix().detach()
                activ = s_orig @ U_cur.t()
                _, top_idx = torch.topk(activ, k=m_cur, dim=0)
                img_indices = top_idx.reshape(-1)
                ch_indices = torch.arange(D, device=device).unsqueeze(0).expand(m_cur, D).reshape(-1)

        perm = torch.randperm(img_indices.numel(), device=device)
        loss_sum = 0.0
        n_count = 0

        for start in range(0, perm.numel(), batch_size):
            b_sel = perm[start : start + batch_size]
            b_imgs = img_indices[b_sel]
            b_chs = ch_indices[b_sel]
            B = b_imgs.numel()

            optimizer.zero_grad()
            U = epic_model.disentangle.get_matrix()
            z_flat = z_gpu[b_imgs].reshape(B, D, -1)

            with torch.no_grad():
                target_map = torch.einsum("bd,bdp->bp", U.detach()[b_chs], z_flat)
                flat_pos = target_map.argmax(dim=1)

            z_pix = z_flat[torch.arange(B, device=device), :, flat_pos]
            p_pos = F.relu(z_pix @ U.t())
            pk = p_pos[torch.arange(B, device=device), b_chs]
            purity = pk / p_pos.norm(dim=1).clamp_min(1e-8)

            loss = 1.0 - purity.mean()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * B
            n_count += B

        pur_ep, _ = compute_channel_purities(
            z_train, epic_model.disentangle.get_matrix(), m_proto=m_eval_purity, device=device
        )
        avg_loss = loss_sum / max(n_count, 1)
        mean_pur = float(pur_ep.mean().item())
        history.append(
            {
                "epoch": ep + 1,
                "m_prototypes": m_cur,
                "loss": round(avg_loss, 4),
                "mean_purity": round(mean_pur, 4),
            }
        )
        pbar.set_postfix({"m": m_cur, "loss": f"{avg_loss:.4f}", "purity": f"{mean_pur:.4f}"})

    return pd.DataFrame(history)


def assign_purity_groups(purities: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    論文 N-1-43 表1 の定義に従い，512チャネルを Purity に基づいて5群 (G1〜G5) に分類する：
      - G1: Purity >= 0.60
      - G2: 0.55 <= Purity < 0.60
      - G3: 0.50 <= Purity < 0.55
      - G4: 0.45 <= Purity < 0.50
      - G5: Purity < 0.45
    """
    p = purities.detach().cpu().float()
    return {
        "G1": p >= 0.60,
        "G2": (p >= 0.55) & (p < 0.60),
        "G3": (p >= 0.50) & (p < 0.55),
        "G4": (p >= 0.45) & (p < 0.50),
        "G5": p < 0.45,
    }


@torch.no_grad()
def evaluate_single_and_cumulative_groups(
    epic_model: ChannelMaskedEPICResNet34,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    groups: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    論文 N-1-43 の主実験（表1・図1）および相補的な累積・除去評価を行う：
    1. single_df : 各群 (G1〜G5) 単独使用時の分類精度と論文参照値の比較表（表1対応）
    2. cum_high_df: 高Purity側から順次チャネル群を追加した場合の精度推移（図1対応：keep_G1 〜 keep_all）
    3. extra_df  : 低Purity側からの累積追加および各群1つを除去（Leave-One-Group-Out）した際の精度変化
    """
    epic_model.eval().to(device)
    U = epic_model.disentangle.get_matrix()
    v_orig_val = z_val.to(device=device, dtype=torch.float32).mean(dim=(2, 3))  # (N_val, 512)
    v_dis_val = v_orig_val @ U.t()  # avg_pool(U ⊛ Z_val) ∈ R^{N_val×512}
    y_dev = y_val.to(device)

    def calc_acc(mask_bool: torch.Tensor) -> float:
        logits = epic_model.forward_from_pooled(v_dis_val, channel_mask=mask_bool.float())
        return float((logits.argmax(dim=1) == y_dev).float().mean().item())

    # 1. 各群単独評価（表1対応）
    single_rows = []
    for g_name in ["G1", "G2", "G3", "G4", "G5"]:
        mask = groups[g_name]
        n_ch = int(mask.sum().item())
        acc = calc_acc(mask) if n_ch > 0 else 0.0
        rng_str, ref_ch, ref_acc = PAPER_TABLE1_REF[g_name]
        single_rows.append(
            {
                "群": g_name,
                "Purity範囲": rng_str,
                "チャネル数(本実験)": n_ch,
                "Accuracy(本実験)": round(acc, 4),
                "チャネル数(論文表1)": ref_ch,
                "Accuracy(論文表1)": ref_acc,
            }
        )

    all_mask = torch.ones(epic_model.num_channels, dtype=torch.bool)
    base_acc = calc_acc(all_mask)
    rng_b, ref_ch_b, ref_acc_b = PAPER_TABLE1_REF["Baseline"]
    single_rows.append(
        {
            "群": "Baseline",
            "Purity範囲": rng_b,
            "チャネル数(本実験)": int(all_mask.sum().item()),
            "Accuracy(本実験)": round(base_acc, 4),
            "チャネル数(論文表1)": ref_ch_b,
            "Accuracy(論文表1)": ref_acc_b,
        }
    )
    single_df = pd.DataFrame(single_rows)

    # 2. 高Purity側から順次追加（論文 図1 対応）
    cum_specs = [
        ("keep_G1", ["G1"]),
        ("keep_G1_G2", ["G1", "G2"]),
        ("keep_G1_G2_G3", ["G1", "G2", "G3"]),
        ("keep_G1_G2_G3_G4", ["G1", "G2", "G3", "G4"]),
        ("keep_all", ["G1", "G2", "G3", "G4", "G5"]),
    ]
    cum_rows = []
    for label, g_list in cum_specs:
        mask = torch.zeros(epic_model.num_channels, dtype=torch.bool)
        for g in g_list:
            mask |= groups[g]
        acc_val = calc_acc(mask)
        cum_rows.append(
            {
                "設定": label,
                "使用群": "+".join(g_list),
                "累積チャネル数": int(mask.sum().item()),
                "Accuracy": round(acc_val, 4),
                "Baseline比差分": round(acc_val - base_acc, 4),
            }
        )
    cum_high_df = pd.DataFrame(cum_rows)

    # 3. 低Purity側からの累積追加 ＆ Leave-One-Group-Out（冗長性検証）
    extra_rows = []
    low_specs = [
        ("keep_G5", ["G5"]),
        ("keep_G5_G4", ["G5", "G4"]),
        ("keep_G5_G4_G3", ["G5", "G4", "G3"]),
        ("keep_G5_G4_G3_G2", ["G5", "G4", "G3", "G2"]),
        ("keep_all", ["G5", "G4", "G3", "G2", "G1"]),
    ]
    for label, g_list in low_specs:
        mask = torch.zeros(epic_model.num_channels, dtype=torch.bool)
        for g in g_list:
            mask |= groups[g]
        acc_val = calc_acc(mask)
        extra_rows.append(
            {
                "評価種別": "低Purity側から累積追加",
                "設定": label,
                "使用チャネル数": int(mask.sum().item()),
                "Accuracy": round(acc_val, 4),
                "Baseline比差分": round(acc_val - base_acc, 4),
            }
        )

    for g_drop in ["G1", "G2", "G3", "G4", "G5"]:
        mask = ~groups[g_drop]
        acc_val = calc_acc(mask)
        extra_rows.append(
            {
                "評価種別": "単一群のみ除去 (Leave-One-Out)",
                "設定": f"drop_{g_drop}",
                "使用チャネル数": int(mask.sum().item()),
                "Accuracy": round(acc_val, 4),
                "Baseline比差分": round(acc_val - base_acc, 4),
            }
        )
    extra_df = pd.DataFrame(extra_rows)

    return single_df, cum_high_df, extra_df


@torch.no_grad()
def evaluate_m_proto_sensitivity(
    epic_model: ChannelMaskedEPICResNet34,
    z_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    m_list: List[int],
    device: torch.device,
) -> pd.DataFrame:
    """
    Purity 評価時の上位プロトタイプ枚数 m_eval を変化させた際の群別チャネル数・単独精度・累積精度を比較する．
    """
    U = epic_model.disentangle.get_matrix()
    rows = []
    for m_eval in m_list:
        pur, _ = compute_channel_purities(z_train, U, m_proto=m_eval, device=device)
        groups = assign_purity_groups(pur)
        s_df, c_df, _ = evaluate_single_and_cumulative_groups(epic_model, z_val, y_val, groups, device=device)
        s_map = dict(zip(s_df["群"], zip(s_df["チャネル数(本実験)"], s_df["Accuracy(本実験)"])))
        c_map = dict(zip(c_df["設定"], c_df["Accuracy"]))
        rows.append(
            {
                "m_eval": m_eval,
                "平均Purity": round(float(pur.mean().item()), 4),
                "G1_ch": s_map["G1"][0],
                "G2_ch": s_map["G2"][0],
                "G3_ch": s_map["G3"][0],
                "G4_ch": s_map["G4"][0],
                "G5_ch": s_map["G5"][0],
                "G1単独Acc": s_map["G1"][1],
                "G2単独Acc": s_map["G2"][1],
                "G3単独Acc": s_map["G3"][1],
                "G4単独Acc": s_map["G4"][1],
                "G5単独Acc": s_map["G5"][1],
                "G1..G4累積Acc": c_map["keep_G1_G2_G3_G4"],
                "Baseline_Acc": s_map["Baseline"][1],
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def analyze_confounding_factors(
    epic_model: ChannelMaskedEPICResNet34,
    purities: torch.Tensor,
    groups: Dict[str, torch.Tensor],
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    subsample_k: int = 14,
    num_trials: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    論文 N-1-43 3節・4節で指摘されている交絡要因（チャネル数の違い・分類層重みの大きさ・対応クラスの偏り）
    を定量的に切り分ける追加検証関数：
    1. group_analysis_df: 各群の重みL2ノルム・実効寄与度・カバークラス数・同数チャネル抽出精度
    2. quintile_df      : 全512チャネルを等チャネル数（約102チャネル×5群）の5分位群 Q1〜Q5 に分割した統制評価
    3. class_dist_df    : 各群 G1〜G5 において各クラス (0〜9) を最大重みとするチャネル数の分布表
    """
    epic_model.eval().to(device)
    U = epic_model.disentangle.get_matrix()
    a_prime = epic_model.get_modified_fc_weight().detach()  # (10, 512)
    v_dis_val = z_val.to(device=device, dtype=torch.float32).mean(dim=(2, 3)) @ U.t()  # (N_val, 512)
    y_dev = y_val.to(device)

    weight_l2 = a_prime.norm(dim=0).cpu()  # (512,)
    weight_max = a_prime.max(dim=0).values.cpu()  # (512,)
    dominant_cls = a_prime.argmax(dim=0).cpu()  # (512,)
    mean_pos_act = F.relu(v_dis_val).mean(dim=0).cpu()  # (512,)
    effective_imp = weight_l2 * mean_pos_act  # (512,)

    rng = np.random.default_rng(seed)
    group_stats_rows = []
    class_dist_rows = []

    # 最小群のチャネル数に合わせて全群同数抽出 K を設定
    non_empty_counts = [int(groups[g].sum().item()) for g in ["G1", "G2", "G3", "G4", "G5"] if int(groups[g].sum().item()) > 0]
    k_uniform = min(subsample_k, min(non_empty_counts))

    for g_name in ["G1", "G2", "G3", "G4", "G5"]:
        idx = torch.where(groups[g_name])[0].numpy()
        n_ch = len(idx)
        if n_ch == 0:
            continue

        trial_accs = []
        for _ in range(num_trials):
            chosen = rng.choice(idx, size=k_uniform, replace=False)
            mask = torch.zeros(epic_model.num_channels, device=device)
            mask[chosen] = 1.0
            logits = epic_model.forward_from_pooled(v_dis_val, channel_mask=mask)
            trial_accs.append(float((logits.argmax(dim=1) == y_dev).float().mean().item()))

        dom_c = dominant_cls[idx].numpy()
        cls_counts = [int((dom_c == c).sum()) for c in range(10)]
        cls_covered = int(sum(cnt > 0 for cnt in cls_counts))
        # 正規化エントロピー（1.0 に近いほど全10クラスを均等にカバー）
        probs = np.array(cls_counts, dtype=np.float64) / max(n_ch, 1)
        probs_nz = probs[probs > 0]
        norm_entropy = float(-np.sum(probs_nz * np.log(probs_nz)) / np.log(10.0))

        group_stats_rows.append(
            {
                "群": g_name,
                "チャネル数": n_ch,
                "平均Purity": round(float(purities[idx].mean()), 4),
                "分類層重みL2ノルム平均": round(float(weight_l2[idx].mean()), 4),
                "最大正重み平均": round(float(weight_max[idx].mean()), 4),
                "平均正活性量": round(float(mean_pos_act[idx].mean()), 4),
                "実効寄与度平均": round(float(effective_imp[idx].mean()), 4),
                "カバークラス数(/10)": cls_covered,
                "クラス均等度(Entropy)": round(norm_entropy, 4),
                f"同数({k_uniform}ch)抽出精度_平均": round(float(np.mean(trial_accs)), 4),
                f"同数({k_uniform}ch)抽出精度_標準偏差": round(float(np.std(trial_accs)), 4),
            }
        )

        row_cls = {"群": g_name}
        for c_i, c_name in enumerate(IMAGENETTE_CLASSES_JA):
            row_cls[c_name] = cls_counts[c_i]
        class_dist_rows.append(row_cls)

    group_analysis_df = pd.DataFrame(group_stats_rows)
    class_dist_df = pd.DataFrame(class_dist_rows)

    # 等チャネル数（約102チャネル×5群）の5分位群 Q1(最高Purity)〜Q5(最低Purity) による統制評価
    sorted_chs = purities.argsort(descending=True)
    quintile_chunks = torch.tensor_split(sorted_chs, 5)
    quintile_rows = []
    cum_mask = torch.zeros(epic_model.num_channels, device=device)
    for q_i, q_idx in enumerate(quintile_chunks):
        q_mask = torch.zeros(epic_model.num_channels, device=device)
        q_mask[q_idx.to(device)] = 1.0
        cum_mask = torch.clamp(cum_mask + q_mask, max=1.0)

        acc_single = float(
            (epic_model.forward_from_pooled(v_dis_val, channel_mask=q_mask).argmax(dim=1) == y_dev)
            .float()
            .mean()
            .item()
        )
        acc_cum = float(
            (epic_model.forward_from_pooled(v_dis_val, channel_mask=cum_mask).argmax(dim=1) == y_dev)
            .float()
            .mean()
            .item()
        )
        p_sub = purities[q_idx]
        quintile_rows.append(
            {
                "等数分位群": f"Q{q_i + 1}",
                "チャネル数": int(q_idx.numel()),
                "Purity範囲": f"{float(p_sub.min()):.3f}-{float(p_sub.max()):.3f}",
                "平均Purity": round(float(p_sub.mean()), 4),
                "単独Accuracy": round(acc_single, 4),
                "高Purity側からの累積Accuracy": round(acc_cum, 4),
                "重みL2ノルム平均": round(float(weight_l2[q_idx].mean()), 4),
                "実効寄与度平均": round(float(effective_imp[q_idx].mean()), 4),
            }
        )
    quintile_df = pd.DataFrame(quintile_rows)

    return group_analysis_df, quintile_df, class_dist_df


def find_activation_bbox(
    channel_map_2d: torch.Tensor,
    img_size: int = 224,
    rel_threshold: float = 0.50,
) -> Tuple[int, int, int, int]:
    """
    特定チャネルの 2D 活性化マップ (H×W) を入力画像サイズ (224×224) に双線形補間し，
    最大活性位置を含む連結領域のバウンディングボックス (x_min, y_min, x_max, y_max) を返す．
    """
    act = F.relu(channel_map_2d.detach().float()).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(act, size=(img_size, img_size), mode="bilinear", align_corners=False)[0, 0]
    max_val = float(up.max().item())
    if max_val <= 1e-8:
        return (img_size // 4, img_size // 4, 3 * img_size // 4, 3 * img_size // 4)

    mask = (up >= rel_threshold * max_val).cpu().numpy()
    peak_idx = int(up.argmax().item())
    py, px = divmod(peak_idx, img_size)
    labeled, num_feat = connected_components(mask)
    peak_label = labeled[py, px]
    if num_feat > 0 and peak_label > 0:
        ys, xs = np.where(labeled == peak_label)
    else:
        ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
