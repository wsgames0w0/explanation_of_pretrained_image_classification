# 事前学習済み画像分類ネットワークのプロトタイプベース説明手法（EPIC）とチャネル分離度に関する再現実験

本リポジトリは，事前学習済み画像分類ネットワークに対する Post-hoc 型プロトタイプ説明手法 **EPIC（Explanation of Pretrained Image Classification Networks via Prototypes）** および，EPIC によって分離された特徴チャネルの **分離度（Purity）と分類精度の関係** に関する下記2編の論文の再現・検証実験をまとめたプロジェクトです．

## 対象論文

1. **実験1（[`ex001/`](./ex001/) / レポート：[`report/ex001.md`](./report/ex001.md)）**
   - Piotr Borycki, Magdalena Trędowicz, Szymon Janusz, Jacek Tabor, Przemysław Spurek, Arkadiusz Lewicki, and Łukasz Struski, **"EPIC: Explanation of Pretrained Image Classification Networks via Prototypes,"** *Proceedings of the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)*, pp. 17366–17373, 2026.
   - バックボーンを再学習せず，Global Average Pooling 直前への可逆線形変換（Disentanglement Module）の挿入と最終分類層への逆変換適用により，**元の予測ロジット・分類精度を完全保存したまま**特徴チャネルを単一概念プロトタイプへ分離できることを検証します．

2. **実験2（[`ex002/`](./ex002/) / レポート：[`report/ex002.md`](./report/ex002.md)）**
   - 室岡 結南, 代 美月, 神野 健哉, **「EPICにおけるチャネル分離度と分類精度の関係に関する検討」**, *2026年 電子情報通信学会ソサイエティ大会*, N-1-43, 2026.
   - Imagenette でファインチューニングした ResNet-34 に対して EPIC を適用し，512チャネルを Purity（純度）に応じて5群（G1〜G5）へ分割した際の単独分類精度・累積追加精度を検証するとともに，同論文で指摘された交絡要因（チャネル数・分類層重みノルム・対応クラスの偏り・低 Purity 群の冗長性）を定量的に解明します．

> 2つの実験を統合した総合技術レポートは **[`report/report.md`](./report/report.md)** を参照してください．

---

## ディレクトリ構成

```text
.
├─ ex001/                  # 実験1: AAAI 2026 EPIC 原論文の再現実験
│  ├─ model.py             # 事前学習バックボーン分解・DisentanglementModule・EPICModel の定義
│  ├─ utils.py             # Purity 計算・EPIC 高速学習・Grad-CAM・Faithfulness 評価関数群
│  └─ ex001.ipynb          # 実験1の実行・可視化ノートブック
│
├─ ex002/                  # 実験2: 電子情報通信学会 N-1-43 論文の再現・発展実験
│  ├─ model.py             # ImagenetteResNet34・ChannelMaskedEPICResNet34 の定義
│  ├─ utils.py             # 高速ファインチューニング・Purity 5群分割・交絡要因解析関数群
│  └─ ex002.ipynb          # 実験2の実行・可視化ノートブック
│
├─ report/                 # 実験レポートおよび結果図表
│  ├─ img/                 # 各レポートで参照される実験結果画像（PNG）
│  ├─ ex001.md             # 実験1の詳細解説・結果・考察レポート
│  ├─ ex002.md             # 実験2の詳細解説・結果・考察レポート
│  └─ report.md            # プロジェクト全体の統合レポート
│
├─ requirements.txt        # Python 依存パッケージ一覧
├─ LICENSE                 # MIT License
└─ README.md               # 本ドキュメント
```

---

## 動作環境とセットアップ手順

### 1. 前提環境
- **OS**: Windows / Linux / macOS
- **Python**: 3.10 以上（動作確認：Python 3.10.11）
- **GPU**: CUDA 対応 NVIDIA GPU 推奨（動作確認：NVIDIA GeForce RTX 4080, PyTorch 2.7.1+cu118）

### 2. 仮想環境の作成とライブラリのインストール

```powershell
# 仮想環境の作成と有効化（Windows PowerShell の場合）
python -m venv venv
.\venv\Scripts\activate

# CUDA 11.8 対応 PyTorch / Torchvision のインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# その他依存パッケージのインストール
pip install -r requirements.txt
```

### 3. データセットの準備
両実験では ImageNet-1K の代表的な10クラス部分集合である **Imagenette (`320px`)**（訓練画像 9,469 枚，検証画像 3,925 枚）を使用します．初回実行時にプロジェクトルート直下の `datasets/` ディレクトリを作成し，`torchvision.datasets.Imagenette` から自動取得できます：

```python
import torchvision.datasets as datasets
datasets.Imagenette(root="./datasets", split="train", size="320px", download=True)
datasets.Imagenette(root="./datasets", split="val", size="320px", download=True)
```

---

## 実験の実行方法

1. **`ex001/ex001.ipynb`（実験1：EPIC の再現実験）**
   - カレントディレクトリを `ex001/` とした状態でノートブックを上から順に実行します．
   - ResNet-18 / ResNet-34 / DenseNet-121 / ResNet-50 に対する EPIC 最適化，予測完全一致性の検証，プロトタイプ可視化，Grad-CAM 比較，および Deletion / Insertion 忠実度評価が実行され，`report/img/ex001_*.png` に図が出力されます．

2. **`ex002/ex002.ipynb`（実験2：チャネル分離度 Purity と分類精度の検証実験）**
   - カレントディレクトリを `ex002/` とした状態でノートブックを上から順に実行します．
   - ResNet-34 の Imagenette ファインチューニング，EPIC 最適化，Purity 5群（G1〜G5）別の単独・累積・除去精度評価，ならびにチャネル数・分類層重みノルム・対応クラス分布に関する統制実験が実行され，`report/img/ex002_*.png` に図が出力されます．

---

## レポート一覧

| レポートファイル | 内容 |
| :--- | :--- |
| **[`report/report.md`](./report/report.md)** | **プロジェクト全体レポート**：EPIC の理論的基盤から Purity と分類性能・冗長性の関係までを統合した総合報告 |
| **[`report/ex001.md`](./report/ex001.md)** | **実験1レポート**：AAAI 2026「EPIC」の再現実験（予測完全保存性・直交制約の効果・Grad-CAM 比較・Faithfulness 評価） |
| **[`report/ex002.md`](./report/ex002.md)** | **実験2レポート**：IEICE 2026「N-1-43」の再現実験および交絡要因（チャネル数・重みノルム・クラス偏り・G5 冗長性）の解明 |

---

## ライセンス

本プロジェクトのソースコードおよびドキュメントは [MIT License](./LICENSE) のもとで公開されています．
