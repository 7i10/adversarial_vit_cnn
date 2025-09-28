# adversarial_vit_cnn

## 実験概要

Vision Transformer (ViT-b-16) と CNN (ResNet34, EfficientNet-B3) の敵対的サンプルに対する内部挙動を比較・分析します。

- **対応データセット:**
  - ImageNet (Deep Lake 経由で自動ダウンロード)
  - Tiny-ImageNet (Hugging Face Datasets 経由で自動ダウンロード)
- **モデル:** ViT-Base, ResNet34, EfficientNet-B3 (ImageNet 事前学習済み)
- **攻撃手法:** FGSM, PGD (ε=8/255)
- **評価指標:** 特徴量の L2 ノルム, 活性化のスパース性, 予測確率の最大値, Accuracy

## 実験環境

- Miniconda (environment.yml で管理)
- PyTorch Lightning
- Hydra (設定管理)
- TensorBoard / WandB (ロギング)
- torch-attacks
- Hugging Face Datasets, Deep Lake

## 実験手順・実行方法

### 1. 準備

1. conda 環境をアクティベート
   ```zsh
   conda activate adv_vit_cnn
   ```
2. WandB を使う場合は事前に `wandb login` を実行

※ データセットは自動でダウンロードされるため、`data/` への手動配置は不要です。

### 2. 実験の実行

- **ImageNet 評価の場合**
  ```zsh
  python imagenet_subset.py
  ```
- **Tiny-ImageNet 評価の場合**
  ```zsh
  python tiny_imagenet.py
  ```

モデル名・攻撃手法・バッチサイズ等は `configs/config.yaml` で設定します。

### 3. 結果の確認

- `results/` の CSV
- `logs/` の TensorBoard
- WandB ダッシュボード（Web）

### 4. モデル・攻撃手法・パラメータの切り替え例

`configs/config.yaml` の例：

```yaml
model:
  name: efficientnet # vit, resnet, efficientnet から選択
attack:
  method: pgd # fgsm, pgd
  epsilon: 0.0314
  pgd_steps: 10
data:
  batch_size: 8
  subset_size: 1000 # ImageNetサブセット用
seed: 42
```

### 5. 実験の流れ（スクリプト共通の処理概要）

1. シード固定・Hydra 設定読込
2. モデル構築（出力層をデータセットに応じて自動置換）
3. データローダー自動作成（ImageNet: Deep Lake, Tiny-ImageNet: Hugging Face）
4. Tiny-ImageNet の場合は train split で出力層のみ簡易ファインチューニング
5. val split またはサブセットで以下を実施：
   - クリーン画像で各層指標・Accuracy 計算
   - FGSM/PGD 敵対的サンプル生成・同様に評価
   - Hydra/TensorBoard/WandB に記録
6. 結果を CSV 保存

## ディレクトリ構成

- configs/: Hydra 設定
- models/: モデル・フック
- attacks/: 敵対的攻撃
- analysis/: 評価・分析
- logs/: 実験出力
- results/: 評価 CSV
- data/: データセット（自動ダウンロード）
- imagenet_subset.py: ImageNet 用評価スクリプト
- tiny_imagenet.py: Tiny-ImageNet 用評価スクリプト
