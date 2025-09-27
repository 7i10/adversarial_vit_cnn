name: efficientnet
method: pgd

# adversarial_vit_cnn

## 実験概要

Vision Transformer (ViT-b-16) と CNN (ResNet34, EfficientNet-B3) の敵対的サンプルに対する内部挙動を比較・分析します。

- データセット: Tiny-ImageNet (Hugging Face Datasets から自動ダウンロード)
- モデル: ViT-Base, ResNet34, EfficientNet-B3 (ImageNet 事前学習済み)
- 攻撃手法: FGSM, PGD (ε=8/255)
- 評価指標: 特徴量の L2 ノルム, 活性化のスパース性, 予測確率の最大値, Accuracy

## 実験環境

- Miniconda (environment.yml で管理)
- PyTorch Lightning
- Hydra (設定管理)
- TensorBoard / WandB (ロギング)
- torch-attacks
- Hugging Face Datasets

## 実験手順・実行方法

### 1. 準備

1. conda 環境をアクティベート
   ```zsh
   conda activate adv_vit_cnn
   ```
2. WandB を使う場合は事前に `wandb login` を実行

※ データセットは自動でダウンロードされるため、`data/` への手動配置は不要です。

### 2. 実験の実行

1. main.py を実行
   ```zsh
   python main.py
   ```
2. モデル名・攻撃手法・バッチサイズ等は `configs/config.yaml` で設定

### 3. 結果の確認

- `results/` の CSV
- `logs/` の TensorBoard
- WandB ダッシュボード（Web）

### 4. モデル・攻撃手法・パラメータの切り替え例

`configs/config.yaml` の例：

```yaml
model:
  name: vit
  num_classes: 200
attack:
  method: fgsm
  epsilon: 0.031
data:
  batch_size: 4
seed: 42
```

### 5. 実験の流れ（main.py の処理概要）

1. シード固定・Hydra 設定読込
2. モデル構築（出力層を Tiny-ImageNet 用に自動置換）
3. Tiny-ImageNet の train/val ローダー自動作成
4. train split で出力層のみ 2 エポック簡易ファインチューニング
5. val split で以下を実施：
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
