# adversarial_vit_cnn

## 実験概要

Vision Transformer (ViT) と CNN (ResNet34, EfficientNet-B3) の敵対的サンプルに対する内部挙動を比較・分析します。

- データセット: tiny-imagenet
- モデル: ViT-Small, ResNet34, EfficientNet-B3 (事前学習済み)
- 攻撃手法: FGSM, PGD (ε=8/255)
- 評価指標: 特徴量の L2 ノルム, 活性化のスパース性, 予測確率の最大値

## 実験環境

- Miniconda (environment.yml で管理)
- PyTorch Lightning
- Hydra
- TensorBoard / WandB
- torch-attacks

## 実験手順・実行方法

### 1. 準備

1. ImageNet-100 検証データセットを `data/` ディレクトリに配置
   - 例: `data/tiny-imagenet/valid/クラス名/画像ファイル.jpg` の ImageFolder 形式
2. WandB を使う場合は事前に `wandb login` を実行
3. conda 環境をアクティベート
   ```zsh
   conda activate adv_vit_cnn
   ```

### 2. 実験の実行

1. main.py を実行
   ```zsh
   python main.py
   ```
2. Hydra 設定（モデル名・攻撃手法等）を変更したい場合は `configs/config.yaml` を編集

### 3. 結果の確認

- Hydra 出力ディレクトリ（`results/`）の CSV
- `logs/` の TensorBoard
- WandB ダッシュボード（Web）

### 4. モデル・攻撃手法の切り替え例

`configs/config.yaml` の

```yaml
model:
  name: efficientnet
attack:
  method: pgd
```

などを編集

### 5. 実験の流れ

1. クリーンデータで各層の指標を計算
2. FGSM・PGD 敵対的サンプルを生成し、同様に指標を計算
3. Hydra/TensorBoard/WandB に記録

## ディレクトリ構成

- data/: データセット
- configs/: Hydra 設定
- models/: モデル・フック
- attacks/: 敵対的攻撃
- analysis/: 評価・分析
- logs/: 実験出力
