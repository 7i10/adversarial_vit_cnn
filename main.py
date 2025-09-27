import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from models import hooks
import os
import csv
import wandb
from torch.utils.tensorboard import SummaryWriter
from attacks.attack_generator import (
    get_tiny_imagenet_loader,
    generate_adversarial_samples,
)
from analysis import metrics


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # シード設定
    pl.seed_everything(cfg.seed)

    # Hydra出力ディレクトリ
    output_dir = os.getcwd()
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_csv = os.path.join(output_dir, "results", f"result_{cfg.model.name}.csv")

    # TensorBoard/WandB初期化
    writer = None
    if cfg.logging.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs", cfg.model.name))
    run_name = f"{cfg.model.name}_{cfg.attack.method}_eps{cfg.attack.epsilon}"
    if cfg.logging.use_wandb:
        wandb.init(project="adv_vit_cnn", name=run_name)
        wandb.config.update(dict(cfg))

    # 結果記録用リスト
    all_results = []
    print("設定:", cfg)

    # モデルとフックの初期化
    model, activations = hooks.get_model_and_hooks(
        cfg.model.name, cfg.model.num_classes
    )
    print(f"Loaded model: {cfg.model.name}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # データローダー
    # ファインチューニング用trainと実験用valを分けて取得
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    # 変換
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    def transform_fn(example):
        imgs = example["image"]
        labels = example["label"]
        if isinstance(imgs, list):
            tensor_imgs = [transform(img.convert("RGB")) for img in imgs]
            return {"image": torch.stack(tensor_imgs), "label": torch.tensor(labels)}
        else:
            return {"image": transform(imgs), "label": torch.tensor(labels)}
    # train/valデータセット
    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train").with_transform(transform_fn)
    val_ds = load_dataset("zh-plus/tiny-imagenet", split="valid").with_transform(transform_fn)
    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch])
        return images, labels
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.get("batch_size", 4), shuffle=False, collate_fn=collate_fn)

    # --- ファインチューニング（出力層のみ） ---
    print("--- Fine-tuning classifier head on Tiny-ImageNet train split ---")
    for param in model.parameters():
        param.requires_grad = False
    # 出力層のみ学習可
    if hasattr(model, "fc"):
        model.fc.requires_grad_(True)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    elif hasattr(model, "classifier"):
        model.classifier[-1].requires_grad_(True)
        optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=1e-3)
    elif hasattr(model, "heads"):
        model.heads.head.requires_grad_(True)
        optimizer = torch.optim.Adam(model.heads.head.parameters(), lr=1e-3)
    else:
        raise RuntimeError("Unknown model head for fine-tuning")
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(2):  # 2エポックだけ
        total, correct, total_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        print(f"[Fine-tune][Epoch {epoch+1}] Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")
    model.eval()

    # --- ここから従来のval splitでの実験 ---
    loader = val_loader

    # クリーンデータ（全バッチ評価 & Accuracy記録）
    print("--- Clean Data ---")
    clean_accs = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        clean_accs.append(acc)
        for name, feat in activations.items():
            l2 = metrics.feature_l2_norm(feat)
            sp = metrics.activation_sparsity(feat)
            print(f"[Clean][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
            if writer:
                writer.add_scalar(f"Clean/{name}/L2", l2)
                writer.add_scalar(f"Clean/{name}/Sparsity", sp)
            if cfg.logging.use_wandb:
                wandb.log({f"Clean/{name}/L2": l2, f"Clean/{name}/Sparsity": sp})
            all_results.append(["Clean", name, l2, sp, None])
        mp = metrics.max_softmax_prob(logits)
        print(f"[Clean] Max Softmax Prob: {mp:.4f}, Acc: {acc:.4f}")
        if writer:
            writer.add_scalar("Clean/MaxSoftmaxProb", mp)
            writer.add_scalar("Clean/Accuracy", acc)
        if cfg.logging.use_wandb:
            wandb.log({"Clean/MaxSoftmaxProb": mp, "Clean/Accuracy": acc})
        all_results.append(["Clean", "MaxSoftmaxProb", None, None, mp])
    # 平均Accuracy表示
    print(f"[Clean] Mean Accuracy: {sum(clean_accs) / len(clean_accs):.4f}")

    # FGSM（全バッチ評価 & Accuracy記録 & 層ごとkeyでログ）
    print("--- FGSM ---")
    fgsm_accs = []
    for adv_images, adv_labels in generate_adversarial_samples(
        model, loader, "fgsm", cfg.attack.epsilon, device
    ):
        with torch.no_grad():
            logits = model(adv_images)
        acc = (logits.argmax(dim=1) == adv_labels).float().mean().item()
        fgsm_accs.append(acc)
        for name, feat in activations.items():
            l2 = metrics.feature_l2_norm(feat)
            sp = metrics.activation_sparsity(feat)
            print(f"[FGSM][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
            if writer:
                writer.add_scalar(f"FGSM/{name}/L2", l2)
                writer.add_scalar(f"FGSM/{name}/Sparsity", sp)
            if cfg.logging.use_wandb:
                wandb.log({f"FGSM/{name}/L2": l2, f"FGSM/{name}/Sparsity": sp})
            all_results.append(["FGSM", name, l2, sp, None])
        mp = metrics.max_softmax_prob(logits)
        print(f"[FGSM] Max Softmax Prob: {mp:.4f}, Acc: {acc:.4f}")
        if writer:
            writer.add_scalar("FGSM/MaxSoftmaxProb", mp)
            writer.add_scalar("FGSM/Accuracy", acc)
        if cfg.logging.use_wandb:
            wandb.log({"FGSM/MaxSoftmaxProb": mp, "FGSM/Accuracy": acc})
        all_results.append(["FGSM", "MaxSoftmaxProb", None, None, mp])
    print(f"[FGSM] Mean Accuracy: {sum(fgsm_accs) / len(fgsm_accs):.4f}")

    # PGD（全バッチ評価 & Accuracy記録 & 層ごとkeyでログ）
    print("--- PGD ---")
    pgd_accs = []
    for adv_images, adv_labels in generate_adversarial_samples(
        model, loader, "pgd", cfg.attack.epsilon, device
    ):
        with torch.no_grad():
            logits = model(adv_images)
        acc = (logits.argmax(dim=1) == adv_labels).float().mean().item()
        pgd_accs.append(acc)
        for name, feat in activations.items():
            l2 = metrics.feature_l2_norm(feat)
            sp = metrics.activation_sparsity(feat)
            print(f"[PGD][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
            if writer:
                writer.add_scalar(f"PGD/{name}/L2", l2)
                writer.add_scalar(f"PGD/{name}/Sparsity", sp)
            if cfg.logging.use_wandb:
                wandb.log({f"PGD/{name}/L2": l2, f"PGD/{name}/Sparsity": sp})
            all_results.append(["PGD", name, l2, sp, None])
        mp = metrics.max_softmax_prob(logits)
        print(f"[PGD] Max Softmax Prob: {mp:.4f}, Acc: {acc:.4f}")
        if writer:
            writer.add_scalar("PGD/MaxSoftmaxProb", mp)
            writer.add_scalar("PGD/Accuracy", acc)
        if cfg.logging.use_wandb:
            wandb.log({"PGD/MaxSoftmaxProb": mp, "PGD/Accuracy": acc})
        all_results.append(["PGD", "MaxSoftmaxProb", None, None, mp])
    print(f"[PGD] Mean Accuracy: {sum(pgd_accs) / len(pgd_accs):.4f}")

    # CSV保存
    with open(result_csv, "w") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Type", "Layer", "L2Norm", "Sparsity", "MaxSoftmaxProb"])
        writer_csv.writerows(all_results)

    if writer:
        writer.close()
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
