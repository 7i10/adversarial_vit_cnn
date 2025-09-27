import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from models import hooks
import os
import csv
import wandb
from torch.utils.tensorboard import SummaryWriter
from attacks.attack_generator import generate_adversarial_samples

from analysis import metrics
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


@hydra.main(config_path="configs", config_name="config", version_base=None)

# --- データローダー作成 ---
def create_tiny_imagenet_loaders(batch_size=4):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    def transform_fn(example):
        imgs = example["image"]
        labels = example["label"]
        if isinstance(imgs, list):
            tensor_imgs = [transform(img.convert("RGB")) for img in imgs]
            return {"image": torch.stack(tensor_imgs), "label": torch.tensor(labels)}
        else:
            return {"image": transform(imgs), "label": torch.tensor(labels)}

    train_ds = load_dataset("zh-plus/tiny-imagenet", split="train").with_transform(
        transform_fn
    )
    val_ds = load_dataset("zh-plus/tiny-imagenet", split="valid").with_transform(
        transform_fn
    )

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch])
        return images, labels

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader


# --- ファインチューニング ---
def finetune_classifier(model, train_loader, device, epochs=2):
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
    for epoch in range(epochs):
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
        print(
            f"[Fine-tune][Epoch {epoch + 1}] Loss: {total_loss / total:.4f}, Acc: {correct / total:.4f}"
        )
    model.eval()


# --- 評価ループ ---
def evaluate(
    model, loader, activations, device, writer, cfg, all_results, phase, attack_fn=None
):
    accs = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if attack_fn is not None:
            images, labels = attack_fn(model, images, labels, device)
        with torch.no_grad():
            logits = model(images)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        accs.append(acc)
        for name, feat in activations.items():
            l2 = metrics.feature_l2_norm(feat)
            sp = metrics.activation_sparsity(feat)
            print(f"[{phase}][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
            if writer:
                writer.add_scalar(f"{phase}/{name}/L2", l2)
                writer.add_scalar(f"{phase}/{name}/Sparsity", sp)
            if cfg.logging.use_wandb:
                wandb.log({f"{phase}/{name}/L2": l2, f"{phase}/{name}/Sparsity": sp})
            all_results.append([phase, name, l2, sp, None])
        mp = metrics.max_softmax_prob(logits)
        print(f"[{phase}] Max Softmax Prob: {mp:.4f}, Acc: {acc:.4f}")
        if writer:
            writer.add_scalar(f"{phase}/MaxSoftmaxProb", mp)
            writer.add_scalar(f"{phase}/Accuracy", acc)
        if cfg.logging.use_wandb:
            wandb.log({f"{phase}/MaxSoftmaxProb": mp, f"{phase}/Accuracy": acc})
        all_results.append([phase, "MaxSoftmaxProb", None, None, mp])
    print(f"[{phase}] Mean Accuracy: {sum(accs) / len(accs):.4f}")


# --- メインルーチン ---
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    output_dir = os.getcwd()
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_csv = os.path.join(output_dir, "results", f"result_{cfg.model.name}.csv")
    writer = None
    if cfg.logging.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs", cfg.model.name))
    run_name = f"{cfg.model.name}_{cfg.attack.method}_eps{cfg.attack.epsilon}"
    if cfg.logging.use_wandb:
        wandb.init(project="adv_vit_cnn", name=run_name)
        wandb.config.update(dict(cfg))
    all_results = []
    print("設定:", cfg)
    model, activations = hooks.get_model_and_hooks(
        cfg.model.name, cfg.model.num_classes
    )
    print(f"Loaded model: {cfg.model.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    train_loader, val_loader = create_tiny_imagenet_loaders(
        batch_size=cfg.data.get("batch_size", 4)
    )
    finetune_classifier(model, train_loader, device, epochs=2)
    model.eval()
    # クリーンデータ
    print("--- Clean Data ---")
    evaluate(
        model, val_loader, activations, device, writer, cfg, all_results, phase="Clean"
    )
    # FGSM
    print("--- FGSM ---")

    def fgsm_attack_fn(model, images, labels, device):
        # 1バッチ分だけ攻撃
        adv = next(
            generate_adversarial_samples(
                model, [(images, labels)], "fgsm", cfg.attack.epsilon, device
            )
        )
        return adv[0], adv[1]

    evaluate(
        model,
        val_loader,
        activations,
        device,
        writer,
        cfg,
        all_results,
        phase="FGSM",
        attack_fn=fgsm_attack_fn,
    )
    # PGD
    print("--- PGD ---")

    def pgd_attack_fn(model, images, labels, device):
        adv = next(
            generate_adversarial_samples(
                model, [(images, labels)], "pgd", cfg.attack.epsilon, device
            )
        )
        return adv[0], adv[1]

    evaluate(
        model,
        val_loader,
        activations,
        device,
        writer,
        cfg,
        all_results,
        phase="PGD",
        attack_fn=pgd_attack_fn,
    )
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
