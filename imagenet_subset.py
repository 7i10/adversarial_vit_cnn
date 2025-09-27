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
import numpy as np


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # シード固定
    pl.seed_everything(cfg.seed)
    output_dir = os.getcwd()
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_csv = os.path.join(
        output_dir, "results", f"result_{cfg.model.name}_imagenet.csv"
    )
    writer = None
    if cfg.logging.use_tensorboard:
        writer = SummaryWriter(
            log_dir=os.path.join(output_dir, "logs", cfg.model.name + "_imagenet")
        )
    run_name = f"{cfg.model.name}_eps{cfg.attack.epsilon}_imagenet"
    if cfg.logging.use_wandb:
        wandb.init(project="adv_vit_cnn_imagenet", name=run_name)
        wandb.config.update(dict(cfg))
    all_results = []
    print("設定:", cfg)
    model, activations = hooks.get_model_and_hooks(cfg.model.name, 1000)
    print(f"Loaded model: {cfg.model.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # --- ImageNetサブセットデータローダー ---

    def create_imagenet_subset_loader(batch_size, subset_size, seed=42):
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        def transform_fn(example):
            img = example["image"]
            label = example["label"]
            return {
                "image": transform(img.convert("RGB")),
                "label": torch.tensor(label),
            }

        # validation splitを使い、shuffle+selectでサブセットを作成
        full_ds = load_dataset("ILSVRC/imagenet-1k", split="validation")
        subset_ds = full_ds.shuffle(seed=seed).select(range(subset_size))
        ds = subset_ds.with_transform(transform_fn)

        def collate_fn(batch):
            images = torch.stack([b["image"] for b in batch])
            labels = torch.tensor([b["label"] for b in batch])
            return images, labels

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        return loader

    loader = create_imagenet_subset_loader(
        batch_size=cfg.data.batch_size, subset_size=cfg.data.subset_size, seed=cfg.seed
    )

    # --- 評価ループ ---
    def evaluate(
        model,
        loader,
        activations,
        device,
        writer,
        cfg,
        all_results,
        phase,
        attack_fn=None,
    ):
        accs = []
        l2s = {name: [] for name in activations}
        sparsities = {name: [] for name in activations}
        max_softmax_probs = []
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
                l2s[name].append(l2)
                sparsities[name].append(sp)
                print(f"[{phase}][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
                if writer:
                    writer.add_scalar(f"{phase}/{name}/L2", l2)
                    writer.add_scalar(f"{phase}/{name}/Sparsity", sp)
                if cfg.logging.use_wandb:
                    wandb.log(
                        {f"{phase}/{name}/L2": l2, f"{phase}/{name}/Sparsity": sp}
                    )
                all_results.append([phase, name, l2, sp, None])
            mp = metrics.max_softmax_prob(logits)
            max_softmax_probs.append(mp)
            print(f"[{phase}] Max Softmax Prob: {mp:.4f}, Acc: {acc:.4f}")
            if writer:
                writer.add_scalar(f"{phase}/MaxSoftmaxProb", mp)
                writer.add_scalar(f"{phase}/Accuracy", acc)
            if cfg.logging.use_wandb:
                wandb.log({f"{phase}/MaxSoftmaxProb": mp, f"{phase}/Accuracy": acc})
            all_results.append([phase, "MaxSoftmaxProb", None, None, mp])
        # 平均・分散を計算
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_mp = float(np.mean(max_softmax_probs))
        std_mp = float(np.std(max_softmax_probs))
        print(f"[{phase}] Mean Accuracy: {mean_acc:.4f} (std: {std_acc:.4f})")
        print(f"[{phase}] Mean MaxSoftmaxProb: {mean_mp:.4f} (std: {std_mp:.4f})")
        if cfg.logging.use_wandb:
            wandb.log(
                {
                    f"{phase}/MeanAccuracy": mean_acc,
                    f"{phase}/StdAccuracy": std_acc,
                    f"{phase}/MeanMaxSoftmaxProb": mean_mp,
                    f"{phase}/StdMaxSoftmaxProb": std_mp,
                }
            )
        for name in activations:
            l2_arr = np.array(l2s[name])
            sp_arr = np.array(sparsities[name])
            mean_l2, std_l2 = float(l2_arr.mean()), float(l2_arr.std())
            mean_sp, std_sp = float(sp_arr.mean()), float(sp_arr.std())
            print(
                f"[{phase}][{name}] Mean L2: {mean_l2:.4f} (std: {std_l2:.4f}), Mean Sparsity: {mean_sp:.4f} (std: {std_sp:.4f})"
            )
            if cfg.logging.use_wandb:
                wandb.log(
                    {
                        f"{phase}/{name}/MeanL2": mean_l2,
                        f"{phase}/{name}/StdL2": std_l2,
                        f"{phase}/{name}/MeanSparsity": mean_sp,
                        f"{phase}/{name}/StdSparsity": std_sp,
                    }
                )

    # --- クリーンデータ ---
    print("--- Clean Data ---")
    evaluate(
        model, loader, activations, device, writer, cfg, all_results, phase="Clean"
    )

    # --- FGSM ---
    print("--- FGSM ---")

    def fgsm_attack_fn(model, images, labels, device):
        adv = next(
            generate_adversarial_samples(
                model, [(images, labels)], "fgsm", cfg.attack.epsilon, device
            )
        )
        return adv[0], adv[1]

    evaluate(
        model,
        loader,
        activations,
        device,
        writer,
        cfg,
        all_results,
        phase="FGSM",
        attack_fn=fgsm_attack_fn,
    )

    # --- PGD ---
    print("--- PGD ---")

    def pgd_attack_fn(model, images, labels, device):
        adv = next(
            generate_adversarial_samples(
                model,
                [(images, labels)],
                "pgd",
                cfg.attack.epsilon,
                device,
                steps=cfg.attack.pgd_steps,
            )
        )
        return adv[0], adv[1]

    evaluate(
        model,
        loader,
        activations,
        device,
        writer,
        cfg,
        all_results,
        phase="PGD",
        attack_fn=pgd_attack_fn,
    )

    # --- CSV保存 ---
    header = ["Type", "Layer", "L2Norm", "Sparsity", "MaxSoftmaxProb"]
    with open(result_csv, "w") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(header)
        writer_csv.writerows(all_results)
    # --- WandB Table保存 ---
    if cfg.logging.use_wandb:
        table = wandb.Table(columns=header, data=all_results)
        wandb.log({"all_results": table})
        wandb.finish()


if __name__ == "__main__":
    main()
