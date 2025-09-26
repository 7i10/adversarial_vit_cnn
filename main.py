import hydra
from omegaconf import DictConfig
import torch
from models import hooks

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    import os
    import csv
    from torch.utils.tensorboard import SummaryWriter
    import wandb

    # Hydra出力ディレクトリ
    output_dir = os.getcwd()
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    result_csv = os.path.join(output_dir, "results", f"result_{cfg.model.name}.csv")

    # TensorBoard/WandB初期化
    writer = None
    if cfg.logging.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs", cfg.model.name))
    if cfg.logging.use_wandb:
        wandb.init(project="adv_vit_cnn", name=cfg.model.name)

    # 結果記録用リスト
    all_results = []
    print("設定:", cfg)
    # モデルとフックの初期化
    model, activations = hooks.get_model_and_hooks(cfg.model.name)
    print(f"Loaded model: {cfg.model.name}")

    from attacks.attack_generator import get_tiny_imagenet_loader, generate_adversarial_samples
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    from analysis import metrics

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # データローダー
    loader = get_tiny_imagenet_loader()

    # クリーンデータ
    print("--- Clean Data ---")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(images)
        for name, feat in activations.items():
            l2 = metrics.feature_l2_norm(feat)
            sp = metrics.activation_sparsity(feat)
            print(f"[Clean][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
            if writer:
                writer.add_scalar(f"Clean/{name}/L2", l2)
                writer.add_scalar(f"Clean/{name}/Sparsity", sp)
            if cfg.logging.use_wandb:
                wandb.log({f"Clean_{name}_L2": l2, f"Clean_{name}_Sparsity": sp})
            all_results.append(["Clean", name, l2, sp, None])
        mp = metrics.max_softmax_prob(logits)
        print(f"[Clean] Max Softmax Prob: {mp:.4f}")
        if writer:
            writer.add_scalar("Clean/MaxSoftmaxProb", mp)
        if cfg.logging.use_wandb:
            wandb.log({"Clean_MaxSoftmaxProb": mp})
        all_results.append(["Clean", "MaxSoftmaxProb", None, None, mp])
        break

    # FGSM
    print("--- FGSM ---")
    adv_images, adv_labels = generate_adversarial_samples(model, loader, 'fgsm', cfg.attack.epsilon, device)
    with torch.no_grad():
        logits = model(adv_images.to(device))
    for name, feat in activations.items():
        l2 = metrics.feature_l2_norm(feat)
        sp = metrics.activation_sparsity(feat)
        print(f"[FGSM][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
        if writer:
            writer.add_scalar(f"FGSM/{name}/L2", l2)
            writer.add_scalar(f"FGSM/{name}/Sparsity", sp)
        if cfg.logging.use_wandb:
            wandb.log({f"FGSM_{name}_L2": l2, f"FGSM_{name}_Sparsity": sp})
        all_results.append(["FGSM", name, l2, sp, None])
    mp = metrics.max_softmax_prob(logits)
    print(f"[FGSM] Max Softmax Prob: {mp:.4f}")
    if writer:
        writer.add_scalar("FGSM/MaxSoftmaxProb", mp)
    if cfg.logging.use_wandb:
        wandb.log({"FGSM_MaxSoftmaxProb": mp})
    all_results.append(["FGSM", "MaxSoftmaxProb", None, None, mp])

    # PGD
    print("--- PGD ---")
    adv_images, adv_labels = generate_adversarial_samples(model, loader, 'pgd', cfg.attack.epsilon, device)
    with torch.no_grad():
        logits = model(adv_images.to(device))
    for name, feat in activations.items():
        l2 = metrics.feature_l2_norm(feat)
        sp = metrics.activation_sparsity(feat)
        print(f"[PGD][{name}] L2: {l2:.4f}, Sparsity: {sp:.4f}")
        if writer:
            writer.add_scalar(f"PGD/{name}/L2", l2)
            writer.add_scalar(f"PGD/{name}/Sparsity", sp)
        if cfg.logging.use_wandb:
            wandb.log({f"PGD_{name}_L2": l2, f"PGD_{name}_Sparsity": sp})
        all_results.append(["PGD", name, l2, sp, None])
    mp = metrics.max_softmax_prob(logits)
    print(f"[PGD] Max Softmax Prob: {mp:.4f}")
    if writer:
        writer.add_scalar("PGD/MaxSoftmaxProb", mp)
    if cfg.logging.use_wandb:
        wandb.log({"PGD_MaxSoftmaxProb": mp})
    all_results.append(["PGD", "MaxSoftmaxProb", None, None, mp])

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
