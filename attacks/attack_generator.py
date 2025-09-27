import torch
from torch.utils.data import DataLoader
import torchattacks
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np


def get_tiny_imagenet_loader(batch_size=4):
    # Hugging Face datasetsからvalデータ取得
    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    # datasetsのitemはdict: {'image': PIL.Image, 'label': int}
    def transform_fn(example):
        imgs = example["image"]
        labels = example["label"]
        # バッチの場合（リスト）
        if isinstance(imgs, list):
            tensor_imgs = [transform(img.convert("RGB")) for img in imgs]
            return {"image": torch.stack(tensor_imgs), "label": torch.tensor(labels)}
        # 1枚の場合（PIL.Image）
        else:
            return {"image": transform(imgs), "label": torch.tensor(labels)}

    dataset = dataset.with_transform(transform_fn)

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch])
        return images, labels

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return loader


def generate_adversarial_samples(
    model, loader, attack_name="fgsm", epsilon=8 / 255, device="cuda"
):
    model.to(device)
    model.eval()
    if attack_name == "fgsm":
        attack = torchattacks.FGSM(model, eps=epsilon)
    elif attack_name == "pgd":
        attack = torchattacks.PGD(model, eps=epsilon)
    else:
        raise ValueError("Unknown attack")
    # バッチ逐次処理型に変更
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        adv = attack(images, targets)
        yield adv, targets
