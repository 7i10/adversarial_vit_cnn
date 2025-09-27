import torchattacks


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
