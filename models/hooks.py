import torch
import torchvision.models as models
from typing import Dict, List


# ViT-Small (torchvision: vit_b_16, vit_l_16 しかないため、vit_b_16を代用)
def get_vit_model_and_hooks(num_classes=1000):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    # 出力層をnum_classesに置き換え
    if model.heads.head.out_features != num_classes:
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.eval()
    # ViTのTransformer Block
    blocks = model.encoder.layers
    hook_layers = [blocks[0], blocks[len(blocks) // 2], blocks[-1]]
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    for idx, layer in enumerate(hook_layers):
        layer.register_forward_hook(get_hook(f"block_{idx}"))
    return model, activations


# ResNet34
def get_resnet_model_and_hooks(num_classes=1000):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    # 出力層をnum_classesに置き換え
    if model.fc.out_features != num_classes:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    hook_layers = [model.layer1, model.layer2, model.layer4]
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    for idx, layer in enumerate(hook_layers):
        layer.register_forward_hook(get_hook(f"layer{idx + 1}"))
    return model, activations


# EfficientNet-B3
def get_efficientnet_model_and_hooks(num_classes=1000):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # 出力層をnum_classesに置き換え
    if model.classifier[-1].out_features != num_classes:
        model.classifier[-1] = torch.nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
    model.eval()
    features = model.features
    hook_layers = [features[0], features[6], features[-1]]
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()

        return hook

    for idx, layer in enumerate(hook_layers):
        layer.register_forward_hook(get_hook(f"features_{idx}"))
    return model, activations


# モデル名で取得できるようにまとめる
def get_model_and_hooks(model_name: str, num_classes: int = 1000):
    if model_name == "vit":
        return get_vit_model_and_hooks(num_classes)
    elif model_name == "resnet":
        return get_resnet_model_and_hooks(num_classes)
    elif model_name == "efficientnet":
        return get_efficientnet_model_and_hooks(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
