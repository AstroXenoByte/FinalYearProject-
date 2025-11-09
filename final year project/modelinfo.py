import torch
import torch.nn as nn
import time
import os
from ptflops import get_model_complexity_info


class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=7, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(ViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (B, dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, dim)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x


def load_model(model_path, device):
    model = ViT()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            new_checkpoint[k[7:]] = v
        else:
            new_checkpoint[k] = v
    model.load_state_dict(new_checkpoint)
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def average_inference_time(model, device, input_size=(3, 224, 224), runs=100):
    dummy_input = torch.randn((1, *input_size)).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        start = time.time()
        for _ in range(runs):
            _ = model(dummy_input)
        end = time.time()
    avg_time = (end - start) / runs
    return avg_time


def print_report(model_path, device):
    model = load_model(model_path, device)

    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model_path)

    print(f"Model architecture:\n{model}\n")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model file size: {model_size:.2f} MB")

    avg_time = average_inference_time(model, device)
    print(f"Average inference time per image on {device.type}: {avg_time * 1000:.2f} ms")

    input_res = (3, 224, 224)
    flops, params = get_model_complexity_info(model, input_res, as_strings=True,
                                              print_per_layer_stat=False, verbose=False)
    print(f"Computational complexity (FLOPs): {flops}")
    print(f"Parameters counted by ptflops: {params}")


if __name__ == "__main__":
    model_path = "vit_groundnut_cpu_friendly.pth"
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found in current directory.")
        exit(1)

    print("=== Benchmark on CPU ===")
    device_cpu = torch.device("cpu")
    print_report(model_path, device_cpu)

    if torch.cuda.is_available():
        print("\n=== Benchmark on GPU ===")
        device_gpu = torch.device("cuda")
        print_report(model_path, device_gpu)
