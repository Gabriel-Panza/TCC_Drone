"""Exporta um checkpoint Depth Anything V2 Metric para ONNX FP32."""

import argparse
import sys
import types
from pathlib import Path

import onnx
import torch


def install_torchvision_compose_stub():
    """Fornece apenas Compose, unica parte de torchvision importada pelo DPT."""

    class Compose:
        def __init__(self, transforms):
            self.transforms = list(transforms)

        def __call__(self, value):
            for transform in self.transforms:
                value = transform(value)
            return value

    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    transforms.Compose = Compose
    torchvision.transforms = transforms
    sys.modules.setdefault("torchvision", torchvision)
    sys.modules.setdefault("torchvision.transforms", transforms)


MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_model(source_dir, checkpoint, encoder, max_depth):
    install_torchvision_compose_stub()
    metric_root = source_dir / "metric_depth"
    sys.path.insert(0, str(metric_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(
        encoder=encoder,
        features=config["features"],
        out_channels=config["out_channels"],
        max_depth=max_depth,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=686)
    parser.add_argument("--input-height", type=int, default=518)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--encoder", choices=MODEL_CONFIGS, default="vits")
    parser.add_argument("--max-depth", type=float, default=80.0)
    args = parser.parse_args()

    model = load_model(
        args.source_dir, args.checkpoint, args.encoder, args.max_depth
    )
    example = torch.zeros(
        1,
        3,
        args.input_height,
        args.input_width,
        dtype=torch.float32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            args.output,
            input_names=["image"],
            output_names=["depth_m"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    print(f"ONNX valido: {args.output}")
    print(f"Entrada: [1, 3, {args.input_height}, {args.input_width}]")
    print(
        "Saida: profundidade metrica em metros; "
        f"encoder={args.encoder}; max_depth={args.max_depth:g} m"
    )


if __name__ == "__main__":
    main()
