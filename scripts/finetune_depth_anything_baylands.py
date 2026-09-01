#!/usr/bin/env python3
"""Fine-tune Depth Anything V2 Metric Small with recorded Baylands pairs."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


MODEL_WIDTH = 686
MODEL_HEIGHT = 518
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class SpatialFrameDataset(Dataset):
    def __init__(self, runs, *, training, extra_frames=()):
        candidates = [
            frame
            for run in runs
            for frame in sorted((run / "frames").glob("frame_*.npz"))
        ]
        candidates.extend(Path(frame) for frame in extra_frames)
        self.frames = []
        self.invalid_frames = []
        for frame in candidates:
            with np.load(frame) as sample:
                valid = (
                    sample["rgb_bgr"].size > 0
                    and sample["reference_depth_m"].size > 0
                )
            (self.frames if valid else self.invalid_frames).append(frame)
        if not self.frames:
            raise FileNotFoundError("nenhum frame espacial valido encontrado")
        self.training = training

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        with np.load(self.frames[index]) as sample:
            bgr = sample["rgb_bgr"]
            depth = sample["reference_depth_m"].astype(np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(
            depth, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        if self.training:
            if random.random() < 0.5:
                rgb = rgb[:, ::-1]
                depth = depth[:, ::-1]
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            gamma = random.uniform(0.9, 1.1)
            mean_rgb = np.mean(rgb, axis=(0, 1), keepdims=True)
            rgb = np.clip((rgb - mean_rgb) * contrast + mean_rgb, 0.0, 1.0)
            rgb = np.clip(rgb * brightness, 0.0, 1.0) ** gamma
        image = np.transpose((rgb - MEAN) / STD, (2, 0, 1)).copy()
        return torch.from_numpy(image), torch.from_numpy(depth.copy())


def masked_loss(
    prediction,
    target,
    *,
    edge_multiplier=2.0,
    unsafe_tail_weight=0.12,
    unsafe_max_depth_m=20.0,
    unsafe_overestimate_weight=0.20,
    edge_gradient_weight=0.10,
    unsafe_overestimate_tail_weight=0.20,
    unsafe_overestimate_tail_fraction=0.05,
    edge_overestimate_weight=0.25,
    unsafe_underestimate_weight=0.20,
    unsafe_underestimate_tail_weight=0.20,
    unsafe_underestimate_tail_fraction=0.05,
    edge_underestimate_weight=0.25,
):
    valid = torch.isfinite(target) & (target >= 0.5) & (target <= 30.0)
    prediction = prediction.clamp_min(0.05)
    safe_target = target.clamp_min(0.5)

    gradient_x = torch.zeros_like(target)
    gradient_y = torch.zeros_like(target)
    gradient_x[..., :, 1:] = torch.abs(
        target[..., :, 1:] - target[..., :, :-1]
    )
    gradient_y[..., 1:, :] = torch.abs(
        target[..., 1:, :] - target[..., :-1, :]
    )
    relative_gradient = torch.maximum(gradient_x, gradient_y) / safe_target
    edge_weight = 1.0 + edge_multiplier * torch.clamp(
        relative_gradient / 0.10,
        0.0,
        1.0,
    )

    log_error = torch.log(prediction[valid]) - torch.log(safe_target[valid])
    silog = torch.sqrt(
        torch.mean(log_error.square()) - 0.5 * torch.mean(log_error).square()
        + 1e-8
    )
    relative_error = torch.abs(prediction - target) / safe_target.clamp_min(1.0)
    valid_weights = edge_weight[valid]
    weighted_relative_l1 = torch.sum(
        relative_error[valid] * valid_weights
    ) / torch.sum(valid_weights)

    near = valid & (target <= 5.0)
    unsafe_positive_error = (
        torch.relu(prediction[near] - target[near]) * edge_weight[near]
    )
    if unsafe_positive_error.numel():
        tail_size = max(1, unsafe_positive_error.numel() // 10)
        unsafe_tail = torch.topk(unsafe_positive_error, tail_size).values.mean()
    else:
        unsafe_tail = prediction.new_zeros(())

    danger = valid & (target <= unsafe_max_depth_m)
    positive_relative_error = torch.relu(prediction - target) / safe_target
    danger_weights = edge_weight[danger]
    if danger_weights.numel():
        unsafe_overestimate = torch.sum(
            positive_relative_error[danger] * danger_weights
        ) / torch.sum(danger_weights)
    else:
        unsafe_overestimate = prediction.new_zeros(())

    danger_values = (
        positive_relative_error[danger] * danger_weights
        if danger_weights.numel()
        else prediction.new_empty((0,))
    )
    if danger_values.numel():
        tail_size = max(
            1,
            int(danger_values.numel() * unsafe_overestimate_tail_fraction),
        )
        unsafe_overestimate_tail = torch.topk(
            danger_values,
            tail_size,
        ).values.mean()
    else:
        unsafe_overestimate_tail = prediction.new_zeros(())

    edge_mask = valid & (relative_gradient >= 0.10)
    edge_danger_weights = edge_weight[edge_mask]
    if edge_danger_weights.numel():
        edge_overestimate = torch.sum(
            positive_relative_error[edge_mask] * edge_danger_weights
        ) / torch.sum(edge_danger_weights)
    else:
        edge_overestimate = prediction.new_zeros(())

    negative_relative_error = torch.relu(target - prediction) / safe_target
    if danger_weights.numel():
        unsafe_underestimate = torch.sum(
            negative_relative_error[danger] * danger_weights
        ) / torch.sum(danger_weights)
    else:
        unsafe_underestimate = prediction.new_zeros(())

    underestimate_values = (
        negative_relative_error[danger] * danger_weights
        if danger_weights.numel()
        else prediction.new_empty((0,))
    )
    if underestimate_values.numel():
        tail_size = max(
            1,
            int(underestimate_values.numel() * unsafe_underestimate_tail_fraction),
        )
        unsafe_underestimate_tail = torch.topk(
            underestimate_values,
            tail_size,
        ).values.mean()
    else:
        unsafe_underestimate_tail = prediction.new_zeros(())

    if edge_danger_weights.numel():
        edge_underestimate = torch.sum(
            negative_relative_error[edge_mask] * edge_danger_weights
        ) / torch.sum(edge_danger_weights)
    else:
        edge_underestimate = prediction.new_zeros(())

    log_prediction = torch.log(prediction)
    log_target = torch.log(safe_target)
    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    gradient_errors = []
    if torch.any(valid_x):
        predicted_gradient_x = (
            log_prediction[..., :, 1:] - log_prediction[..., :, :-1]
        )
        target_gradient_x = log_target[..., :, 1:] - log_target[..., :, :-1]
        gradient_errors.append(
            torch.abs(predicted_gradient_x[valid_x] - target_gradient_x[valid_x])
        )
    if torch.any(valid_y):
        predicted_gradient_y = (
            log_prediction[..., 1:, :] - log_prediction[..., :-1, :]
        )
        target_gradient_y = log_target[..., 1:, :] - log_target[..., :-1, :]
        gradient_errors.append(
            torch.abs(predicted_gradient_y[valid_y] - target_gradient_y[valid_y])
        )
    edge_gradient_loss = (
        torch.cat(gradient_errors).mean()
        if gradient_errors
        else prediction.new_zeros(())
    )
    return (
        silog
        + 0.30 * weighted_relative_l1
        + unsafe_tail_weight * unsafe_tail
        + unsafe_overestimate_weight * unsafe_overestimate
        + unsafe_overestimate_tail_weight * unsafe_overestimate_tail
        + edge_overestimate_weight * edge_overestimate
        + unsafe_underestimate_weight * unsafe_underestimate
        + unsafe_underestimate_tail_weight * unsafe_underestimate_tail
        + edge_underestimate_weight * edge_underestimate
        + edge_gradient_weight * edge_gradient_loss
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    totals = {"absolute": 0.0, "square": 0.0, "relative": 0.0, "bias": 0.0}
    near_bias_total = 0.0
    near_positive_errors = []
    near_pixels = 0
    pixels = 0
    dangerous_far_pixels = 0
    dangerous_near_pixels = 0
    edge_pixels = 0
    dangerous_far_edge_pixels = 0
    dangerous_near_edge_pixels = 0
    model.eval()
    for image, target in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(image)
        prediction = functional.interpolate(
            prediction[:, None], target.shape[-2:], mode="bilinear", align_corners=True
        )[:, 0]
        valid = torch.isfinite(target) & (target >= 0.5) & (target <= 30.0)
        error = prediction[valid] - target[valid]
        ref = target[valid]
        count = int(error.numel())
        totals["absolute"] += torch.abs(error).sum().item()
        totals["square"] += error.square().sum().item()
        totals["relative"] += (torch.abs(error) / ref).sum().item()
        totals["bias"] += error.sum().item()
        pixels += count
        near = valid & (target <= 5.0)
        near_error = prediction[near] - target[near]
        near_bias_total += near_error.sum().item()
        near_positive_errors.append(torch.relu(near_error).cpu())
        near_pixels += int(torch.count_nonzero(near))
        dangerous_far_pixels += int(
            torch.count_nonzero(valid & ((prediction - target) > 1.0))
        )
        dangerous_near_pixels += int(
            torch.count_nonzero(valid & ((target - prediction) > 1.0))
        )
        gradient_x = torch.zeros_like(target)
        gradient_y = torch.zeros_like(target)
        gradient_x[..., :, 1:] = torch.abs(
            target[..., :, 1:] - target[..., :, :-1]
        )
        gradient_y[..., 1:, :] = torch.abs(
            target[..., 1:, :] - target[..., :-1, :]
        )
        edge = valid & (
            torch.maximum(gradient_x, gradient_y)
            / target.clamp_min(0.5)
            >= 0.10
        )
        edge_pixels += int(torch.count_nonzero(edge))
        dangerous_far_edge_pixels += int(
            torch.count_nonzero(edge & ((prediction - target) > 1.0))
        )
        dangerous_near_edge_pixels += int(
            torch.count_nonzero(edge & ((target - prediction) > 1.0))
        )
    near_positive_p95_m = float(
        torch.quantile(torch.cat(near_positive_errors), 0.95).item()
    )
    return {
        "mae_m": totals["absolute"] / pixels,
        "rmse_m": (totals["square"] / pixels) ** 0.5,
        "abs_rel": totals["relative"] / pixels,
        "bias_m": totals["bias"] / pixels,
        "near_bias_m": near_bias_total / near_pixels,
        "near_positive_error_p95_m": near_positive_p95_m,
        "dangerous_far_rate_1m": (
            dangerous_far_pixels / pixels if pixels else 0.0
        ),
        "dangerous_far_edge_rate_1m": (
            dangerous_far_edge_pixels / edge_pixels if edge_pixels else 0.0
        ),
        "dangerous_near_rate_1m": (
            dangerous_near_pixels / pixels if pixels else 0.0
        ),
        "dangerous_near_edge_rate_1m": (
            dangerous_near_edge_pixels / edge_pixels if edge_pixels else 0.0
        ),
        "pixels": pixels,
    }


def configure_encoder_training(
    model,
    *,
    freeze_encoder=False,
    unfreeze_encoder_blocks=0,
):
    """Seleciona encoder inteiro, congelado, ou apenas seus ultimos blocos."""

    if unfreeze_encoder_blocks < 0:
        raise ValueError("unfreeze_encoder_blocks nao pode ser negativo")
    if freeze_encoder and unfreeze_encoder_blocks:
        raise ValueError(
            "freeze_encoder e unfreeze_encoder_blocks sao mutuamente exclusivos"
        )
    named_encoder = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("pretrained.")
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("pretrained.")
    ]
    if freeze_encoder or unfreeze_encoder_blocks:
        for _, parameter in named_encoder:
            parameter.requires_grad_(False)
    if unfreeze_encoder_blocks:
        indices = sorted(
            {
                int(name.split(".")[2])
                for name, _ in named_encoder
                if name.startswith("pretrained.blocks.")
            }
        )
        if unfreeze_encoder_blocks > len(indices):
            raise ValueError(
                "unfreeze_encoder_blocks excede a quantidade de blocos"
            )
        selected = set(indices[-unfreeze_encoder_blocks:])
        for name, parameter in named_encoder:
            if (
                name.startswith("pretrained.blocks.")
                and int(name.split(".")[2]) in selected
            ):
                parameter.requires_grad_(True)
    encoder_parameters = [
        parameter for _, parameter in named_encoder if parameter.requires_grad
    ]
    return encoder_parameters, head_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, action="append", required=True)
    parser.add_argument("--focus-run", type=Path, action="append", default=[])
    parser.add_argument("--focus-repeat", type=int, default=4)
    parser.add_argument("--focus-frame", type=Path, action="append", default=[])
    parser.add_argument("--focus-frame-repeat", type=int, default=8)
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--encoder", choices=("vits", "vitb"), default="vits")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--unfreeze-encoder-blocks", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-7)
    parser.add_argument("--edge-multiplier", type=float, default=2.0)
    parser.add_argument("--unsafe-tail-weight", type=float, default=0.12)
    parser.add_argument("--unsafe-max-depth-m", type=float, default=20.0)
    parser.add_argument("--unsafe-overestimate-weight", type=float, default=0.20)
    parser.add_argument("--unsafe-overestimate-tail-weight", type=float, default=0.20)
    parser.add_argument("--unsafe-overestimate-tail-fraction", type=float, default=0.05)
    parser.add_argument("--edge-overestimate-weight", type=float, default=0.25)
    parser.add_argument("--unsafe-underestimate-weight", type=float, default=0.20)
    parser.add_argument("--unsafe-underestimate-tail-weight", type=float, default=0.20)
    parser.add_argument("--unsafe-underestimate-tail-fraction", type=float, default=0.05)
    parser.add_argument("--edge-underestimate-weight", type=float, default=0.25)
    parser.add_argument("--edge-gradient-weight", type=float, default=0.10)
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.focus_repeat < 1:
        parser.error("--focus-repeat deve ser positivo")
    if args.unfreeze_encoder_blocks < 0:
        parser.error("--unfreeze-encoder-blocks nao pode ser negativo")
    if args.freeze_encoder and args.unfreeze_encoder_blocks:
        parser.error(
            "--freeze-encoder e --unfreeze-encoder-blocks sao incompatíveis"
        )
    if args.focus_frame_repeat < 1:
        parser.error("--focus-frame-repeat deve ser positivo")
    missing_focus_frames = [
        frame for frame in args.focus_frame if not frame.is_file()
    ]
    if missing_focus_frames:
        parser.error(
            "focus frames inexistentes: "
            + ", ".join(str(frame) for frame in missing_focus_frames)
        )
    train_unique = {run.resolve() for run in [*args.train_run, *args.focus_run]}
    validation_unique = {run.resolve() for run in args.validation_run}
    overlap = train_unique & validation_unique
    if overlap:
        parser.error(
            "runs presentes simultaneamente em treino e validacao: "
            + ", ".join(str(run) for run in sorted(overlap))
        )
    focus_validation_overlap = [
        frame
        for frame in args.focus_frame
        if any(run in frame.resolve().parents for run in validation_unique)
    ]
    if focus_validation_overlap:
        parser.error(
            "focus frames pertencem a runs de validacao: "
            + ", ".join(str(frame) for frame in focus_validation_overlap)
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    metric_root = args.source_dir / "metric_depth"
    sys.path.insert(0, str(metric_root))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    }
    model_config = model_configs[args.encoder]
    model = DepthAnythingV2(
        encoder=args.encoder,
        features=model_config["features"],
        out_channels=model_config["out_channels"],
        max_depth=80.0,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    device = torch.device("cuda")
    model.to(device)

    encoder_parameters, head_parameters = configure_encoder_training(
        model,
        freeze_encoder=args.freeze_encoder,
        unfreeze_encoder_blocks=args.unfreeze_encoder_blocks,
    )
    optimizer_groups = []
    if encoder_parameters:
        optimizer_groups.append(
            {"params": encoder_parameters, "lr": args.encoder_lr}
        )
    optimizer_groups.append({"params": head_parameters, "lr": args.head_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    effective_train_runs = [
        *args.train_run,
        *[run for run in args.focus_run for _ in range(args.focus_repeat)],
    ]
    effective_focus_frames = [
        frame
        for frame in args.focus_frame
        for _ in range(args.focus_frame_repeat)
    ]
    train_dataset = SpatialFrameDataset(
        effective_train_runs, training=True, extra_frames=effective_focus_frames
    )
    validation_dataset = SpatialFrameDataset(args.validation_run, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, target in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            prediction = functional.interpolate(
                prediction[:, None],
                target.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[:, 0]
            loss = masked_loss(
                prediction,
                target,
                edge_multiplier=args.edge_multiplier,
                unsafe_tail_weight=args.unsafe_tail_weight,
                unsafe_max_depth_m=args.unsafe_max_depth_m,
                unsafe_overestimate_weight=args.unsafe_overestimate_weight,
                unsafe_overestimate_tail_weight=args.unsafe_overestimate_tail_weight,
                unsafe_overestimate_tail_fraction=args.unsafe_overestimate_tail_fraction,
                edge_overestimate_weight=args.edge_overestimate_weight,
                unsafe_underestimate_weight=args.unsafe_underestimate_weight,
                unsafe_underestimate_tail_weight=args.unsafe_underestimate_tail_weight,
                unsafe_underestimate_tail_fraction=args.unsafe_underestimate_tail_fraction,
                edge_underestimate_weight=args.edge_underestimate_weight,
                edge_gradient_weight=args.edge_gradient_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, validation_loader, device)
        metrics.update({"epoch": epoch, "train_loss": float(np.mean(losses))})
        history.append(metrics)
        safety_excess = max(
            0.0,
            metrics["near_positive_error_p95_m"] - 1.25,
        )
        metrics["checkpoint_score"] = (
            metrics["mae_m"]
            + 2.0 * safety_excess
            + 2.0 * metrics["dangerous_far_rate_1m"]
            + 4.0 * metrics["dangerous_far_edge_rate_1m"]
            + metrics["dangerous_near_rate_1m"]
            + 2.0 * metrics["dangerous_near_edge_rate_1m"]
        )
        print(json.dumps(metrics), flush=True)
        if args.save_each_epoch:
            epoch_output = args.output.with_name(
                f"{args.output.stem}_epoch{epoch:02d}{args.output.suffix}"
            )
            torch.save(model.state_dict(), epoch_output)
        if metrics["checkpoint_score"] < best_score:
            best_score = metrics["checkpoint_score"]
            torch.save(model.state_dict(), args.output)
    args.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "train_runs": [str(run.resolve()) for run in args.train_run],
        "focus_runs": [str(run.resolve()) for run in args.focus_run],
        "focus_repeat": args.focus_repeat,
        "focus_frames": [
            str(frame.resolve()) for frame in args.focus_frame
        ],
        "focus_frame_repeat": args.focus_frame_repeat,
        "effective_focus_frame_samples": len(effective_focus_frames),
        "validation_runs": [str(run.resolve()) for run in args.validation_run],
        "effective_train_frames": len(train_dataset),
        "discarded_train_frames": len(train_dataset.invalid_frames),
        "discarded_train_frame_paths": [
            str(frame.resolve()) for frame in train_dataset.invalid_frames
        ],
        "validation_frames": len(validation_dataset),
        "discarded_validation_frames": len(validation_dataset.invalid_frames),
        "discarded_validation_frame_paths": [
            str(frame.resolve()) for frame in validation_dataset.invalid_frames
        ],
        "epochs": args.epochs,
        "encoder": args.encoder,
        "freeze_encoder": args.freeze_encoder,
        "unfreeze_encoder_blocks": args.unfreeze_encoder_blocks,
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in encoder_parameters
        ),
        "batch_size": args.batch_size,
        "head_lr": args.head_lr,
        "encoder_lr": args.encoder_lr,
        "edge_multiplier": args.edge_multiplier,
        "unsafe_tail_weight": args.unsafe_tail_weight,
        "unsafe_max_depth_m": args.unsafe_max_depth_m,
        "unsafe_overestimate_weight": args.unsafe_overestimate_weight,
        "unsafe_overestimate_tail_weight": args.unsafe_overestimate_tail_weight,
        "unsafe_overestimate_tail_fraction": args.unsafe_overestimate_tail_fraction,
        "edge_overestimate_weight": args.edge_overestimate_weight,
        "unsafe_underestimate_weight": args.unsafe_underestimate_weight,
        "unsafe_underestimate_tail_weight": args.unsafe_underestimate_tail_weight,
        "unsafe_underestimate_tail_fraction": args.unsafe_underestimate_tail_fraction,
        "edge_underestimate_weight": args.edge_underestimate_weight,
        "edge_gradient_weight": args.edge_gradient_weight,
        "save_each_epoch": args.save_each_epoch,
        "seed": args.seed,
    }
    args.output.with_suffix(".training.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
