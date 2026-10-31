"""Evaluate a CrossTrust checkpoint on validation or test splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from dataloaders.pathology_dataset import PathologyNucleiDataset, discover_pathology_pairs
from networks.nnunet2d import NNUNet2D


def binary_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    pred_count = prediction.sum()
    target_count = target.sum()
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0, 1.0, 0.0, 0.0
    dice = (2.0 * intersection) / max(pred_count + target_count, 1)
    jaccard = intersection / union

    pred_surface = prediction ^ ndimage.binary_erosion(prediction)
    target_surface = target ^ ndimage.binary_erosion(target)
    if not pred_surface.any() and not target_surface.any():
        return dice, jaccard, 0.0, 0.0
    if not pred_surface.any() or not target_surface.any():
        diagonal = float(np.hypot(*target.shape))
        return dice, jaccard, diagonal, diagonal

    distance_to_target = ndimage.distance_transform_edt(~target_surface)[pred_surface]
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface)[target_surface]
    symmetric = np.concatenate((distance_to_target, distance_to_prediction))
    assd = 0.5 * (distance_to_target.mean() + distance_to_prediction.mean())
    hd95 = np.percentile(symmetric, 95)
    return dice, jaccard, float(assd), float(hd95)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CrossTrust checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_options = config.get("model", {})
    model = NNUNet2D(
        n_channels=3,
        n_classes=2,
        base_channels=int(model_options.get("base_channels", 32)),
        max_channels=int(model_options.get("max_channels", 320)),
        depth=int(model_options.get("depth", 5)),
        norm=model_options.get("normalization", "instance"),
        dropout=float(model_options.get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("teacher", checkpoint.get("student", checkpoint)))
    model.eval()

    results = {}
    metric_names = ("DC", "JC", "ASSD", "HD95")
    with torch.no_grad():
        for dataset_config in config["datasets"]:
            pairs = discover_pathology_pairs(dataset_config, args.split)
            if not pairs:
                raise FileNotFoundError(
                    f"No {args.split} image/mask pairs found for {dataset_config['name']}."
                )
            loader = DataLoader(
                PathologyNucleiDataset(pairs, training=False),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            per_case = []
            for batch in loader:
                logits = model(batch["image"].to(device))
                predictions = logits.argmax(dim=1).cpu().numpy()
                targets = batch["label"].numpy()
                per_case.extend(
                    binary_metrics(pred, target)
                    for pred, target in zip(predictions, targets)
                )
            values = np.asarray(per_case, dtype=np.float64)
            results[dataset_config["name"]] = {
                metric: {"mean": float(values[:, i].mean()), "std": float(values[:, i].std(ddof=1)) if len(values) > 1 else 0.0}
                for i, metric in enumerate(metric_names)
            }

    serialized = json.dumps(results, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
