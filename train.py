"""Train CrossTrust with the three paper modules and the stated objective."""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.pathology_dataset import (
    PathologyNucleiDataset,
    discover_pathology_pairs,
    group_train_validation_split,
    split_labeled_unlabeled,
)
from modules.bpsp import boundary_preserved_spectral_perturbation
from modules.ms_dctl import MultiSourceDynamicCenterTrustLearning
from modules.mtd_csi import morphology_texture_decoupled_cross_source_interaction
from networks.nnunet2d import NNUNet2D


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrossTrust on multi-domain nuclei data.")
    parser.add_argument("--config", default="configs/crosstrust_four_domains.example.json")
    parser.add_argument("--output", default="runs/crosstrust")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-ratio", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--unsupervised-warmup", type=int, default=20)
    parser.add_argument("--center-momentum", type=float, default=0.99)
    parser.add_argument("--scale-momentum", type=float, default=0.99)
    parser.add_argument("--anchor-fraction", type=float, default=0.20)
    parser.add_argument("--trust-threshold", type=float, default=0.40)
    parser.add_argument("--soft-trust-floor", type=float, default=0.05)
    parser.add_argument("--boundary-risk-strength", type=float, default=0.75)

    parser.add_argument("--bpsp-probability", type=float, default=0.5)
    parser.add_argument("--bpsp-start-epoch", type=int, default=5)
    parser.add_argument("--real-band-high", type=float, default=0.16)
    parser.add_argument("--imag-band-low", type=float, default=0.06)
    parser.add_argument("--imag-band-high", type=float, default=0.28)
    parser.add_argument("--real-strength", type=float, default=0.35)
    parser.add_argument("--imag-strength", type=float, default=0.10)
    parser.add_argument("--core-radius", type=int, default=3)

    parser.add_argument("--mtd-csi-probability", type=float, default=0.75)
    parser.add_argument("--mtd-csi-start-epoch", type=int, default=50)
    parser.add_argument("--mtd-csi-strength", type=float, default=0.6)
    parser.add_argument("--mtd-csi-weight", type=float, default=0.5)
    parser.add_argument("--context-radius", type=int, default=4)
    return parser.parse_args()


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_data(config, args):
    domains = []
    for domain_id, item in enumerate(config["datasets"]):
        train_pairs = discover_pathology_pairs(item, "train")
        validation_pairs = discover_pathology_pairs(item, "val")
        if not validation_pairs:
            train_pairs, validation_pairs = group_train_validation_split(
                train_pairs,
                float(item.get("validation_fraction", 0.2)),
                args.seed + domain_id,
            )
        labeled, unlabeled = split_labeled_unlabeled(
            train_pairs, args.label_ratio, args.seed + 100 + domain_id
        )
        aug = config.get("augmentation", {})
        labeled_set = PathologyNucleiDataset(labeled, True, aug)
        validation_set = PathologyNucleiDataset(validation_pairs, False)
        unlabeled_set = PathologyNucleiDataset(
            unlabeled, True, aug, return_strong_view=True
        )
        domains.append(
            {
                "name": item["name"],
                "labeled": CyclingLoader(make_loader(labeled_set, args, True)),
                "unlabeled": CyclingLoader(make_loader(unlabeled_set, args, True))
                if unlabeled
                else None,
                "validation": make_loader(validation_set, args, False),
                "train_count": len(train_pairs),
            }
        )
    return domains


def segmentation_loss(logits, target, pixel_weight=None):
    if pixel_weight is None:
        pixel_weight = torch.ones_like(target, dtype=logits.dtype)
    pixel_weight = pixel_weight.to(logits.dtype)
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    ce = (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    foreground_probability = torch.softmax(logits, dim=1)[:, 1]
    foreground_target = target.to(logits.dtype)
    intersection = (foreground_probability * foreground_target * pixel_weight).sum()
    denominator = (
        (foreground_probability.square() + foreground_target.square()) * pixel_weight
    ).sum()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return ce + dice


def soft_consistency_loss(student_logits, teacher_probability, trust_weight):
    pixel_kl = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_probability.detach(),
        reduction="none",
    ).sum(dim=1)
    return (pixel_kl * trust_weight).sum() / trust_weight.sum().clamp_min(1.0)


def update_ema(student, teacher, decay, step):
    decay = min(decay, 1.0 - 1.0 / (step + 1.0))
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(
            teacher.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)


def match_batch(tensor, batch_size):
    if tensor.shape[0] >= batch_size:
        return tensor[:batch_size]
    repeats = math.ceil(batch_size / tensor.shape[0])
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:batch_size]


@torch.no_grad()
def evaluate(model, domains, device):
    model.eval()
    scores = {}
    for domain in domains:
        intersection = prediction_count = target_count = 0.0
        for batch in domain["validation"]:
            image = batch["image"].to(device)
            target = batch["label"].to(device).bool()
            prediction = model(image).argmax(dim=1).bool()
            intersection += (prediction & target).sum().item()
            prediction_count += prediction.sum().item()
            target_count += target.sum().item()
        scores[domain["name"]] = (2.0 * intersection + 1e-6) / (
            prediction_count + target_count + 1e-6
        )
    return scores


def train(args):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    domains = build_data(config, args)
    if len(domains) < 2:
        raise ValueError("Cross-domain training requires at least two datasets.")

    model_options = config.get("model", {})
    student = NNUNet2D(
        n_channels=3,
        n_classes=2,
        base_channels=int(model_options.get("base_channels", 32)),
        max_channels=int(model_options.get("max_channels", 320)),
        depth=int(model_options.get("depth", 5)),
        norm=model_options.get("normalization", "instance"),
        dropout=float(model_options.get("dropout", 0.0)),
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    teacher.requires_grad_(False)

    ms_dctl = MultiSourceDynamicCenterTrustLearning(
        len(domains),
        device,
        center_momentum=args.center_momentum,
        scale_momentum=args.scale_momentum,
        anchor_fraction=args.anchor_fraction,
        hard_threshold=args.trust_threshold,
        soft_floor=args.soft_trust_floor,
        boundary_risk_strength=args.boundary_risk_strength,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    total_samples = sum(domain["train_count"] for domain in domains)
    steps_per_epoch = math.ceil(total_samples / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    global_step = 0
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            domain_id = random.randrange(len(domains))
            domain = domains[domain_id]
            labeled = domain["labeled"].next()
            x_l = labeled["image"].to(device)
            y_l = labeled["label"].to(device)
            loss_s = segmentation_loss(student(x_l), y_l)

            loss_u = loss_s.new_zeros(())
            loss_mt = loss_s.new_zeros(())
            if domain["unlabeled"] is not None:
                unlabeled = domain["unlabeled"].next()
                u_clean = unlabeled["image"].to(device)
                u_strong = unlabeled["strong_image"].to(device)
                with torch.no_grad():
                    p_clean = torch.softmax(teacher(u_clean), dim=1)
                    preliminary_label = p_clean.argmax(dim=1)

                donor_id = random.choice([i for i in range(len(domains)) if i != domain_id])
                donor_batch = domains[donor_id]["labeled"].next()
                reference = match_batch(donor_batch["image"].to(device), u_strong.shape[0])
                reference_mask = match_batch(donor_batch["label"].to(device), u_strong.shape[0])
                bpsp_view = u_strong
                if epoch >= args.bpsp_start_epoch:
                    bpsp_view = boundary_preserved_spectral_perturbation(
                        u_strong,
                        preliminary_label,
                        reference,
                        probability=args.bpsp_probability,
                        real_band_high=args.real_band_high,
                        imag_band_low=args.imag_band_low,
                        imag_band_high=args.imag_band_high,
                        real_strength=args.real_strength,
                        imag_strength=args.imag_strength,
                        core_radius=args.core_radius,
                    )

                with torch.no_grad():
                    p_perturbed = torch.softmax(teacher(bpsp_view), dim=1)
                    p_teacher, pseudo_label, hard_trust, soft_trust = ms_dctl(
                        domain_id, p_clean, p_perturbed
                    )

                student_logits = student(bpsp_view)
                loss_pl = segmentation_loss(student_logits, pseudo_label, hard_trust)
                loss_con = soft_consistency_loss(student_logits, p_teacher, soft_trust)
                loss_u = loss_pl + loss_con

                if epoch >= args.mtd_csi_start_epoch:
                    mixed, transition, _, applied = (
                        morphology_texture_decoupled_cross_source_interaction(
                            bpsp_view,
                            pseudo_label,
                            hard_trust,
                            reference,
                            reference_mask,
                            strength=args.mtd_csi_strength,
                            core_radius=args.core_radius,
                            context_radius=args.context_radius,
                            probability=args.mtd_csi_probability,
                        )
                    )
                    valid_mixed = hard_trust * (1.0 - transition)
                    valid_mixed *= applied[:, None, None]
                    loss_mt = segmentation_loss(student(mixed), pseudo_label, valid_mixed)

            ramp_length = max(1, int(0.2 * args.epochs))
            ramp = min(1.0, max(0.0, (epoch - args.unsupervised_warmup) / ramp_length))
            loss = loss_s + ramp * (loss_u + args.mtd_csi_weight * loss_mt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(student, teacher, args.ema_decay, global_step)
            learning_rate = args.learning_rate * (1.0 - global_step / total_steps) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            global_step += 1
            epoch_loss += loss.item()

        scores = evaluate(teacher, domains, device)
        mean_dice = float(np.mean(list(scores.values())))
        state = {
            "epoch": epoch,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ms_dctl_centers": ms_dctl.centers,
            "ms_dctl_scales": ms_dctl.scales,
            "args": vars(args),
            "validation_dice": scores,
        }
        torch.save(state, output / "last.pth")
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(state, output / "best.pth")
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={epoch_loss / steps_per_epoch:.4f} "
            f"val_dice={mean_dice:.4f} domains={scores}"
        )


if __name__ == "__main__":
    train(parse_args())
