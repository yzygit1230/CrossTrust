import math
import torch
import torch.nn.functional as F


def normalized_entropy(probability):
    entropy = -(probability * probability.clamp_min(1e-6).log()).sum(dim=1)
    return entropy / math.log(probability.shape[1])


def prediction_stability(clean_probability, perturbed_probability):
    mean_probability = 0.5 * (clean_probability + perturbed_probability)
    clean_kl = F.kl_div(
        mean_probability.clamp_min(1e-6).log(), clean_probability, reduction="none"
    ).sum(dim=1)
    perturbed_kl = F.kl_div(
        mean_probability.clamp_min(1e-6).log(), perturbed_probability, reduction="none"
    ).sum(dim=1)
    return torch.exp(-0.5 * (clean_kl + perturbed_kl))


class MultiSourceDynamicCenterTrustLearning:
    def __init__(
        self,
        num_domains,
        device,
        center_momentum=0.99,
        scale_momentum=0.99,
        anchor_fraction=0.20,
        hard_threshold=0.40,
        soft_floor=0.05,
        boundary_risk_strength=0.75,
    ):
        initial_center = torch.tensor([0.8, 0.5, 0.8], device=device)
        initial_scale = torch.tensor([0.08, 0.12, 0.08], device=device)
        self.centers = initial_center.view(1, 1, 3).repeat(num_domains, 2, 1)
        self.scales = initial_scale.view(1, 1, 3).repeat(num_domains, 2, 1)
        self.center_momentum = center_momentum
        self.scale_momentum = scale_momentum
        self.anchor_fraction = anchor_fraction
        self.hard_threshold = hard_threshold
        self.soft_floor = soft_floor
        self.boundary_risk_strength = boundary_risk_strength

    @torch.no_grad()
    def _update_centers(self, domain_id, evidence, pseudo_label):
        for class_id in range(2):
            class_evidence = evidence[pseudo_label == class_id]
            if class_evidence.numel() == 0:
                continue
            quality = class_evidence.prod(dim=-1)
            count = max(1, math.ceil(class_evidence.shape[0] * self.anchor_fraction))
            anchors = class_evidence[quality.topk(min(count, quality.numel())).indices]
            center = anchors.mean(dim=0)
            scale = (anchors - center).abs().mean(dim=0).clamp_min(0.02)
            self.centers[domain_id, class_id].lerp_(center, 1.0 - self.center_momentum)
            self.scales[domain_id, class_id].lerp_(scale, 1.0 - self.scale_momentum)
        self.centers[domain_id].clamp_(0.05, 0.98)
        self.scales[domain_id].clamp_(0.02, 0.35)

    @staticmethod
    def _boundary_risk(teacher_probability, clean_probability, perturbed_probability):
        foreground = teacher_probability[:, 1]
        dx = F.pad((foreground[:, :, 1:] - foreground[:, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((foreground[:, 1:, :] - foreground[:, :-1, :]).abs(), (0, 0, 0, 1))
        gradient_risk = (dx + dy).clamp(0.0, 1.0)
        disagreement = (clean_probability[:, 1] - perturbed_probability[:, 1]).abs()
        return (0.6 * gradient_risk + 0.4 * disagreement).clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(self, domain_id, clean_probability, perturbed_probability):
        teacher_probability = 0.5 * (clean_probability + perturbed_probability)
        confidence, pseudo_label = teacher_probability.max(dim=1)
        certainty = 1.0 - normalized_entropy(teacher_probability)
        stability = prediction_stability(clean_probability, perturbed_probability)
        evidence = torch.stack((confidence, certainty, stability), dim=-1)

        self._update_centers(domain_id, evidence, pseudo_label)
        center = self.centers[domain_id][pseudo_label]
        scale = self.scales[domain_id][pseudo_label]
        cue_trust = torch.sigmoid((evidence - center) / scale.clamp_min(1e-4))
        multi_source_trust = cue_trust.prod(dim=-1).pow(1.0 / 3.0)
        risk = self._boundary_risk(
            teacher_probability, clean_probability, perturbed_probability
        )
        trust = (multi_source_trust * torch.exp(-self.boundary_risk_strength * risk))
        trust = trust.clamp(0.0, 1.0)
        hard_weight = trust * (trust >= self.hard_threshold)
        soft_weight = ((trust - self.soft_floor) / (1.0 - self.soft_floor)).clamp(0.0, 1.0)
        return teacher_probability, pseudo_label, hard_weight, soft_weight
