import torch
import torch.nn.functional as F

def _dilate(mask, radius):
    return F.max_pool2d(
        mask.float().unsqueeze(1), 2 * radius + 1, stride=1, padding=radius
    )[:, 0]


def _erode(mask, radius):
    return 1.0 - _dilate(1.0 - mask.float(), radius)


def _source_regions(pseudo_mask, core_radius, context_radius):
    foreground = pseudo_mask.float()
    core = _erode(foreground, core_radius)
    boundary = (foreground - core).clamp(0.0, 1.0)
    context = (_dilate(foreground, context_radius) - foreground).clamp(0.0, 1.0)
    return core, boundary, context


@torch.no_grad()
def morphology_texture_decoupled_cross_source_interaction(
    source,
    pseudo_mask,
    hard_trust,
    reference,
    reference_mask,
    strength=0.6,
    core_radius=3,
    context_radius=4,
    context_weight=0.15,
    probability=0.75,
    transition_low=0.02,
    transition_high=0.95,
):
    source = source.float()
    reference = reference.float()
    core, boundary, context = _source_regions(
        pseudo_mask, core_radius, context_radius
    )
    reliable_core = core * hard_trust

    ref_mask = reference_mask.float().unsqueeze(1)
    src_weight = reliable_core.unsqueeze(1)
    ref_count = ref_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    src_count = src_weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    ref_mean = (reference * ref_mask).sum(dim=(2, 3), keepdim=True) / ref_count
    src_mean = (source * src_weight).sum(dim=(2, 3), keepdim=True) / src_count
    ref_var = ((reference - ref_mean).square() * ref_mask).sum(
        dim=(2, 3), keepdim=True
    ) / ref_count
    src_var = ((source - src_mean).square() * src_weight).sum(
        dim=(2, 3), keepdim=True
    ) / src_count
    ref_std = ref_var.add(1e-5).sqrt()
    src_std = src_var.add(1e-5).sqrt()

    high_frequency_residual = reference - F.avg_pool2d(
        reference, kernel_size=5, stride=1, padding=2
    )
    texture_image = (
        (source - src_mean) / src_std * ref_std
        + ref_mean
        + 0.35 * high_frequency_residual
    ).clamp(-1.0, 1.0)

    apply = (
        torch.rand(source.shape[0], 1, 1, 1, device=source.device) < probability
    ).float()
    alpha = strength * src_weight * apply
    alpha += (
        context_weight
        * strength
        * F.avg_pool2d(context.unsqueeze(1), kernel_size=5, stride=1, padding=2)
    )
    alpha = alpha.clamp(0.0, 1.0) * (1.0 - boundary.unsqueeze(1))
    mixed = source * (1.0 - alpha) + texture_image * alpha
    transition = ((alpha > transition_low) & (alpha < transition_high)).float()[:, 0]
    return mixed, transition, reliable_core, apply[:, 0, 0, 0]
