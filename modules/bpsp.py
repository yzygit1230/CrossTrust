import torch
import torch.nn.functional as F


def _dilate(mask, radius):
    return F.max_pool2d(
        mask.float().unsqueeze(1), 2 * radius + 1, stride=1, padding=radius
    )[:, 0]


def _regions(mask, core_radius):
    foreground = mask.float()
    core = 1.0 - _dilate(1.0 - foreground, core_radius)
    boundary = (foreground - core).clamp(0.0, 1.0)
    background = 1.0 - foreground
    return core, boundary, background


@torch.no_grad()
def boundary_preserved_spectral_perturbation(
    source,
    pseudo_mask,
    reference,
    probability=0.5,
    real_band_high=0.16,
    imag_band_low=0.06,
    imag_band_high=0.28,
    real_strength=0.35,
    imag_strength=0.10,
    core_radius=3,
    mask_sharpness=80.0,
):

    batch, channels, height, width = source.shape
    apply = torch.rand(batch, 1, 1, 1, device=source.device) < probability
    if not apply.any():
        return source

    src = (source.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    ref = (reference.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    src_fft = torch.fft.fftshift(torch.fft.fft2(src), dim=(-2, -1))
    ref_fft = torch.fft.fftshift(torch.fft.fft2(ref), dim=(-2, -1))

    fy = torch.linspace(-1.0, 1.0, height, device=source.device)
    fx = torch.linspace(-1.0, 1.0, width, device=source.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    real_mask = torch.sigmoid(mask_sharpness * (real_band_high - radius))
    imag_mask = torch.sigmoid(mask_sharpness * (radius - imag_band_low))
    imag_mask *= torch.sigmoid(mask_sharpness * (imag_band_high - radius))
    real_alpha = torch.rand(batch, 1, 1, 1, device=source.device) * real_strength
    imag_alpha = torch.rand(batch, 1, 1, 1, device=source.device) * imag_strength
    real_ratio = real_alpha * real_mask.view(1, 1, height, width)
    imag_ratio = imag_alpha * imag_mask.view(1, 1, height, width)

    real_part = torch.lerp(src_fft.real, ref_fft.real, real_ratio)
    imag_part = torch.lerp(src_fft.imag, ref_fft.imag, imag_ratio)
    z_appearance = torch.fft.ifft2(
        torch.fft.ifftshift(torch.complex(real_part, src_fft.imag), dim=(-2, -1))
    ).real.clamp(0.0, 1.0)
    z_texture = torch.fft.ifft2(
        torch.fft.ifftshift(torch.complex(real_part, imag_part), dim=(-2, -1))
    ).real.clamp(0.0, 1.0)
    z_appearance = (z_appearance - 0.5) / 0.5
    z_texture = (z_texture - 0.5) / 0.5

    core, boundary, background = _regions(pseudo_mask, core_radius)
    candidate = (
        source * boundary.unsqueeze(1)
        + z_texture * core.unsqueeze(1)
        + z_appearance * background.unsqueeze(1)
    )
    return torch.where(apply, candidate, source)
