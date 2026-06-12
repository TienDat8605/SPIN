import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SpectralFeatures:
    lambda_max: torch.Tensor
    spread: torch.Tensor
    last_token_focus: torch.Tensor


def _take_last_row(A: torch.Tensor) -> torch.Tensor:
    if A.dim() == 4:
        return A[:, :, -1, :]
    if A.dim() == 3:
        return A[:, -1, :]
    if A.dim() == 2:
        return A[-1:, :]
    raise ValueError(f"Unexpected attention tensor shape: {tuple(A.shape)}")


def _spread_and_focus(A_last: torch.Tensor) -> tuple:
    spread = A_last.std(dim=-1)
    focus = A_last.sum(dim=-1)
    return spread, focus


def fft_hf_ratio(A: torch.Tensor) -> SpectralFeatures:
    A_last = _take_last_row(A).to(torch.float32)
    spec = torch.fft.rfft(A_last, dim=-1)
    mag2 = spec.abs().pow(2)
    n_freq = mag2.shape[-1]
    cutoff = n_freq // 2
    if cutoff <= 0:
        hf_ratio = torch.zeros(mag2.shape[:-1], dtype=mag2.dtype, device=mag2.device)
    else:
        high = mag2[..., cutoff:].sum(dim=-1)
        total = mag2.sum(dim=-1) + 1e-8
        hf_ratio = high / total
    spread, focus = _spread_and_focus(A_last)
    return SpectralFeatures(
        lambda_max=hf_ratio.to(A_last.dtype),
        spread=spread,
        last_token_focus=focus,
    )


def power_iter_lambda_max(A: torch.Tensor, k: int = 4) -> SpectralFeatures:
    A_last = _take_last_row(A).to(torch.float32)
    n = A_last.shape[-1]
    squeeze = A_last.dim() == 2
    if squeeze:
        A_last = A_last.unsqueeze(0)

    B, H = A_last.shape[:2]

    idx = torch.arange(n, device=A_last.device, dtype=torch.long)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)) % n
    C = A_last[:, :, diff]

    eye = torch.eye(n, device=A_last.device, dtype=A_last.dtype)
    L = eye - C

    L_flat = L.reshape(B * H, n, n)
    b = torch.randn(B * H, n, device=L.device, dtype=L.dtype)
    for _ in range(k):
        b = torch.bmm(L_flat, b.unsqueeze(-1)).squeeze(-1)
        norm = b.norm(dim=-1, keepdim=True) + 1e-8
        b = b / norm

    Lb = torch.bmm(L_flat, b.unsqueeze(-1))
    rayleigh = (b.unsqueeze(1) * Lb.squeeze(-1).unsqueeze(1)).sum(dim=(-2, -1))
    rayleigh = rayleigh.view(B, H)
    if squeeze:
        rayleigh = rayleigh.squeeze(0)

    spread, focus = _spread_and_focus(A_last)
    return SpectralFeatures(
        lambda_max=rayleigh.to(A_last.dtype),
        spread=spread,
        last_token_focus=focus,
    )


def block_coarsen_spectrum(A: torch.Tensor, G: int) -> SpectralFeatures:
    A_last = _take_last_row(A).to(torch.float32)
    if A_last.dim() == 2:
        n = A_last.shape[-1]
        A_last_b = A_last.unsqueeze(0)
    else:
        n = A_last.shape[-1]
        A_last_b = A_last

    assert n == G * G, f"n_img={n} must equal G^2={G * G}"

    reshaped = A_last_b.view(*A_last_b.shape[:-1], G, n // G).mean(dim=-1)
    reshaped = reshaped - reshaped.mean(dim=-1, keepdim=True)

    spec = torch.fft.rfft(reshaped, dim=-1)
    mag2 = spec.abs().pow(2)
    n_freq = mag2.shape[-1]
    cutoff = max(1, n_freq // 2)
    high = mag2[..., cutoff:].sum(dim=-1)
    total = mag2.sum(dim=-1) + 1e-8
    hf_2d = (high / total)

    if A_last.dim() == 2:
        hf_2d = hf_2d.squeeze(0)

    spread, focus = _spread_and_focus(A_last)
    return SpectralFeatures(
        lambda_max=hf_2d.to(A_last.dtype),
        spread=spread,
        last_token_focus=focus,
    )


def compute_spectral_features(
    A: torch.Tensor,
    mode: str = "fft",
    n_power_iters: int = 4,
    patch_group_size: int = 6,
) -> SpectralFeatures:
    if mode == "fft":
        return fft_hf_ratio(A)
    if mode == "power":
        return power_iter_lambda_max(A, k=n_power_iters)
    if mode == "block":
        return block_coarsen_spectrum(A, G=patch_group_size)
    raise ValueError(f"Unknown spectral_mode: {mode!r}. Expected one of: fft, power, block.")


if __name__ == "__main__":
    import unittest

    class TestSpectral(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(0)
            self.B, self.H, self.n = 2, 4, 576

            sharp = torch.zeros(self.B, self.H, self.n)
            center = self.n // 2
            sharp[..., center - 5 : center + 5] = 1.0
            sharp = sharp / sharp.sum(-1, keepdim=True)
            self.A_sharp = sharp.unsqueeze(2)

            diffuse = torch.ones(self.B, self.H, self.n) / self.n
            self.A_diffuse = diffuse.unsqueeze(2)

        def test_fft_sharp_higher(self):
            f_s = fft_hf_ratio(self.A_sharp)
            f_d = fft_hf_ratio(self.A_diffuse)
            self.assertGreater(
                f_s.lambda_max.mean().item(), f_d.lambda_max.mean().item()
            )

        def test_power_sharp_higher(self):
            f_s = power_iter_lambda_max(self.A_sharp)
            f_d = power_iter_lambda_max(self.A_diffuse)
            self.assertGreater(
                f_s.lambda_max.mean().item(), f_d.lambda_max.mean().item()
            )

        def test_block_sharp_higher(self):
            f_s = block_coarsen_spectrum(self.A_sharp, G=24)
            f_d = block_coarsen_spectrum(self.A_diffuse, G=24)
            self.assertGreater(
                f_s.lambda_max.mean().item(), f_d.lambda_max.mean().item()
            )

        def test_dispatcher(self):
            for mode in ("fft", "power", "block"):
                feats = compute_spectral_features(
                    self.A_sharp, mode=mode, patch_group_size=24
                )
                self.assertEqual(feats.lambda_max.shape, (self.B, self.H))

        def test_block_shape_mismatch(self):
            with self.assertRaises(AssertionError):
                block_coarsen_spectrum(self.A_sharp, G=8)

    unittest.main()
