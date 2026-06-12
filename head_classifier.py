import json
from typing import Dict, Tuple

import torch


def fit_thresholds(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_layers: int,
    l2_smoothing: float = 1e-3,
    default_weak: float = 0.3,
    default_strong: float = 0.6,
) -> Dict[int, Tuple[float, float]]:
    features = features.detach().to(torch.float32)
    labels = labels.detach().to(torch.float32)
    thresholds: Dict[int, Tuple[float, float]] = {}
    for l in range(n_layers):
        if l >= features.shape[1]:
            thresholds[l] = (default_weak, default_strong)
            continue
        feat_l = features[:, l, :]
        feat_flat = feat_l.reshape(-1)
        lab_flat = labels.unsqueeze(1).expand_as(feat_l).reshape(-1)

        pos = feat_flat[lab_flat > 0.5]
        neg = feat_flat[lab_flat <= 0.5]

        if pos.numel() == 0 or neg.numel() == 0:
            thresholds[l] = (default_weak, default_strong)
            continue

        mu_pos = pos.mean()
        mu_neg = neg.mean()
        std_pos = pos.std() + 1e-6
        std_neg = neg.std() + 1e-6

        tau_weak = (mu_pos * std_neg + mu_neg * std_pos) / (std_pos + std_neg)
        tau_strong = mu_pos + 0.7 * (mu_neg - mu_pos)

        tau_weak = tau_weak - l2_smoothing * (tau_weak - 0.5)
        tau_strong = tau_strong - l2_smoothing * (tau_strong - 0.5)

        thresholds[l] = (
            float(tau_weak.clamp(0.0, 1.0).item()),
            float(tau_strong.clamp(0.0, 1.0).item()),
        )
    return thresholds


def save_thresholds(
    thresholds: Dict[int, Tuple[float, float]], path: str
) -> None:
    serial = {str(k): [float(v[0]), float(v[1])] for k, v in thresholds.items()}
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)


def load_thresholds(path: str) -> Dict[int, Tuple[float, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): (float(v[0]), float(v[1])) for k, v in data.items()}


def threshold_to_tau_tensor(
    thresholds: Dict[int, Tuple[float, float]],
    n_layers: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tau_weak = torch.zeros(n_layers, device=device, dtype=dtype)
    tau_strong = torch.zeros(n_layers, device=device, dtype=dtype)
    for l in range(n_layers):
        w, s = thresholds.get(l, (0.3, 0.6))
        tau_weak[l] = w
        tau_strong[l] = s
    return tau_weak, tau_strong


if __name__ == "__main__":
    import unittest

    class TestClassifier(unittest.TestCase):
        def test_fit_thresholds_separable(self):
            torch.manual_seed(0)
            n_ex, n_layers, n_heads = 100, 4, 8
            feat = torch.zeros(n_ex, n_layers, n_heads)
            labels = torch.zeros(n_ex)
            labels[50:] = 1.0
            for l in range(n_layers):
                feat[:50, l, :] = 0.8 + 0.05 * torch.randn(50, n_heads)
                feat[50:, l, :] = 0.2 + 0.05 * torch.randn(50, n_heads)
            th = fit_thresholds(feat, labels, n_layers, l2_smoothing=1e-3)
            for l in range(n_layers):
                w, s = th[l]
                self.assertGreater(w, 0.4)
                self.assertLess(w, 0.6)
                self.assertGreater(s, w)

        def test_default_for_missing_layer(self):
            feat = torch.zeros(10, 2, 4)
            labels = torch.zeros(10)
            th = fit_thresholds(feat, labels, n_layers=5)
            for l in range(2, 5):
                self.assertEqual(th[l], (0.3, 0.6))

        def test_save_load(self):
            tmp = "/tmp/_thresholds_test.json"
            th = {0: (0.3, 0.6), 1: (0.4, 0.7)}
            save_thresholds(th, tmp)
            loaded = load_thresholds(tmp)
            self.assertEqual(loaded, th)

    unittest.main()
