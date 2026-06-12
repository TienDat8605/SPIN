from dataclasses import dataclass, field
from typing import Tuple, Dict


@dataclass
class SpectralConfig:
    spectral_mode: str = "fft"
    n_power_iters: int = 4
    patch_group_size: int = 6
    suppression_coeff: float = 0.3
    reinforcement_coeff: float = 0.15
    temperature: float = 0.1
    small_num_mask: float = 1e-4
    amp_max_start: float = 0.2
    amp_max_end: float = 0.05
    l2_smoothing: float = 1e-3
    held_out_chair_ratio: float = 0.1
    n_calib_examples: int = 64
    seed: int = 927

    def amp_max(self, layer_idx: int, n_layers: int) -> float:
        if n_layers <= 1:
            return self.amp_max_start
        t = layer_idx / (n_layers - 1)
        return self.amp_max_start + (self.amp_max_end - self.amp_max_start) * t

    def resolve_thresholds(
        self, thresholds: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Tuple[float, float]]:
        return {l: thresholds.get(l, (0.3, 0.6)) for l in range(self.max_layer(thresholds) + 1)}

    @staticmethod
    def max_layer(thresholds: Dict[int, Tuple[float, float]]) -> int:
        if not thresholds:
            return 0
        return max(thresholds.keys())
