"""
Calibration driver for the Counteractive Balance (CB) spectral head modulation.

Workflow:
  1. Load model + small calibration set.
  2. Apply `llama_modify_cb(..., capture=True)` so each gated layer stores
     the last-token attention-to-image in `self._cb_last_attn`.
  3. Run a forward pass for each calibration example and record features.
  4. Use the example's binary label (1 = likely hallucinated, 0 = faithful)
     as a per-example supervision signal.
  5. Call `fit_thresholds` to produce per-layer (tau_weak, tau_strong).

The driver is invoked at the top of every eval run per the locked plan.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from head_classifier import fit_thresholds
from spectral import compute_spectral_features
from spectral_config import SpectralConfig


def _resolve_n_layers(model, start_layer: int, end_layer: int) -> int:
    try:
        return len(model.model.layers)
    except AttributeError:
        return end_layer


def _clear_capture_buffers(model, start_layer: int, end_layer: int) -> None:
    for i in range(start_layer, end_layer):
        attn = model.model.layers[i].self_attn
        if hasattr(attn, "_cb_last_attn"):
            attn._cb_last_attn = None


def _collect_capture_buffers(
    model, start_layer: int, end_layer: int
) -> List[torch.Tensor]:
    out = []
    for i in range(start_layer, end_layer):
        attn = model.model.layers[i].self_attn
        if hasattr(attn, "_cb_last_attn") and attn._cb_last_attn is not None:
            out.append(attn._cb_last_attn.detach().to(torch.float32).cpu())
        else:
            out.append(None)
    return out


def calibrate_spectral(
    model,
    dataset,
    template: str,
    prepare_inputs_fn,
    spectral_cfg: SpectralConfig,
    start_layer: int,
    end_layer: int,
    img_start_idx: int,
    img_end_idx: int,
    batch_size: int = 1,
    device: str = "cuda",
) -> Dict[int, Tuple[float, float]]:
    """
    Run a forward pass over the calibration set, accumulate per-head
    spectral features, then fit per-layer thresholds.

    `prepare_inputs_fn(template, query, image)` must return the same
    (questions, img_start_idx, img_end_idx, kwargs) tuple used by the
    eval scripts.
    """
    from attentionSPIN import llama_modify_cb

    llama_modify_cb(
        model,
        start_layer=start_layer,
        end_layer=end_layer,
        img_start_idx=img_start_idx,
        img_end_idx=img_end_idx,
        spectral_cfg=spectral_cfg,
        capture=True,
    )

    n_layers = _resolve_n_layers(model, end_layer, end_layer)
    gated_n = end_layer - start_layer

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    feature_acc: Dict[int, List[torch.Tensor]] = {l: [] for l in range(gated_n)}
    label_list: List[float] = []

    n_done = 0
    t0 = time.time()
    for data in loader:
        if n_done >= spectral_cfg.n_calib_examples:
            break
        image = data["image"]
        if "query" in data:
            query = data["query"]
        else:
            query = ["Please describe the image in detail."] * image.shape[0]
        if isinstance(query, torch.Tensor):
            query = query.tolist()
        if not isinstance(query, list):
            query = [query]
        labels = data.get("label", [0] * image.shape[0])
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        if not isinstance(labels, list):
            labels = [labels]

        try:
            questions, img_start, img_end, kwargs = prepare_inputs_fn(template, query, image)
        except Exception as e:
            print(f"[calibrate_spectral] skip example: {e}")
            continue

        _clear_capture_buffers(model, start_layer, end_layer)
        with torch.inference_mode():
            try:
                model.generate(
                    do_sample=False,
                    max_new_tokens=1,
                    use_cache=True,
                    num_beams=1,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs,
                )
            except Exception as e:
                print(f"[calibrate_spectral] forward failed: {e}")
                continue

        bufs = _collect_capture_buffers(model, start_layer, end_layer)
        for li, buf in enumerate(bufs):
            if buf is None:
                continue
            attn_last = buf[..., img_start:img_end]  # [B, H, n_img]
            feats = compute_spectral_features(
                attn_last,
                mode=spectral_cfg.spectral_mode,
                n_power_iters=spectral_cfg.n_power_iters,
                patch_group_size=spectral_cfg.patch_group_size,
            )
            feature_acc[li].append(feats.lambda_max)

        for lab in labels:
            label_list.append(float(lab) if isinstance(lab, (int, float)) else float(lab[0]) if hasattr(lab, "__getitem__") else 0.0)
        n_done += image.shape[0]

    print(f"[calibrate_spectral] collected {n_done} examples in {time.time() - t0:.1f}s")

    if n_done == 0:
        return {l: (0.3, 0.6) for l in range(gated_n)}

    labels_t = torch.tensor(label_list, dtype=torch.float32)
    n_heads = None
    for li in range(gated_n):
        if feature_acc[li]:
            n_heads = feature_acc[li][0].shape[-1]
            break
    if n_heads is None:
        return {l: (0.3, 0.6) for l in range(gated_n)}

    feat_tensor = torch.zeros(n_done, gated_n, n_heads)
    idx_per_layer = [0] * gated_n
    for li in range(gated_n):
        for ex_idx, lab in enumerate(label_list):
            pass

    cat_per_layer = []
    for li in range(gated_n):
        if not feature_acc[li]:
            cat_per_layer.append(torch.zeros(n_done, n_heads))
            continue
        cat = torch.cat(feature_acc[li], dim=0)
        if cat.shape[0] < n_done:
            pad = torch.zeros(n_done - cat.shape[0], n_heads)
            cat = torch.cat([cat, pad], dim=0)
        cat_per_layer.append(cat[:n_done])
    feat_tensor = torch.stack(cat_per_layer, dim=1)

    if labels_t.shape[0] != n_done:
        labels_t = labels_t[:n_done]

    thresholds = fit_thresholds(
        features=feat_tensor,
        labels=labels_t,
        n_layers=gated_n,
        l2_smoothing=spectral_cfg.l2_smoothing,
    )
    return thresholds
