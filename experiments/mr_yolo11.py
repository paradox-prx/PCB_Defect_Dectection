"""
MR-YOLOv11: Modality-Robust YOLOv11 for cross-dataset PCB defect detection.

Two architectural modifications to stock YOLOv11n:
  1. IBN backbone     -- split-channel Instance-Batch Norm in layers 0..4
                         (stem + first two stages). IN half removes per-image
                         style statistics; BN half preserves content.
  2. Edge-Prior Stem  -- Sobel/Laplacian-initialised gradient branch on the
                         grayscale input, fused into the stem via 1x1 Conv.

Usage (training):
    from ultralytics import YOLO
    from mr_yolo11 import MRTrainer
    model = YOLO("yolo11n.pt")
    model.train(trainer=MRTrainer, data="data.yaml", epochs=100, imgsz=640)

Ablation trainers: IBNOnlyTrainer, EdgeOnlyTrainer.

IMPORTANT: always `import mr_yolo11` BEFORE loading a trained MR checkpoint
(YOLO("runs/.../best.pt")) -- the pickled model needs these classes, and the
import also installs an IBN-safe model.fuse() patch used at validation time.
"""

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import BaseModel

N_EARLY_LAYERS = 5   # yolo11 backbone: 0 Conv, 1 Conv, 2 C3k2, 3 Conv, 4 C3k2
IBN_RATIO = 0.5


# ---------------------------------------------------------------- IBN module
class IBN(nn.Module):
    """Split-channel Instance/Batch Norm (Pan et al., ECCV 2018, IBN-a style)."""

    def __init__(self, planes: int, ratio: float = IBN_RATIO):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, [self.half, x.shape[1] - self.half], dim=1)
        return torch.cat([self.IN(a.contiguous()), self.BN(b.contiguous())], 1)


def _swap_bn_to_ibn(conv_block: Conv):
    """Replace conv_block.bn (BatchNorm2d) with IBN, warm-starting affine params."""
    bn = conv_block.bn
    ibn = IBN(bn.num_features)
    with torch.no_grad():
        h = ibn.half
        ibn.IN.weight.copy_(bn.weight[:h])
        ibn.IN.bias.copy_(bn.bias[:h])
        ibn.BN.weight.copy_(bn.weight[h:])
        ibn.BN.bias.copy_(bn.bias[h:])
        ibn.BN.running_mean.copy_(bn.running_mean[h:])
        ibn.BN.running_var.copy_(bn.running_var[h:])
    conv_block.bn = ibn


def apply_ibn(det_model, n_layers: int = N_EARLY_LAYERS):
    """Swap BN -> IBN in every Conv block of the first n_layers backbone layers."""
    n = 0
    for layer in det_model.model[:n_layers]:
        for m in layer.modules():
            if isinstance(m, Conv) and isinstance(getattr(m, "bn", None), nn.BatchNorm2d):
                _swap_bn_to_ibn(m)
                n += 1
    print(f"[MR-YOLO] IBN installed in {n} Conv blocks (layers 0..{n_layers - 1})")
    return det_model


# --------------------------------------------------------- Edge-Prior Stem
class EdgePriorStem(nn.Module):
    """Wraps the original stem Conv and adds a gradient-prior branch.

    Branch: grayscale -> 3x3 conv (stride matches stem) initialised with
    Sobel-x/y, Laplacian and their negatives; concatenated with the stem
    output and fused back to the original channel count by a 1x1 Conv.
    """

    def __init__(self, stem: Conv, edge_ch: int = 8):
        super().__init__()
        self.stem = stem
        c_out = stem.conv.out_channels
        stride = stem.conv.stride[0]
        self.edge = nn.Conv2d(1, edge_ch, 3, stride=stride, padding=1, bias=False)
        with torch.no_grad():
            kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            ky = kx.t().clone()
            lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            kernels = [kx, ky, lap, -kx, -ky, -lap, kx + ky, kx - ky]
            for i in range(min(edge_ch, len(kernels))):
                self.edge.weight[i, 0].copy_(kernels[i])
        self.fuse = Conv(c_out + edge_ch, c_out, 1, 1)  # conv+bn+SiLU
        # graph bookkeeping expected by ultralytics forward loop
        self.i, self.f, self.type = stem.i, stem.f, "EdgePriorStem"
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)              # luminance
        return self.fuse(torch.cat([self.stem(x), self.edge(g)], dim=1))


def add_edge_stem(det_model, edge_ch: int = 8):
    det_model.model[0] = EdgePriorStem(det_model.model[0], edge_ch)
    print("[MR-YOLO] Edge-Prior Stem installed")
    return det_model


# ------------------------------------------------ IBN-safe fuse() at val/export
_orig_fuse = BaseModel.fuse

def _ibn_safe_fuse(self, verbose=True):
    """fuse_conv_and_bn() crashes on IBN (no running stats); hide IBN blocks
    during fusion and restore them afterwards (they stay unfused, which is fine)."""
    stash = []
    for m in self.model.modules():
        if isinstance(m, Conv) and isinstance(getattr(m, "bn", None), IBN):
            stash.append((m, m.bn))
            del m.bn
    out = _orig_fuse(self, verbose)
    for m, bn in stash:
        m.bn = bn
    return out

BaseModel.fuse = _ibn_safe_fuse


# ------------------------------------------------------------------ Trainers
class MRTrainer(DetectionTrainer):
    """Builds YOLOv11, then applies MR surgery inside get_model() -- i.e. BEFORE
    the optimizer is created, so every new parameter is registered and trained."""

    use_ibn = True
    use_edge_stem = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        if self.use_ibn:
            apply_ibn(model)
        if self.use_edge_stem:
            add_edge_stem(model)
        return model


class IBNOnlyTrainer(MRTrainer):
    use_edge_stem = False


class EdgeOnlyTrainer(MRTrainer):
    use_ibn = False
