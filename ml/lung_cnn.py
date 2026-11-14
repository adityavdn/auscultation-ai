"""ImageNet-pretrained EfficientNet-B0 fine-tuned on log-mel breath cycles; exported to ONNX for serving.

Training-only module (needs torch/torchvision). The app serves the ONNX file with onnxruntime.
"""
import copy
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ml.metrics import icbhi_score

log = logging.getLogger(__name__)


class SpecNet(nn.Module):
    """Wraps an image CNN: [B, 128 mels, T] -> 3 identical channels -> class logits."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, n_classes)

    def forward(self, x):
        return self.net(x.unsqueeze(1).expand(-1, 3, -1, -1))


class _Softmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), 1)


def build_model(n_classes, pretrained=True):
    return SpecNet(n_classes, pretrained)


def _augment(xb, rng):
    """SpecAugment-style: 2 frequency masks (<=16 bins), 2 time masks (<=10% of frames), random time roll."""
    xb = xb.clone()
    n_mels, t = xb.shape[1], xb.shape[2]
    for i in range(len(xb)):
        for _ in range(2):
            f = int(rng.integers(0, 17)); f0 = int(rng.integers(0, n_mels - f))
            xb[i, f0:f0 + f, :] = 0
            w = int(rng.integers(0, max(1, t // 10))); t0 = int(rng.integers(0, t - w))
            xb[i, :, t0:t0 + w] = 0
        xb[i] = torch.roll(xb[i], int(rng.integers(0, t)), dims=1)
    return xb


def _predict(model, X, device, batch=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def fit(X, y, Xv, yv, labels, normal_label="normal", epochs=30, batch=32, lr=3e-4, pretrained=True,
        device=None, seed=42):
    """Train with class-weighted CE; keep the epoch with the best validation ICBHI score."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = {lab: i for i, lab in enumerate(labels)}
    yi = np.array([idx[v] for v in y])
    counts = np.bincount(yi, minlength=len(labels))
    weights = torch.tensor((counts.max() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=device)
    model = build_model(len(labels), pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = order[i:i + batch]
            xb = _augment(torch.from_numpy(X[b]), rng).to(device)
            loss = loss_fn(model(xb), torch.from_numpy(yi[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pv = _predict(model, Xv, device)
        score = icbhi_score(list(yv), [labels[k] for k in pv.argmax(1)], normal_label=normal_label)["score"]
        history.append(round(float(score), 4))
        log.info("epoch %d val ICBHI %.4f", ep + 1, score)
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu(), {"best_val_score": float(best), "history": history}


def export_onnx(model, path, n_frames):
    path = Path(path)
    wrapped = _Softmax(model.cpu().eval())
    torch.onnx.export(wrapped, torch.zeros(1, 128, n_frames), str(path), input_names=["mel"], output_names=["probs"],
                      dynamic_axes={"mel": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    return path
