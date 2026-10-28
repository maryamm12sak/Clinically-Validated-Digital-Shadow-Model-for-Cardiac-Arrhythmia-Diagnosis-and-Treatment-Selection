"""
1D CNNs for 12-lead ECG. Input (B, T, 12) as produced by preprocessing; output raw logits (B, 12)
in labels.yaml order. Any input length works (global pooling), e.g. 2.5 s crops or 10 s records.

  simple_cnn  FCN-style baseline: 3 conv blocks (128-256-128, kernels 8/5/3), BN, ReLU, global pool
  resnet1d    xresnet1d-style: 3-conv stem + 4 stages of basic residual blocks
Both take concat_pool=True to use [avg, max] pooling instead of avg only.
"""
import torch
from torch import nn


def conv_bn_relu(c_in, c_out, kernel, stride=1, relu=True):
    layers = [nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
              nn.BatchNorm1d(c_out)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PoolHead(nn.Module):
    def __init__(self, c_in, n_classes, concat_pool=False, dropout=0.0):
        super().__init__()
        self.concat_pool = concat_pool
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in * (2 if concat_pool else 1), n_classes)

    def forward(self, x):                      # x: (B, C, T)
        pooled = x.mean(dim=-1)
        if self.concat_pool:
            pooled = torch.cat([pooled, x.amax(dim=-1)], dim=1)
        return self.fc(self.dropout(pooled))


class SimpleCNN(nn.Module):
    def __init__(self, n_leads=12, n_classes=12, concat_pool=False, dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(n_leads, 128, 8), conv_bn_relu(128, 256, 5),
                                  conv_bn_relu(256, 128, 3))
        self.head = PoolHead(128, n_classes, concat_pool, dropout)

    def forward(self, x):                      # x: (B, T, 12)
        return self.head(self.body(x.transpose(1, 2)))


class BasicBlock1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride):
        super().__init__()
        self.convs = nn.Sequential(conv_bn_relu(c_in, c_out, kernel, stride),
                                   conv_bn_relu(c_out, c_out, kernel, relu=False))
        nn.init.zeros_(self.convs[1][1].weight)        # zero-init last BN (xresnet)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(stride, ceil_mode=True) if stride != 1 else nn.Identity(),
                conv_bn_relu(c_in, c_out, 1, relu=False))
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convs(x)
        sc = self.shortcut(x)
        if sc.shape[-1] != out.shape[-1]:
            sc = sc[..., :out.shape[-1]]
        return self.act(out + sc)


class ResNet1d(nn.Module):
    def __init__(self, n_leads=12, n_classes=12, widths=(64, 128, 128, 256), blocks=(2, 2, 2, 2),
                 kernel=5, stem_width=32, concat_pool=True, dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(n_leads, stem_width, kernel, stride=2),
                                  conv_bn_relu(stem_width, stem_width, kernel),
                                  conv_bn_relu(stem_width, widths[0], kernel),
                                  nn.MaxPool1d(3, stride=2, padding=1))
        stages, c_in = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                stride = 2 if (b == 0 and i > 0) else 1
                stages.append(BasicBlock1d(c_in, w, kernel, stride))
                c_in = w
        self.stages = nn.Sequential(*stages)
        self.head = PoolHead(c_in, n_classes, concat_pool, dropout)

    def forward(self, x):                      # x: (B, T, 12)
        return self.head(self.stages(self.stem(x.transpose(1, 2))))


DEEP_MODELS = {"simple_cnn": SimpleCNN, "resnet1d": ResNet1d}


def build_model(name, n_classes=12, **kwargs):
    if name not in DEEP_MODELS:
        raise ValueError(f"unknown deep model {name!r}")
    return DEEP_MODELS[name](n_classes=n_classes, **kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in DEEP_MODELS:
        print(f"{name}: {count_params(build_model(name)):,} parameters")
