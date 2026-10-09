import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k, stride=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class CNN1D(nn.Module):
    """Small baseline: 4 conv blocks on raw I/Q -> global average pool -> linear."""

    def __init__(self, n_classes=25, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn(2, w, 7), nn.MaxPool1d(2),
            conv_bn(w, 2 * w, 5), nn.MaxPool1d(2),
            conv_bn(2 * w, 4 * w, 3), nn.MaxPool1d(2),
            conv_bn(4 * w, 4 * w, 3),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.skip(x))


class ResCNN1D(nn.Module):
    """Deeper variant: stem + 8 residual blocks (4 stages)."""

    def __init__(self, n_classes=25, widths=(64, 96, 128, 192)):
        super().__init__()
        layers, cin = [conv_bn(2, widths[0], 7)], widths[0]
        for i, w in enumerate(widths):
            layers += [ResBlock1D(cin, w, 1 if i == 0 else 2), ResBlock1D(w, w, 1)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


class SpecCNN(nn.Module):
    """Spectrogram variant: complex STFT of I+jQ -> log-magnitude (two-sided, fftshifted) -> 2-D CNN."""

    def __init__(self, n_classes=25, n_fft=64, hop=16):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft))

        def c2(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(c2(1, 32), nn.MaxPool2d(2), c2(32, 64), nn.MaxPool2d(2),
                                      c2(64, 128), nn.MaxPool2d(2), c2(128, 128))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def spectrogram(self, x):
        with torch.autocast("cuda", enabled=False):
            z = torch.complex(x[:, 0].float(), x[:, 1].float())
            S = torch.stft(z, self.n_fft, self.hop, window=self.win, return_complex=True, onesided=False)
            S = torch.fft.fftshift(S, dim=1)
            m = torch.log1p(S.abs())
            m = (m - m.mean((1, 2), keepdim=True)) / (m.std((1, 2), keepdim=True) + 1e-5)
        return m.unsqueeze(1)

    def forward(self, x):
        return self.head(self.features(self.spectrogram(x)).mean((-1, -2)))


def build_model(name):
    return {"cnn1d": CNN1D, "rescnn1d": ResCNN1D, "spec2d": SpecCNN}[name]()
