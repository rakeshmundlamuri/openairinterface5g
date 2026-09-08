"""Deep joint source-channel coding (JSCC) model for image transmission.

Independent implementation of the encoder/decoder architecture described in
Bourtsoulatze, Burth Kurka, Gunduz, "Deep Joint Source-Channel Coding for Wireless
Image Transmission" (IEEE TCCN 2019), cross-checked against the layer shapes used by
the reference implementation at https://github.com/chunbaobao/Deep-JSCC-PyTorch (no
LICENSE file there, so this is written from scratch rather than copied).

The encoder maps a (3, 32, 32) CIFAR-10-sized image to a (2*c, 8, 8) real tensor: the
first c channels are the real parts and the next c channels the imaginary parts of c
complex symbols at each of the 64 spatial positions, giving c*64 complex channel
symbols total. A final normalization layer scales the whole per-image vector to a
fixed average power P (individual symbols can still exceed magnitude sqrt(P) since
only the average is constrained).

`Channel` (AWGN/Rayleigh) is used only for training-time robustness - the real
transport (rfsim + the custom RE PDSCH feature) replaces it at inference time.
"""
import torch
import torch.nn as nn


class _ConvPReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.act = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.act(self.conv(x))


class _TransConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0, output_padding=0, act=None):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
        self.act = act if act is not None else nn.PReLU()
        if isinstance(self.act, nn.PReLU):
            nn.init.kaiming_normal_(self.tconv.weight, mode="fan_out", nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(self.tconv.weight)

    def forward(self, x):
        return self.act(self.tconv(x))


def _power_normalize(z, power=1.0):
    # z: (B, 2c, 8, 8). Scale each sample so its whole flattened vector has average
    # power `power` per element (sum of squares = power * numel_per_sample).
    b = z.size(0)
    k = z[0].numel()
    flat = z.reshape(b, -1)
    norm = flat.norm(dim=1, keepdim=True)
    scale = (power * k) ** 0.5 / norm
    return z * scale.view(b, 1, 1, 1)


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = _ConvPReLU(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvPReLU(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvPReLU(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = _ConvPReLU(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv5 = _ConvPReLU(32, 2 * c, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return _power_normalize(x, power=1.0)


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tconv1 = _TransConvAct(2 * c, 32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvAct(32, 32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvAct(32, 32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvAct(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvAct(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1, act=nn.Sigmoid())

    def forward(self, z):
        x = self.tconv1(z)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


class Channel(nn.Module):
    """Training-time-only channel simulation (never used for the real transport)."""

    def __init__(self, channel_type="AWGN", snr_db=10.0):
        super().__init__()
        if channel_type not in ("AWGN", "Rayleigh"):
            raise ValueError(f"unknown channel type {channel_type}")
        self.channel_type = channel_type
        self.snr_db = snr_db

    def forward(self, z):
        b = z.size(0)
        k = z[0].numel()
        sig_pwr = (z.reshape(b, -1) ** 2).sum(dim=1) / k
        noise_pwr = sig_pwr / (10 ** (self.snr_db / 10))
        noise = torch.randn_like(z) * noise_pwr.sqrt().view(b, 1, 1, 1)
        if self.channel_type == "Rayleigh":
            c = z.size(1) // 2
            h_re = torch.randn(b, device=z.device).view(b, 1, 1, 1)
            h_im = torch.randn(b, device=z.device).view(b, 1, 1, 1)
            zr, zi = z[:, :c], z[:, c:]
            z = torch.cat([h_re * zr - h_im * zi, h_re * zi + h_im * zr], dim=1)
        return z + noise


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type=None, snr_db=None):
        super().__init__()
        self.c = c
        self.encoder = Encoder(c)
        self.decoder = Decoder(c)
        self.channel = Channel(channel_type, snr_db) if channel_type else None

    def forward(self, x):
        z = self.encoder(x)
        if self.channel is not None:
            z = self.channel(z)
        return self.decoder(z)

    def set_channel(self, channel_type=None, snr_db=None):
        self.channel = Channel(channel_type, snr_db) if channel_type else None
