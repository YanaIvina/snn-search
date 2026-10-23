# Auto-generated – do not edit by hand.
# Model: LSRSNNHourglass

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


class MultiGaussianSurrogate(torch.autograd.Function):
    """Multi-Gaussian surrogate gradient for backprop through spikes."""

    @staticmethod
    def forward(ctx, input, h, l, s):
        ctx.save_for_backward(input)
        ctx.h, ctx.l, ctx.s = h, l, s
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        h, l, s = ctx.h, ctx.l, ctx.s
        t1 = (1.0 / h) * torch.exp(-inp**2 / (2 * np.sqrt(2 * np.pi) * l))
        t2 = h * torch.exp(-(inp - l*s)**2 / (2 * np.sqrt(2*np.pi) * s * l))
        t3 = h * torch.exp(-(inp + l*s)**2 / (2 * np.sqrt(2*np.pi) * s * l))
        return grad_output * (t1 - t2 - t3), None, None, None


class BimodalLIFLayer(nn.Module):
    """LIF neuron layer with trainable, bimodally distributed time constants."""

    def __init__(self, size, tau_short=(0, 20), tau_long=(80, 100),
                 fraction_long=0.5, vth=1.0, dt=1.0):
        super().__init__()
        a, b = (0 - 50) / 15, (100 - 50) / 15
        tau_m = truncnorm.rvs(a, b, loc=50, scale=15,
                              size=size).astype(np.float32)
        self.tau_m = nn.Parameter(torch.from_numpy(tau_m), requires_grad=True)
        self.vth, self.dt, self.size = vth, dt, size

    def forward(self, x: torch.Tensor, v: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        rho   = torch.exp(-self.dt / self.tau_m.to(x.device))
        return rho * v + (1 - rho) * x - self.vth * z

    def reset(self, batch_size: int, device='cpu') -> torch.Tensor:
        return torch.zeros(batch_size, self.size, device=device)


class LSRSNNHourglass(nn.Module):
    """
    Nodes  : inp, enc1, enc2, lat, dec1, dec2, out
    Topology: enc1 -> enc2 -> lat -> dec1 -> dec2 -> out
    """

    def __init__(
            self,
            inp_size: int,
            enc1_size: int,
            enc2_size: int,
            lat_size: int,
            dec1_size: int,
            dec2_size: int,
            out_size: int,
            fraction_long: float = 0.5,
            vth: float = 1.0,
            dt: float = 1.0
    ):
        super().__init__()
        self.vth, self.dt = vth, dt

        self.inp_size = inp_size
        self.enc1_size = enc1_size
        self.enc2_size = enc2_size
        self.lat_size = lat_size
        self.dec1_size = dec1_size
        self.dec2_size = dec2_size
        self.out_size = out_size

        # ── LIF neuron layers ──────────────────────────────
        self.lif_enc1 = BimodalLIFLayer(
            enc1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_enc2 = BimodalLIFLayer(
            enc2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.7, vth=vth, dt=dt)
        self.lif_lat = BimodalLIFLayer(
            lat_size, tau_short=(0, 20), tau_long=(100, 200),
            fraction_long=1.0, vth=vth, dt=dt)
        self.lif_dec1 = BimodalLIFLayer(
            dec1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.7, vth=vth, dt=dt)
        self.lif_dec2 = BimodalLIFLayer(
            dec2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)

        # ── Synaptic weight matrices ───────────────────────
        self.w_inp_to_enc1 = nn.Linear(inp_size, enc1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_enc1.weight, a=0)
        self.w_rec_enc1 = nn.Linear(enc1_size, enc1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_enc1.weight, gain=0.9)
        self.w_enc1_to_enc2 = nn.Linear(enc1_size, enc2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_enc1_to_enc2.weight, a=0)
        self.w_rec_enc2 = nn.Linear(enc2_size, enc2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_enc2.weight, gain=0.9)
        self.w_enc2_to_lat = nn.Linear(enc2_size, lat_size, bias=False)
        nn.init.kaiming_uniform_(self.w_enc2_to_lat.weight, a=0)
        self.w_rec_lat = nn.Linear(lat_size, lat_size, bias=False)
        nn.init.orthogonal_(self.w_rec_lat.weight, gain=0.9)
        self.w_lat_to_dec1 = nn.Linear(lat_size, dec1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_lat_to_dec1.weight, a=0)
        self.w_enc2_to_dec1 = nn.Linear(enc2_size, dec1_size, bias=False)
        nn.init.xavier_uniform_(self.w_enc2_to_dec1.weight)
        self.w_rec_dec1 = nn.Linear(dec1_size, dec1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_dec1.weight, gain=0.9)
        self.w_dec1_to_dec2 = nn.Linear(dec1_size, dec2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_dec1_to_dec2.weight, a=0)
        self.w_enc1_to_dec2 = nn.Linear(enc1_size, dec2_size, bias=False)
        nn.init.xavier_uniform_(self.w_enc1_to_dec2.weight)
        self.w_rec_dec2 = nn.Linear(dec2_size, dec2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_dec2.weight, gain=0.9)
        self.w_dec2_to_out = nn.Linear(dec2_size, out_size, bias=False)
        nn.init.kaiming_uniform_(self.w_dec2_to_out.weight, a=0)

    def count_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Trainable parameters: {total:,}')
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f'  {name:<50s} {p.numel():>12,}')
        return total

    def spike(self, v: torch.Tensor) -> torch.Tensor:
        return MultiGaussianSurrogate.apply(v - self.vth, 0.15, 0.5, 6)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:  x_seq – (T, B, inp_size)
        Returns: (T, B, out_size)
        """
        T, B, _ = x_seq.shape
        device  = x_seq.device

        # initialise membrane potentials and spike trains
        v_enc1 = torch.zeros(B, self.enc1_size, device=device)
        z_enc1 = torch.zeros(B, self.enc1_size, device=device)
        v_enc2 = torch.zeros(B, self.enc2_size, device=device)
        z_enc2 = torch.zeros(B, self.enc2_size, device=device)
        v_lat = torch.zeros(B, self.lat_size, device=device)
        z_lat = torch.zeros(B, self.lat_size, device=device)
        v_dec1 = torch.zeros(B, self.dec1_size, device=device)
        z_dec1 = torch.zeros(B, self.dec1_size, device=device)
        v_dec2 = torch.zeros(B, self.dec2_size, device=device)
        z_dec2 = torch.zeros(B, self.dec2_size, device=device)

        outputs = []
        for t in range(T):
            x = x_seq[t]  # (B, inp_size)

            # reservoir: enc1
            v_enc1 = self.lif_enc1(
                self.w_inp_to_enc1(x) + self.w_rec_enc1(z_enc1),
                v_enc1, z_enc1)
            z_enc1 = self.spike(v_enc1)

            # reservoir: enc2
            v_enc2 = self.lif_enc2(
                self.w_enc1_to_enc2(z_enc1) + self.w_rec_enc2(z_enc2),
                v_enc2, z_enc2)
            z_enc2 = self.spike(v_enc2)

            # reservoir: lat
            v_lat = self.lif_lat(
                self.w_enc2_to_lat(z_enc2) + self.w_rec_lat(z_lat),
                v_lat, z_lat)
            z_lat = self.spike(v_lat)

            # reservoir: dec1
            v_dec1 = self.lif_dec1(
                self.w_lat_to_dec1(z_lat) + self.w_enc2_to_dec1(z_enc2) + self.w_rec_dec1(z_dec1),
                v_dec1, z_dec1)
            z_dec1 = self.spike(v_dec1)

            # reservoir: dec2
            v_dec2 = self.lif_dec2(
                self.w_dec1_to_dec2(z_dec1) + self.w_enc1_to_dec2(z_enc1) + self.w_rec_dec2(z_dec2),
                v_dec2, z_dec2)
            z_dec2 = self.spike(v_dec2)

            # output projection
            out = self.w_dec2_to_out(z_dec2)
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (T, B, output_size)


# ── Example usage ────────────────────────────────────────────────────
# import torch
# model = LSRSNNHourglass(inp_size=4096, enc1_size=1024, enc2_size=256, lat_size=64, dec1_size=256, dec2_size=1024, out_size=30)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# model.count_parameters()
#
# T, B = 20, 64   # timesteps, batch size
# x = torch.zeros(T, B, 4096)  # (T, B, inp_size)
# out = model(x)   # → (T, B, out_size)
# pred = out.mean(dim=0).argmax(dim=-1)  # majority vote over time
