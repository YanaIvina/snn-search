# Auto-generated – do not edit by hand.
# Model: LSRSNNLadder

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


class LSRSNNLadder(nn.Module):
    """
    Nodes  : inp, stem, upper1, lower1, upper2, lower2, fuse, readout, out
    Topology: stem -> upper1 -> lower1 -> upper2 -> lower2 -> fuse -> readout -> out
    """

    def __init__(
            self,
            inp_size: int,
            stem_size: int,
            upper1_size: int,
            lower1_size: int,
            upper2_size: int,
            lower2_size: int,
            fuse_size: int,
            readout_size: int,
            out_size: int,
            fraction_long: float = 0.5,
            vth: float = 1.0,
            dt: float = 1.0
    ):
        super().__init__()
        self.vth, self.dt = vth, dt

        self.inp_size = inp_size
        self.stem_size = stem_size
        self.upper1_size = upper1_size
        self.lower1_size = lower1_size
        self.upper2_size = upper2_size
        self.lower2_size = lower2_size
        self.fuse_size = fuse_size
        self.readout_size = readout_size
        self.out_size = out_size

        # ── LIF neuron layers ──────────────────────────────
        self.lif_stem = BimodalLIFLayer(
            stem_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_upper1 = BimodalLIFLayer(
            upper1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.6, vth=vth, dt=dt)
        self.lif_lower1 = BimodalLIFLayer(
            lower1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.4, vth=vth, dt=dt)
        self.lif_upper2 = BimodalLIFLayer(
            upper2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.7, vth=vth, dt=dt)
        self.lif_lower2 = BimodalLIFLayer(
            lower2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.7, vth=vth, dt=dt)
        self.lif_fuse = BimodalLIFLayer(
            fuse_size, tau_short=(0, 20), tau_long=(100, 180),
            fraction_long=0.8, vth=vth, dt=dt)
        self.lif_readout = BimodalLIFLayer(
            readout_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)

        # ── Synaptic weight matrices ───────────────────────
        self.w_inp_to_stem = nn.Linear(inp_size, stem_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_stem.weight, a=0)
        self.w_rec_stem = nn.Linear(stem_size, stem_size, bias=False)
        nn.init.orthogonal_(self.w_rec_stem.weight, gain=0.9)
        self.w_stem_to_upper1 = nn.Linear(stem_size, upper1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_stem_to_upper1.weight, a=0)
        self.w_rec_upper1 = nn.Linear(upper1_size, upper1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_upper1.weight, gain=0.9)
        self.w_stem_to_lower1 = nn.Linear(stem_size, lower1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_stem_to_lower1.weight, a=0)
        self.w_rec_lower1 = nn.Linear(lower1_size, lower1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_lower1.weight, gain=0.9)
        self.w_upper1_to_upper2 = nn.Linear(upper1_size, upper2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_upper1_to_upper2.weight, a=0)
        self.w_lower1_to_upper2 = nn.Linear(lower1_size, upper2_size, bias=False)
        nn.init.xavier_uniform_(self.w_lower1_to_upper2.weight)
        self.w_rec_upper2 = nn.Linear(upper2_size, upper2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_upper2.weight, gain=0.9)
        self.w_lower1_to_lower2 = nn.Linear(lower1_size, lower2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_lower1_to_lower2.weight, a=0)
        self.w_upper1_to_lower2 = nn.Linear(upper1_size, lower2_size, bias=False)
        nn.init.xavier_uniform_(self.w_upper1_to_lower2.weight)
        self.w_rec_lower2 = nn.Linear(lower2_size, lower2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_lower2.weight, gain=0.9)
        self.w_stem_to_fuse = nn.Linear(stem_size, fuse_size, bias=False)
        nn.init.xavier_uniform_(self.w_stem_to_fuse.weight)
        self.w_upper2_to_fuse = nn.Linear(upper2_size, fuse_size, bias=False)
        nn.init.kaiming_uniform_(self.w_upper2_to_fuse.weight, a=0)
        self.w_lower2_to_fuse = nn.Linear(lower2_size, fuse_size, bias=False)
        nn.init.kaiming_uniform_(self.w_lower2_to_fuse.weight, a=0)
        self.w_rec_fuse = nn.Linear(fuse_size, fuse_size, bias=False)
        nn.init.orthogonal_(self.w_rec_fuse.weight, gain=0.9)
        self.w_fuse_to_readout = nn.Linear(fuse_size, readout_size, bias=False)
        nn.init.kaiming_uniform_(self.w_fuse_to_readout.weight, a=0)
        self.w_rec_readout = nn.Linear(readout_size, readout_size, bias=False)
        nn.init.orthogonal_(self.w_rec_readout.weight, gain=0.9)
        # Output proj: [fuse, readout] -> out
        self.wout = nn.Linear(fuse_size + readout_size, out_size, bias=False)
        nn.init.kaiming_uniform_(self.wout.weight, a=0)

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
        v_stem = torch.zeros(B, self.stem_size, device=device)
        z_stem = torch.zeros(B, self.stem_size, device=device)
        v_upper1 = torch.zeros(B, self.upper1_size, device=device)
        z_upper1 = torch.zeros(B, self.upper1_size, device=device)
        v_lower1 = torch.zeros(B, self.lower1_size, device=device)
        z_lower1 = torch.zeros(B, self.lower1_size, device=device)
        v_upper2 = torch.zeros(B, self.upper2_size, device=device)
        z_upper2 = torch.zeros(B, self.upper2_size, device=device)
        v_lower2 = torch.zeros(B, self.lower2_size, device=device)
        z_lower2 = torch.zeros(B, self.lower2_size, device=device)
        v_fuse = torch.zeros(B, self.fuse_size, device=device)
        z_fuse = torch.zeros(B, self.fuse_size, device=device)
        v_readout = torch.zeros(B, self.readout_size, device=device)
        z_readout = torch.zeros(B, self.readout_size, device=device)

        outputs = []
        for t in range(T):
            x = x_seq[t]  # (B, inp_size)

            # reservoir: stem
            v_stem = self.lif_stem(
                self.w_inp_to_stem(x) + self.w_rec_stem(z_stem),
                v_stem, z_stem)
            z_stem = self.spike(v_stem)

            # reservoir: upper1
            v_upper1 = self.lif_upper1(
                self.w_stem_to_upper1(z_stem) + self.w_rec_upper1(z_upper1),
                v_upper1, z_upper1)
            z_upper1 = self.spike(v_upper1)

            # reservoir: lower1
            v_lower1 = self.lif_lower1(
                self.w_stem_to_lower1(z_stem) + self.w_rec_lower1(z_lower1),
                v_lower1, z_lower1)
            z_lower1 = self.spike(v_lower1)

            # reservoir: lower2
            v_lower2 = self.lif_lower2(
                self.w_lower1_to_lower2(z_lower1) + self.w_upper1_to_lower2(z_upper1) + self.w_rec_lower2(z_lower2),
                v_lower2, z_lower2)
            z_lower2 = self.spike(v_lower2)

            # reservoir: upper2
            v_upper2 = self.lif_upper2(
                self.w_upper1_to_upper2(z_upper1) + self.w_lower1_to_upper2(z_lower1) + self.w_rec_upper2(z_upper2),
                v_upper2, z_upper2)
            z_upper2 = self.spike(v_upper2)

            # reservoir: fuse
            v_fuse = self.lif_fuse(
                self.w_stem_to_fuse(z_stem) + self.w_upper2_to_fuse(z_upper2) + self.w_lower2_to_fuse(z_lower2) + self.w_rec_fuse(z_fuse),
                v_fuse, z_fuse)
            z_fuse = self.spike(v_fuse)

            # reservoir: readout
            v_readout = self.lif_readout(
                self.w_fuse_to_readout(z_fuse) + self.w_rec_readout(z_readout),
                v_readout, z_readout)
            z_readout = self.spike(v_readout)

            # output projection
            out = self.wout(torch.cat([z_fuse, z_readout], dim=-1))
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (T, B, output_size)


# ── Example usage ────────────────────────────────────────────────────
# import torch
# model = LSRSNNLadder(inp_size=4096, stem_size=768, upper1_size=512, lower1_size=512, upper2_size=384, lower2_size=384, fuse_size=640, readout_size=256, out_size=30)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# model.count_parameters()
#
# T, B = 20, 64   # timesteps, batch size
# x = torch.zeros(T, B, 4096)  # (T, B, inp_size)
# out = model(x)   # → (T, B, out_size)
# pred = out.mean(dim=0).argmax(dim=-1)  # majority vote over time
