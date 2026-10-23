# Auto-generated – do not edit by hand.
# Model: LSRSNNSkip

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


class LSRSNNSkip(nn.Module):
    """
    Nodes  : inp, res1, res2, out
    Topology: res1 -> res2 -> out
    """

    def __init__(
            self,
            inp_size: int,
            res1_size: int,
            res2_size: int,
            out_size: int,
            fraction_long: float = 0.5,
            vth: float = 1.0,
            dt: float = 1.0
    ):
        super().__init__()
        self.vth, self.dt = vth, dt

        self.inp_size = inp_size
        self.res1_size = res1_size
        self.res2_size = res2_size
        self.out_size = out_size

        # ── LIF neuron layers ──────────────────────────────
        self.lif_res1 = BimodalLIFLayer(
            res1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_res2 = BimodalLIFLayer(
            res2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)

        # ── Synaptic weight matrices ───────────────────────
        self.w_inp_to_res1 = nn.Linear(inp_size, res1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_res1.weight, a=0)
        self.w_rec_res1 = nn.Linear(res1_size, res1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_res1.weight, gain=0.9)
        self.w_res1_to_res2 = nn.Linear(res1_size, res2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_res1_to_res2.weight, a=0)
        self.w_rec_res2 = nn.Linear(res2_size, res2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_res2.weight, gain=0.9)
        # Output proj: [res1, res2] -> out
        self.wout = nn.Linear(res1_size + res2_size, out_size, bias=False)
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
        v_res1 = torch.zeros(B, self.res1_size, device=device)
        z_res1 = torch.zeros(B, self.res1_size, device=device)
        v_res2 = torch.zeros(B, self.res2_size, device=device)
        z_res2 = torch.zeros(B, self.res2_size, device=device)

        outputs = []
        for t in range(T):
            x = x_seq[t]  # (B, inp_size)

            # reservoir: res1
            v_res1 = self.lif_res1(
                self.w_inp_to_res1(x) + self.w_rec_res1(z_res1),
                v_res1, z_res1)
            z_res1 = self.spike(v_res1)

            # reservoir: res2
            v_res2 = self.lif_res2(
                self.w_res1_to_res2(z_res1) + self.w_rec_res2(z_res2),
                v_res2, z_res2)
            z_res2 = self.spike(v_res2)

            # output projection
            out = self.wout(torch.cat([z_res1, z_res2], dim=-1))
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (T, B, output_size)


# ── Example usage ────────────────────────────────────────────────────
# import torch
# model = LSRSNNSkip(inp_size=4096, res1_size=800, res2_size=800, out_size=30)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# model.count_parameters()
#
# T, B = 20, 64   # timesteps, batch size
# x = torch.zeros(T, B, 4096)  # (T, B, inp_size)
# out = model(x)   # → (T, B, out_size)
# pred = out.mean(dim=0).argmax(dim=-1)  # majority vote over time
