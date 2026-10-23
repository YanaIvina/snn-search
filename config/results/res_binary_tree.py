# Auto-generated – do not edit by hand.
# Model: LSRSNNBinaryTree

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


class LSRSNNBinaryTree(nn.Module):
    """
    Nodes  : inp, resA1, resA2, resB1, resB2, agg1, agg2, root, out
    Topology: resA1 -> resA2 -> resB1 -> resB2 -> agg1 -> agg2 -> root -> out
    """

    def __init__(
            self,
            inp_size: int,
            resA1_size: int,
            resA2_size: int,
            resB1_size: int,
            resB2_size: int,
            agg1_size: int,
            agg2_size: int,
            root_size: int,
            out_size: int,
            fraction_long: float = 0.5,
            vth: float = 1.0,
            dt: float = 1.0
    ):
        super().__init__()
        self.vth, self.dt = vth, dt

        self.inp_size = inp_size
        self.resA1_size = resA1_size
        self.resA2_size = resA2_size
        self.resB1_size = resB1_size
        self.resB2_size = resB2_size
        self.agg1_size = agg1_size
        self.agg2_size = agg2_size
        self.root_size = root_size
        self.out_size = out_size

        # ── LIF neuron layers ──────────────────────────────
        self.lif_resA1 = BimodalLIFLayer(
            resA1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_resA2 = BimodalLIFLayer(
            resA2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_resB1 = BimodalLIFLayer(
            resB1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_resB2 = BimodalLIFLayer(
            resB2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_agg1 = BimodalLIFLayer(
            agg1_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.6, vth=vth, dt=dt)
        self.lif_agg2 = BimodalLIFLayer(
            agg2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.6, vth=vth, dt=dt)
        self.lif_root = BimodalLIFLayer(
            root_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=0.8, vth=vth, dt=dt)

        # ── Synaptic weight matrices ───────────────────────
        self.w_inp_to_resA1 = nn.Linear(inp_size, resA1_size, bias=False)
        nn.init.uniform_(self.w_inp_to_resA1.weight, a=-0.5, b=0.5)
        self.w_rec_resA1 = nn.Linear(resA1_size, resA1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resA1.weight, gain=1.0)
        self.w_inp_to_resA2 = nn.Linear(inp_size, resA2_size, bias=False)
        nn.init.uniform_(self.w_inp_to_resA2.weight, a=-0.5, b=0.5)
        self.w_rec_resA2 = nn.Linear(resA2_size, resA2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resA2.weight, gain=1.0)
        self.w_inp_to_resB1 = nn.Linear(inp_size, resB1_size, bias=False)
        nn.init.uniform_(self.w_inp_to_resB1.weight, a=-0.5, b=0.5)
        self.w_rec_resB1 = nn.Linear(resB1_size, resB1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resB1.weight, gain=1.0)
        self.w_inp_to_resB2 = nn.Linear(inp_size, resB2_size, bias=False)
        nn.init.uniform_(self.w_inp_to_resB2.weight, a=-0.5, b=0.5)
        self.w_rec_resB2 = nn.Linear(resB2_size, resB2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resB2.weight, gain=1.0)
        self.w_resA1_to_agg1 = nn.Linear(resA1_size, agg1_size, bias=False)
        nn.init.uniform_(self.w_resA1_to_agg1.weight, a=-0.5, b=0.5)
        self.w_resA2_to_agg1 = nn.Linear(resA2_size, agg1_size, bias=False)
        nn.init.uniform_(self.w_resA2_to_agg1.weight, a=-0.5, b=0.5)
        self.w_rec_agg1 = nn.Linear(agg1_size, agg1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_agg1.weight, gain=1.0)
        self.w_resB1_to_agg2 = nn.Linear(resB1_size, agg2_size, bias=False)
        nn.init.uniform_(self.w_resB1_to_agg2.weight, a=-0.5, b=0.5)
        self.w_resB2_to_agg2 = nn.Linear(resB2_size, agg2_size, bias=False)
        nn.init.uniform_(self.w_resB2_to_agg2.weight, a=-0.5, b=0.5)
        self.w_rec_agg2 = nn.Linear(agg2_size, agg2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_agg2.weight, gain=1.0)
        self.w_agg1_to_root = nn.Linear(agg1_size, root_size, bias=False)
        nn.init.uniform_(self.w_agg1_to_root.weight, a=-0.5, b=0.5)
        self.w_agg2_to_root = nn.Linear(agg2_size, root_size, bias=False)
        nn.init.uniform_(self.w_agg2_to_root.weight, a=-0.5, b=0.5)
        self.w_rec_root = nn.Linear(root_size, root_size, bias=False)
        nn.init.orthogonal_(self.w_rec_root.weight, gain=1.0)
        self.w_root_to_out = nn.Linear(root_size, out_size, bias=False)
        nn.init.uniform_(self.w_root_to_out.weight, a=-0.5, b=0.5)

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
        v_resA1 = torch.zeros(B, self.resA1_size, device=device)
        z_resA1 = torch.zeros(B, self.resA1_size, device=device)
        v_resA2 = torch.zeros(B, self.resA2_size, device=device)
        z_resA2 = torch.zeros(B, self.resA2_size, device=device)
        v_resB1 = torch.zeros(B, self.resB1_size, device=device)
        z_resB1 = torch.zeros(B, self.resB1_size, device=device)
        v_resB2 = torch.zeros(B, self.resB2_size, device=device)
        z_resB2 = torch.zeros(B, self.resB2_size, device=device)
        v_agg1 = torch.zeros(B, self.agg1_size, device=device)
        z_agg1 = torch.zeros(B, self.agg1_size, device=device)
        v_agg2 = torch.zeros(B, self.agg2_size, device=device)
        z_agg2 = torch.zeros(B, self.agg2_size, device=device)
        v_root = torch.zeros(B, self.root_size, device=device)
        z_root = torch.zeros(B, self.root_size, device=device)

        outputs = []
        for t in range(T):
            x = x_seq[t]  # (B, inp_size)

            # reservoir: resA1
            v_resA1 = self.lif_resA1(
                self.w_inp_to_resA1(x) + self.w_rec_resA1(z_resA1),
                v_resA1, z_resA1)
            z_resA1 = self.spike(v_resA1)

            # reservoir: resA2
            v_resA2 = self.lif_resA2(
                self.w_inp_to_resA2(x) + self.w_rec_resA2(z_resA2),
                v_resA2, z_resA2)
            z_resA2 = self.spike(v_resA2)

            # reservoir: resB1
            v_resB1 = self.lif_resB1(
                self.w_inp_to_resB1(x) + self.w_rec_resB1(z_resB1),
                v_resB1, z_resB1)
            z_resB1 = self.spike(v_resB1)

            # reservoir: resB2
            v_resB2 = self.lif_resB2(
                self.w_inp_to_resB2(x) + self.w_rec_resB2(z_resB2),
                v_resB2, z_resB2)
            z_resB2 = self.spike(v_resB2)

            # reservoir: agg1
            v_agg1 = self.lif_agg1(
                self.w_resA1_to_agg1(z_resA1) + self.w_resA2_to_agg1(z_resA2) + self.w_rec_agg1(z_agg1),
                v_agg1, z_agg1)
            z_agg1 = self.spike(v_agg1)

            # reservoir: agg2
            v_agg2 = self.lif_agg2(
                self.w_resB1_to_agg2(z_resB1) + self.w_resB2_to_agg2(z_resB2) + self.w_rec_agg2(z_agg2),
                v_agg2, z_agg2)
            z_agg2 = self.spike(v_agg2)

            # reservoir: root
            v_root = self.lif_root(
                self.w_agg1_to_root(z_agg1) + self.w_agg2_to_root(z_agg2) + self.w_rec_root(z_root),
                v_root, z_root)
            z_root = self.spike(v_root)

            # output projection
            out = self.w_root_to_out(z_root)
            outputs.append(out)

        return torch.stack(outputs, dim=0)  # (T, B, output_size)


# ── Example usage ────────────────────────────────────────────────────
# import torch
# model = LSRSNNBinaryTree(inp_size=4096, resA1_size=700, resA2_size=700, resB1_size=700, resB2_size=700, agg1_size=700, agg2_size=700, root_size=700, out_size=30)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# model.count_parameters()
#
# T, B = 20, 64   # timesteps, batch size
# x = torch.zeros(T, B, 4096)  # (T, B, inp_size)
# out = model(x)   # → (T, B, out_size)
# pred = out.mean(dim=0).argmax(dim=-1)  # majority vote over time
