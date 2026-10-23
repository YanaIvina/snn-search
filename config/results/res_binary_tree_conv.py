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


class SpikingConvBlock(nn.Module):

    def __init__(self, cin, cout, hw, usebn=True, dt=1.0, vth=1.0,
                 fraction_long=0.5):
        super().__init__()
        h, w = hw if isinstance(hw, (tuple, list)) else (hw, hw)
        self.conv      = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not usebn)
        self.bn        = nn.BatchNorm2d(cout) if usebn else nn.Identity()
        self.pool      = nn.AvgPool2d(2)
        self.vth       = vth
        self.feat_size = cout * h * w
        self.lif       = BimodalLIFLayer(
            size=self.feat_size, fraction_long=fraction_long, dt=dt, vth=vth,
        )

    def init_state(self, batch_size, device, dtype=torch.float32):
        return {
            'v': torch.zeros(batch_size, self.feat_size, device=device, dtype=dtype),
            'z': torch.zeros(batch_size, self.feat_size, device=device, dtype=dtype),
        }

    def forward_step(self, x, state, spike_params):
        hs, ls, ss = spike_params
        y      = self.bn(self.conv(x))
        y_flat = y.flatten(1)
        v_new  = self.lif(y_flat, state['v'], state['z'])
        z_new  = MultiGaussianSurrogate.apply(v_new - self.vth, hs, ls, ss)
        y_out  = self.pool(z_new.view_as(y))
        state['v'], state['z'] = v_new, z_new
        return y_out, state

class SpikingConvFrontEnd(nn.Module):

    def __init__(self, inch=1, conv_channels=(32, 64, 128),
                 usebn=True, hw=(128, 32),
                 dt=1.0, vth=1.0, fraction_long=0.5,
                 spike_params=(0.15, 0.5, 6)):
        super().__init__()
        h, w = hw if isinstance(hw, (tuple, list)) else (hw, hw)
        self.spike_params = spike_params
        self.blocks = nn.ModuleList()
        prev = inch
        for ch in conv_channels:
            self.blocks.append(
                SpikingConvBlock(prev, ch, (h, w), usebn, dt, vth, fraction_long)
            )
            prev, h, w = ch, h // 2, w // 2
        self.feat_dim = prev * h * w

    def init_state(self, batch_size, device, dtype=torch.float32):
        return [blk.init_state(batch_size, device, dtype)
                for blk in self.blocks]

    def forward_step(self, x, state, spike_params):
        for i, blk in enumerate(self.blocks):
            x, state[i] = blk.forward_step(x, state[i], spike_params)
        return x, state

    def forward(self, x):
        B = x.shape[0]
        state = self.init_state(B, x.device, x.dtype)
        out, _ = self.forward_step(x, state, self.spike_params)
        return out

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

        self.conv_frontend = SpikingConvFrontEnd(
            inch=1, conv_channels=(32, 64, 128),
            usebn=True, hw=(128, 32),
            dt=dt, vth=vth, fraction_long=fraction_long,
            spike_params=(0.15, 0.5, 6),
        )
        self.cnn_out_size = self.conv_frontend.feat_dim
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
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_agg2 = BimodalLIFLayer(
            agg2_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)
        self.lif_root = BimodalLIFLayer(
            root_size, tau_short=(0, 20), tau_long=(80, 100),
            fraction_long=fraction_long, vth=vth, dt=dt)

        # ── Synaptic weight matrices ───────────────────────
        self.w_inp_to_resA1 = nn.Linear(self._cnn_out_size, resA1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_resA1.weight, a=0)
        self.w_rec_resA1 = nn.Linear(resA1_size, resA1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resA1.weight, gain=0.9)
        self.w_inp_to_resA2 = nn.Linear(self._cnn_out_size, resA2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_resA2.weight, a=0)
        self.w_rec_resA2 = nn.Linear(resA2_size, resA2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resA2.weight, gain=0.9)
        self.w_inp_to_resB1 = nn.Linear(self._cnn_out_size, resB1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_resB1.weight, a=0)
        self.w_rec_resB1 = nn.Linear(resB1_size, resB1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resB1.weight, gain=0.9)
        self.w_inp_to_resB2 = nn.Linear(self._cnn_out_size, resB2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_inp_to_resB2.weight, a=0)
        self.w_rec_resB2 = nn.Linear(resB2_size, resB2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_resB2.weight, gain=0.9)
        self.w_resA1_to_agg1 = nn.Linear(resA1_size, agg1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_resA1_to_agg1.weight, a=0)
        self.w_resA2_to_agg1 = nn.Linear(resA2_size, agg1_size, bias=False)
        nn.init.kaiming_uniform_(self.w_resA2_to_agg1.weight, a=0)
        self.w_rec_agg1 = nn.Linear(agg1_size, agg1_size, bias=False)
        nn.init.orthogonal_(self.w_rec_agg1.weight, gain=0.9)
        self.w_resB1_to_agg2 = nn.Linear(resB1_size, agg2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_resB1_to_agg2.weight, a=0)
        self.w_resB2_to_agg2 = nn.Linear(resB2_size, agg2_size, bias=False)
        nn.init.kaiming_uniform_(self.w_resB2_to_agg2.weight, a=0)
        self.w_rec_agg2 = nn.Linear(agg2_size, agg2_size, bias=False)
        nn.init.orthogonal_(self.w_rec_agg2.weight, gain=0.9)
        self.w_agg1_to_root = nn.Linear(agg1_size, root_size, bias=False)
        nn.init.kaiming_uniform_(self.w_agg1_to_root.weight, a=0)
        self.w_agg2_to_root = nn.Linear(agg2_size, root_size, bias=False)
        nn.init.kaiming_uniform_(self.w_agg2_to_root.weight, a=0)
        self.w_rec_root = nn.Linear(root_size, root_size, bias=False)
        nn.init.orthogonal_(self.w_rec_root.weight, gain=0.9)
        self.w_root_to_out = nn.Linear(root_size, out_size, bias=False)
        nn.init.kaiming_uniform_(self.w_root_to_out.weight, a=0)

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

        # ── CNN front-end: encode mean frame once (outside time loop) ──
        x_img  = x_seq.mean(dim=0).view(B, 1, 128, 32)
        cnn_feat = self.conv_frontend(x_img).flatten(1)  # B, cnn_out_size

        x_proj_inp_to_resA1 = self.w_inp_to_resA1(cnn_feat).unsqueeze(0).expand(T, -1, -1)  # (T, B, resA1_size)
        x_proj_inp_to_resA2 = self.w_inp_to_resA2(cnn_feat).unsqueeze(0).expand(T, -1, -1)  # (T, B, resA2_size)
        x_proj_inp_to_resB1 = self.w_inp_to_resB1(cnn_feat).unsqueeze(0).expand(T, -1, -1)  # (T, B, resB1_size)
        x_proj_inp_to_resB2 = self.w_inp_to_resB2(cnn_feat).unsqueeze(0).expand(T, -1, -1)  # (T, B, resB2_size)

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
            # reservoir: resA1
            v_resA1 = self.lif_resA1(
                x_proj_inp_to_resA1[t] + self.w_rec_resA1(z_resA1),
                v_resA1, z_resA1)
            z_resA1 = self.spike(v_resA1)

            # reservoir: resA2
            v_resA2 = self.lif_resA2(
                x_proj_inp_to_resA2[t] + self.w_rec_resA2(z_resA2),
                v_resA2, z_resA2)
            z_resA2 = self.spike(v_resA2)

            # reservoir: resB1
            v_resB1 = self.lif_resB1(
                x_proj_inp_to_resB1[t] + self.w_rec_resB1(z_resB1),
                v_resB1, z_resB1)
            z_resB1 = self.spike(v_resB1)

            # reservoir: resB2
            v_resB2 = self.lif_resB2(
                x_proj_inp_to_resB2[t] + self.w_rec_resB2(z_resB2),
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
