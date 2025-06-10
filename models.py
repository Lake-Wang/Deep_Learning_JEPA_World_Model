from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import copy

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        layers = []
        out_channels = [repr_dim // (2 ** i) for i in range(4, 0, -1)]

        in_c = 1
        for out_c in out_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool2d((4, 4))) # [B*T, repr_dim // 2, 4, 4]
        layers.append(nn.Flatten())
        layers.append(nn.Linear(int(repr_dim // 2) * 4 * 4, int(repr_dim // 2)))

        self.encoder1 = nn.Sequential(*layers)
        self.encoder2 = copy.deepcopy(self.encoder1)

    def forward(self, x):
        return torch.cat(
            [self.encoder1(x[:, 0:1]), self.encoder2(x[:, 1:2])], 
            dim=1
        )

    
class JEPA(nn.Module):
    def __init__(
        self,
        repr_dim,
        hidden_dim,
        action_dim=2
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        self.encoder = Encoder(repr_dim=repr_dim)

        # Action embedding
        self.action_mlp = nn.Linear(action_dim, repr_dim)

        # BYOL‐style projection heads
        self.predictor_proj = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )

        # Target projection
        self.target_proj = copy.deepcopy(self.predictor_proj)
        for param in self.target_proj.parameters():
            param.requires_grad = False

        '''
        self.rnn = nn.GRU(
            input_size=repr_dim * 2, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        '''
        self.rnn = nn.GRU(
            input_size=repr_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.1
        )
        self.out = nn.Linear(hidden_dim, repr_dim)

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        Tm1 = actions.size(1)

        # Encode
        raw_latents = self.encoder(
            states.view(B * T, C, H, W)
        ).view(B, T, self.repr_dim)  # [B, T, D]

        # Run BYOL‐style projection on the *raw* encoder outputs:
        proj_targets = self.target_proj(raw_latents.view(-1, self.repr_dim))
        proj_targets = proj_targets.view(B, T, self.repr_dim)

        # embed actions
        act_flat = actions.view(-1, self.action_dim)  # [B*(T-1), 2]
        a_emb = self.action_mlp(act_flat)  # [B*(T-1), D]
        a_emb = a_emb.view(B, Tm1, self.repr_dim)  # [B,T-1,D]

        if T > 1:
            # teacher-forcing path (training)
            inp = torch.cat([raw_latents[:, :-1], a_emb], dim=-1)       # [B, T-1, 2D]
            rnn_out, _ = self.rnn(inp)                                  # [B, T-1, H]
            preds = self.out(rnn_out)                                 # [B, T-1, D]
            # preds = rnn_out
            pred_latents = torch.cat([raw_latents[:, :1], preds ], dim=1)
        else:
            # autoregressive generation (probing/eval)
            prev = raw_latents[:, 0]                                    # [B, D]
            hidden = None
            seq = [prev]
            for t in range(Tm1):
                inp = torch.cat([seq[-1], a_emb[:, t]], dim=-1).unsqueeze(1)
                _, hidden = self.rnn(inp, hidden) 
                # out = self.out(hidden.squeeze(0))
                # out = hidden.squeeze(0)
                # out = hidden[-1]
                out = self.out(hidden[-1])
                seq.append(out)
            pred_latents = torch.stack(seq, dim=1)                      # [B, Tm1+1, D]

        # VICReg projection for the predicted latents:
        proj_preds = self.predictor_proj(pred_latents.reshape(-1, self.repr_dim))
        proj_preds = proj_preds.view(B, pred_latents.size(1), self.repr_dim)

        return pred_latents, raw_latents, proj_preds, proj_targets