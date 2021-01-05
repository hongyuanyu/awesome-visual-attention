import math
import torch
import torch.nn as nn


def get_ld_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width=224, height=224, channel=64, fidx_u=16, fidx_v=16):
    dct_weights = torch.zeros(1, channel, width, height)

    # split channel for multi-spectral attention
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                val = get_ld_dct(t_x, u_x, width) * get_ld_dct(t_y, v_y, height)
                dct_weights[:, i * c_part: (i+1) * c_part, t_x, t_y] = val

    return dct_weights


class FcaLayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FcaLayer, self).__init__()
        self.register_buffer("precomputed_dct_weights", get_dct_weights())
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,_,_ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        y = self.fc(y).view(n,c,1,1)
        return x * y.expand_as(x)
