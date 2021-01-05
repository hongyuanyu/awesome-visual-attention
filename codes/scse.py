import torch
from torch import nn
class sSELayer(nn.Module):
    def __init__(self, in_channels):
        super(sSELayer, self).__init__()
        self.Conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv(U) 
        q = self.Sigmoid(q) #[bs, 1 ,h ,w]
        return U * q 

class cSELayer(nn.Module):
    def __init__(self, in_channels):
        super(cSELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) #[bs, c/2, 1, 1]
        z = self.Conv_Excitation(z) #[bs, c, 1, 1]
        z = self.Sigmoid(z)
        return U * z.expand_as(U)

class scSELayer(nn.Module):
    def __init__(self, in_channels):
        super(scSELayer, self).__init__()
        self.cSE = cSELayer(in_channels)
        self.sSE = sSELayer(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse
