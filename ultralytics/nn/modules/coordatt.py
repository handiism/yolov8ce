import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

class DynamicCoordAtt(nn.Module):
    def __init__(self, oup, reduction=32):
        super().__init__()
        self.coordatt = None
        self.oup = oup
        self.reduction = reduction

    def forward(self, x):
        if self.coordatt is None:
            inp = x.shape[1]
            self.coordatt = CoordAtt(inp, self.oup, self.reduction).to(x.device)
        return self.coordatt(x)
