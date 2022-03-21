from torch import nn
from net.moco import MoCo


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


class CBDE(nn.Module):
    def __init__(self, opt):
        super(CBDE, self).__init__()

        dim = 256

        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=opt.batch_size * dim)

    def forward(self, x_query, x_key):
        if self.training:
            # degradation-aware represenetion learning
            fea, logits, labels, inter = self.E(x_query, x_key)

            return fea, logits, labels, inter
        else:
            # degradation-aware represenetion learning
            fea, inter = self.E(x_query, x_query)
            return fea, inter
