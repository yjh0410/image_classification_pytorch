import torch
import torch.nn as nn


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='silu',      # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            convs.append(get_norm(norm_type, c1))
            if act_type is not None:
                convs.append(get_activation(act_type))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(inter_dim, out_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ResBlock
class ResBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 nblocks=1,
                 act_type='silu',
                 norm_type='BN'):
        super(ResBlock, self).__init__()
        assert in_dim == out_dim
        self.m = nn.Sequential(*[
            Bottleneck(in_dim, out_dim, expand_ratio=0.5, shortcut=True,
                       norm_type=norm_type, act_type=act_type)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        return self.m(x)


# CSPBlock
class CSPBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 nblocks=1,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(CSPBlock, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv3 = Conv(2 * inter_dim, out_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*[
            Bottleneck(inter_dim, inter_dim, expand_ratio=1.0, shortcut=shortcut,
                       depthwise=depthwise, norm_type=norm_type, act_type=act_type)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out


# DarkNet53
class DarkNet53_SiLU(nn.Module):
    def __init__(self, csp_block=False, act_type='silu', norm_type='BN', num_classes=1000):
        super(DarkNet53_SiLU, self).__init__()
        self.feat_dims = [256, 512, 1024]

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type=act_type, norm_type=norm_type),
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(64, 64, nblocks=1, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(128, 128, nblocks=2, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(256, 256, nblocks=8, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(512, 512, nblocks=8, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(1024, 1024, nblocks=4, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)


    def make_block(self, in_dim, out_dim, nblocks=1, csp_block=False, act_type='silu', norm_type='BN'):
        if csp_block:
            return CSPBlock(in_dim, out_dim, expand_ratio=0.5, nblocks=nblocks,
                            shortcut=True, act_type=act_type, norm_type=norm_type)
        else:
            return ResBlock(in_dim, out_dim, nblocks=nblocks,
                            act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.flatten(1)
        x = self.fc(x)

        return x


# DarkNet53
class DarkNetTiny(nn.Module):
    def __init__(self, csp_block=False, act_type='silu', norm_type='BN', num_classes=1000):
        super(DarkNetTiny, self).__init__()
        self.feat_dims = [256, 512, 1024]

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(16, 16, nblocks=1, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(16, 32, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(32, 32, nblocks=2, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(64, 64, nblocks=8, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(128, 128, nblocks=8, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            self.make_block(256, 256, nblocks=4, csp_block=csp_block, act_type=act_type, norm_type=norm_type)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)


    def make_block(self, in_dim, out_dim, nblocks=1, csp_block=False, act_type='silu', norm_type='BN'):
        if csp_block:
            return CSPBlock(in_dim, out_dim, expand_ratio=0.5, nblocks=nblocks,
                            shortcut=True, act_type=act_type, norm_type=norm_type)
        else:
            return ResBlock(in_dim, out_dim, nblocks=nblocks,
                            act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.flatten(1)
        x = self.fc(x)

        return x


def build_darknet(model_name='darknet53_silu', csp_block=False, pretrained=False): 
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if model_name in ['darknet53_silu', 'cspdarknet53_silu'] and not csp_block:
        model = DarkNet53_SiLU(csp_block, act_type='silu', norm_type='BN')
    elif model_name in ['darknet_tiny', 'cspdarknet_tiny'] and csp_block:
        model = DarkNetTiny(csp_block, act_type='silu', norm_type='BN')

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_darknet(model_name='cspdarknet_tiny', csp_block=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
