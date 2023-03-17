import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv


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


# ---------------------------- Basic Modules ----------------------------
## Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='',          # activation
                 norm_type='',         # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


## ConvBlocks
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 kernel=[1, 3],
                 shortcut=False,
                 act_type='silu',
                 norm_type='BN',
                 depthwise=False):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=kernel[0], p=kernel[0]//2,
                        norm_type=norm_type, act_type=act_type,
                        depthwise=False if kernel[0] == 1 else depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=kernel[1], p=kernel[1]//2,
                        norm_type=norm_type, act_type=act_type,
                        depthwise=False if kernel[1] == 1 else depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


## CSP-stage block
class CSPBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 kernel=[1, 3],
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
            Bottleneck(inter_dim, inter_dim, expand_ratio=1.0, kernel=kernel, shortcut=shortcut,
                       norm_type=norm_type, act_type=act_type, depthwise=depthwise)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x1)
        out = self.cv3(torch.cat([x3, x2], dim=1))

        return out
    

## Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5, act_type='', norm_type=''):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# ---------------------------- CSPDarkNet ----------------------------
# CSPDarkNet
class CSPDarkNet(nn.Module):
    def __init__(self, depth=1.0, width=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(CSPDarkNet, self).__init__()
        self.feat_dims = [int(224*width), int(512*width), int(1024*width)]

        # P1
        self.layer_1 = Conv(3, int(64*width), k=6, p=2, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        
        # P2
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(128*width), int(128*width), expand_ratio=0.5, nblocks=int(3*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P3
        self.layer_3 = nn.Sequential(
            Conv(int(128*width), int(224*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(224*width), int(224*width), expand_ratio=0.5, nblocks=int(9*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P4
        self.layer_4 = nn.Sequential(
            Conv(int(224*width), int(512*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            CSPBlock(int(512*width), int(512*width), expand_ratio=0.5, nblocks=int(9*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # P5
        self.layer_5 = nn.Sequential(
            Conv(int(512*width), int(1024*width), k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            SPPF(int(1024*width), int(1024*width), expand_ratio=0.5, act_type=act_type, norm_type=norm_type),
            CSPBlock(int(1024*width), int(1024*width), expand_ratio=0.5, nblocks=int(3*depth),
                     shortcut=True, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
## build ELAN-Net
def build_cspdarknet(model_name='cspdarknet_large', pretrained=False): 
    # P5 model
    if model_name == 'cspdarknet_huge':
        model = CSPDarkNet(width=1.25, depth=1.34, act_type='silu', norm_type='BN')
    elif model_name == 'cspdarknet_large':
        model = CSPDarkNet(width=1.0, depth=1.0, act_type='silu', norm_type='BN')
    elif model_name == 'cspdarknet_medium':
        model = CSPDarkNet(width=0.75, depth=0.67, act_type='silu', norm_type='BN')
    elif model_name == 'cspdarknet_small':
        model = CSPDarkNet(width=0.5, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'cspdarknet_nano':
        model = CSPDarkNet(width=0.25, depth=0.34, act_type='lrelu', norm_type='BN')

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_cspdarknet(model_name='cspdarknet_nano')
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
