import torch
import torch.nn as nn


# ---------------------------- 2D CNN ----------------------------
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
    elif act_type is None:
        return nn.Identity()

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)

class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='lrelu',     # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        p = p if d == 1 else d
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


# ---------------------------- Core Modules ----------------------------
## Multi-head Mixed Conv (MHMC)
class MultiHeadMixedConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        # -------------- Network parameters --------------
        ## Scale Modulation
        self.mixed_convs = nn.ModuleList([
            Conv(self.head_dim, self.head_dim, k=2*i+1, p=i, act_type=None, norm_type=None, depthwise=depthwise)
            for i in range(num_heads)])

    def forward(self, x):
        xs = torch.chunk(x, self.num_heads, dim=1)
        ys = [mixed_conv(x_h) for x_h, mixed_conv in zip(xs, self.mixed_convs)]
        ys = torch.cat(ys, dim=1)

        return ys

## Scale-aware Aggregator (SAA)
class ScaleAwareAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        assert in_dim == out_dim
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_groups = in_dim // num_heads
        # -------------- Network parameters --------------
        ## Aggregation 1x1 Conv
        self.aggr_conv = nn.ModuleList()
        for _ in range(self.num_groups):
            self.aggr_conv.append(nn.Conv2d(num_heads, num_heads, kernel_size=1))
        ## Out-proj
        self.out_proj = Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def channel_shuffle(self, x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


    def forward(self, x):
        # channel shuffle
        x = self.channel_shuffle(x, groups=self.num_heads)
        # aggregation conv
        xs = torch.chunk(x, self.num_groups, dim=1)
        xs = [conv(xh) for xh, conv in zip(xs, self.aggr_conv)]
        xs = torch.cat(xs, dim=1)
        # out-proj
        ys = self.out_proj(xs)

        return ys

## Scale Modulation Module (SAM)
class ScaleAwareModule(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.inter_dim = in_dim // 2
        self.shortcut = shortcut
        # -------------- Network parameters --------------
        ## In-proj
        self.cv1 = Conv(self.in_dim, self.in_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.in_dim, self.in_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## MHMC & SAA
        self.mhmc = MultiHeadMixedConv(self.in_dim, self.in_dim, self.num_heads, depthwise)
        self.saa = ScaleAwareAggregator(self.in_dim, self.in_dim, self.num_heads, act_type, norm_type, depthwise)


    def forward(self, x):
        # branch-1
        x1 = self.cv1(x)
        # branch-2
        x2 = self.cv2(x)
        x2 = self.mhmc(x2)
        x2 = self.saa(x2)
        # output
        out = x1 * x2

        return out + x if self.shortcut else out


# ---------------------------- Base Blocks ----------------------------
## Scale Modulation Block
class SMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, nblocks=1, num_heads=4, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nblocks = nblocks
        self.num_heads = num_heads
        self.shortcut = shortcut
        self.inter_dim = in_dim // 2
        # -------------- Network parameters --------------
        ## branch-1
        self.cv1 = Conv(self.in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.smblocks = nn.Sequential(*[
            ScaleAwareModule(self.inter_dim, self.inter_dim, self.num_heads, self.shortcut, act_type, norm_type, depthwise)
            for _ in range(nblocks)])
        ## out proj
        self.out_proj = Conv(self.inter_dim*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def forward(self, x):
        # branch-1
        x1 = self.cv1(x)
        # branch-2
        x2 = self.smblocks(self.cv2(x))
        # output
        out = torch.cat([x1, x2], dim=1)
        out = self.out_proj(out)

        return out

## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = out_dim // 2
        self.num_heads = num_heads
        # branch-1
        self.maxpool = nn.Sequential(
            Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            nn.MaxPool2d((2, 2), 2)
        )
        # branch-2
        self.ds_conv = nn.Sequential(
            Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        ) 


    def forward(self, x):
        # branch-1
        x1 = self.maxpool(x)
        # branch-2
        x2 = self.ds_conv(x)
        # out-proj
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- Scale-Modulation Network ----------------------------
class ScaleModulationNet(nn.Module):
    def __init__(self, width=1.0, depth=1.0, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(ScaleModulationNet, self).__init__()
        # ------------------ Basic parameters ------------------
        self.base_dims = [64, 128, 256, 512, 1024]
        self.base_nblocks = [3, 6, 6, 3]
        self.feat_dims = [round(dim * width) for dim in self.base_dims]
        self.nblocks = [round(nblock * depth) for nblock in self.base_nblocks]
        self.shortcut = True
        self.num_heads = 4
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # ------------------ Network parameters ------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=3, p=1, s=2, act_type=self.act_type, norm_type=self.norm_type),
            Conv(self.feat_dims[0], self.feat_dims[0], k=3, p=1, act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
        )
        ## P2/4
        self.layer_2 = nn.Sequential(   
            DSBlock(self.feat_dims[0], self.feat_dims[1], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            SMBlock(self.feat_dims[1], self.feat_dims[1], self.nblocks[0], self.num_heads, self.shortcut, self.act_type, self.norm_type, self.depthwise)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(self.feat_dims[1], self.feat_dims[2], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            SMBlock(self.feat_dims[2], self.feat_dims[2], self.nblocks[1], self.num_heads, self.shortcut, self.act_type, self.norm_type, self.depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(self.feat_dims[2], self.feat_dims[3], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            SMBlock(self.feat_dims[3], self.feat_dims[3], self.nblocks[2], self.num_heads, self.shortcut, self.act_type, self.norm_type, self.depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(self.feat_dims[3], self.feat_dims[4], self.num_heads, self.act_type, self.norm_type, self.depthwise),             
            SMBlock(self.feat_dims[4], self.feat_dims[4], self.nblocks[3], self.num_heads, self.shortcut, self.act_type, self.norm_type, self.depthwise)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims[4], num_classes)


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


# build ELAN-Net
def build_smnet(model_name='smnet', pretrained=False): 
    if model_name == 'smnet_huge':
        model = ScaleModulationNet(width=1.25, depth=1.34, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_large':
        model = ScaleModulationNet(width=1.0, depth=1.0, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_medium':
        model = ScaleModulationNet(width=0.75, depth=0.67, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_small':
        model = ScaleModulationNet(width=0.5, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_tiny':
        model = ScaleModulationNet(width=0.375, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_nano':
        model = ScaleModulationNet(width=0.25, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'smnet_pico':
        model = ScaleModulationNet(width=0.25, depth=0.34, act_type='silu', norm_type='BN', depthwise=True)

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_smnet(model_name='smnet_pico')
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
