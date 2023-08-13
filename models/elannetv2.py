import torch
import torch.nn as nn
from typing import List


# ---------------------------- Base Modules ----------------------------
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

## Basic conv layer
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

## YOLO-style BottleNeck
class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_sizes :List[int] = [3, 3],
                 expand_ratio :float     = 0.5,
                 shortcut     :bool      = False,
                 act_type     :str       = 'silu',
                 norm_type    :str       = 'BN',
                 depthwise    :bool      = False):
        super(YoloBottleneck, self).__init__()
        # ------------------ Basic parameters ------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = int(out_dim * expand_ratio)
        self.shortcut = shortcut and in_dim == out_dim
        # ------------------ Network parameters ------------------
        self.cv1 = Conv(in_dim, self.inter_dim, k=kernel_sizes[0], p=kernel_sizes[0]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.cv2 = Conv(self.inter_dim, out_dim, k=kernel_sizes[1], p=kernel_sizes[1]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

## ELAN Block for Backbone
class ELANBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio :float=0.5, branch_depth :int=1, shortcut=False, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # ----------- Basic Parameters -----------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = round(in_dim * expand_ratio)
        self.expand_ratio = expand_ratio
        self.branch_depth = branch_depth
        self.shortcut = shortcut
        # ----------- Network Parameters -----------
        ## branch-1
        self.cv1 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-2
        self.cv2 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## branch-3
        self.cv3 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## branch-4
        self.cv4 = nn.Sequential(*[
            YoloBottleneck(self.inter_dim, self.inter_dim, [1, 3], 1.0, shortcut, act_type, norm_type, depthwise)
            for _ in range(branch_depth)
        ])
        ## output proj
        self.out = Conv(self.inter_dim*4, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)

        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))

        return out

## DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        inter_dim = out_dim // 2
        self.branch_1 = nn.Sequential(
            nn.MaxPool2d((2, 2), 2),
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        )
        self.branch_2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = torch.cat([x1, x2], dim=1)

        return out


# ---------------------------- Backbones ----------------------------
## Modified ELANNet-v2
class ELANNetv2(nn.Module):
    def __init__(self, width=1.0, depth=1.0, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(ELANNetv2, self).__init__()
        # ------------------ Basic parameters ------------------
        ## scale factor
        self.width = width
        self.depth = depth
        self.expand_ratio = [0.5, 0.5, 0.5, 0.25]
        self.branch_depths = [round(dep * depth) for dep in [3, 3, 3, 3]]
        ## pyramid feats
        self.feat_dims = [round(dim * width) for dim in [64, 128, 256, 512, 1024, 1024]]
        ## nonlinear
        self.act_type = act_type
        self.norm_type = norm_type
        self.depthwise = depthwise
        
        # ------------------ Network parameters ------------------
        ## P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, self.feat_dims[0], k=6, p=2, s=2, act_type=self.act_type, norm_type=self.norm_type),
            Conv(self.feat_dims[0], self.feat_dims[0], k=3, p=1, act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
        )
        ## P2/4
        self.layer_2 = nn.Sequential(   
            DSBlock(self.feat_dims[0], self.feat_dims[1], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            ELANBlock(self.feat_dims[1], self.feat_dims[2], self.expand_ratio[0], self.branch_depths[0], True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(self.feat_dims[2], self.feat_dims[2], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            ELANBlock(self.feat_dims[2], self.feat_dims[3], self.expand_ratio[1], self.branch_depths[1], True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(self.feat_dims[3], self.feat_dims[3], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            ELANBlock(self.feat_dims[3], self.feat_dims[4], self.expand_ratio[2], self.branch_depths[2], True, self.act_type, self.norm_type, self.depthwise)
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(self.feat_dims[4], self.feat_dims[4], act_type=self.act_type, norm_type=self.norm_type, depthwise=self.depthwise),
            ELANBlock(self.feat_dims[4], self.feat_dims[5], self.expand_ratio[3], self.branch_depths[3], True, self.act_type, self.norm_type, self.depthwise)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024*width), num_classes)


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


# build ELANNet-v2
def build_elannetv2(model_name='elannet_v2_small', pretrained=False): 
    # P5 model
    if model_name == 'elannet_v2_huge':
        model = ELANNetv2(width=1.25, depth=1.34, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_large':
        model = ELANNetv2(width=1.0, depth=1.0, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_medium':
        model = ELANNetv2(width=0.75, depth=0.67, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_small':
        model = ELANNetv2(width=0.5, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_tiny':
        model = ELANNetv2(width=0.375, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_nano':
        model = ELANNetv2(width=0.25, depth=0.34, act_type='silu', norm_type='BN')
    elif model_name == 'elannet_v2_pico':
        model = ELANNetv2(width=0.25, depth=0.34, act_type='silu', norm_type='BN', depthwise=True)

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_elannetv2(model_name='elannet_v2_large')
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    x = torch.randn(1, 3, 256, 256)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
