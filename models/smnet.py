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


# Scale Modulation Block
class SMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='silu', norm_type='BN', depthwise=False):
        super(SMBlock, self).__init__()
        # -------------- Basic parameters --------------
        self.in_dim = in_dim
        self.inter_dim = in_dim // 2
        # -------------- Network parameters --------------
        self.cv1 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        ## Scale Modulation
        self.sm1 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.sm2 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=5, p=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        self.sm3 = nn.Sequential(
            Conv(self.inter_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(self.inter_dim, self.inter_dim, k=7, p=3, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
            )
        ## Aggregation proj
        self.sm_aggregation = Conv(self.inter_dim*3, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)

        # Output proj
        self.out_proj = None
        if in_dim != out_dim:
            self.out_proj = Conv(self.inter_dim*2, out_dim, k=1, act_type=act_type, norm_type=norm_type)


    def channel_shuffle(self, x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        per_group_dim = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, per_group_dim, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C_in, H, W]
        Output:
            out: (Tensor) -> [B, C_out, H, W]
        """
        x1, x2 = torch.chunk(x, 2, dim=1)
        # branch-1
        x1 = self.cv1(x1)
        # branch-2
        x2 = self.cv2(x2)
        x2 = torch.cat([self.sm1(x2), self.sm2(x2), self.sm3(x2)], dim=1)
        x2 = self.sm_aggregation(x2)
        # channel shuffle
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out, groups=2)

        if self.out_proj:
            out = self.out_proj(out)

        return out


# DownSample Block
class DSBlock(nn.Module):
    def __init__(self, in_dim, act_type='silu', norm_type='BN', depthwise=False):
        super().__init__()
        # branch-1
        self.maxpool = nn.MaxPool2d((2, 2), 2)
        # branch-2
        inter_dim = in_dim // 2
        self.sm1 = Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm2 = Conv(inter_dim, inter_dim, k=5, p=2, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.sm3 = Conv(inter_dim, inter_dim, k=7, p=3, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)


    def channel_shuffle(self, x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        per_group_dim = num_channels // groups

        # reshape
        x = x.view(batchsize, groups, per_group_dim, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            out: (Tensor) -> [B, 2C, H/2, W/2]
        """
        x1, x2 = torch.chunk(x, 2, dim=1)
        # branch-1
        x1 = self.maxpool(x1)
        # branch-2
        x2 = torch.cat([self.sm1(x2), self.sm2(x2), self.sm3(x2)], dim=1)
        # channel shuffle
        out = torch.cat([x1, x2], dim=1)
        out = self.channel_shuffle(out, groups=4)

        return out


# Scale-Modulation Network
class ScaleModulationNet(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN', depthwise=False, num_classes=1000):
        super(ScaleModulationNet, self).__init__()
        
        # P1/2
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            Conv(16, 16, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise),
        )

        # P2/4
        self.layer_2 = nn.Sequential(   
            DSBlock(16, act_type, norm_type, depthwise),             
            SMBlock(32, 32, act_type, norm_type, depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            DSBlock(32, act_type, norm_type, depthwise),             
            SMBlock(64, 64, act_type, norm_type, depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            DSBlock(64, act_type, norm_type, depthwise),             
            SMBlock(128, 128, act_type, norm_type, depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            DSBlock(128, act_type, norm_type, depthwise),             
            SMBlock(256, 256, act_type, norm_type, depthwise)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)


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
    if model_name == 'smnet':
        model = ScaleModulationNet(act_type='silu', norm_type='BN', depthwise=True)

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_smnet(model_name='smnet')
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
