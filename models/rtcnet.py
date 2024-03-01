import torch
import torch.nn as nn
from   typing import List


# --------------------- Basic modules ---------------------
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
    else:
        raise NotImplementedError
        
def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

class BasicConv(nn.Module):
    def __init__(self, 
                 in_dim,                   # in channels
                 out_dim,                  # out channels 
                 kernel_size=1,            # kernel size 
                 padding=0,                # padding
                 stride=1,                 # padding
                 dilation=1,               # dilation
                 act_type  :str = 'lrelu', # activation
                 norm_type :str = 'BN',    # normalization
                 depthwise :bool = False
                ):
        super(BasicConv, self).__init__()
        self.depthwise = depthwise
        if not depthwise:
            self.conv = get_conv2d(in_dim, out_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=1)
            self.norm = get_norm(norm_type, out_dim)
        else:
            self.conv1 = get_conv2d(in_dim, in_dim, k=kernel_size, p=padding, s=stride, d=dilation, g=in_dim)
            self.norm1 = get_norm(norm_type, in_dim)
            self.conv2 = get_conv2d(in_dim, out_dim, k=1, p=0, s=1, d=1, g=1)
            self.norm2 = get_norm(norm_type, out_dim)
        self.act  = get_activation(act_type)

    def forward(self, x):
        if not self.depthwise:
            return self.act(self.norm(self.conv(x)))
        else:
            # Depthwise conv
            x = self.norm1(self.conv1(x))
            # Pointwise conv
            x = self.norm2(self.conv2(x))
            return x


# --------------------- Yolov8 modules ---------------------
class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim       :int,
                 out_dim      :int,
                 kernel_size  :List  = [1, 3],
                 expand_ratio :float = 0.5,
                 shortcut     :bool  = False,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,
                 ) -> None:
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        # ----------------- Network setting -----------------
        self.conv_layer1 = BasicConv(in_dim, inter_dim,
                                     kernel_size=kernel_size[0], padding=kernel_size[0]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv_layer2 = BasicConv(inter_dim, out_dim,
                                     kernel_size=kernel_size[1], padding=kernel_size[1]//2, stride=1,
                                     act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.conv_layer2(self.conv_layer1(x))

        return x + h if self.shortcut else h

class ELANLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio :float = 0.5,
                 num_blocks   :int   = 1,
                 shortcut     :bool  = False,
                 act_type     :str   = 'silu',
                 norm_type    :str   = 'BN',
                 depthwise    :bool  = False,
                 ) -> None:
        super(ELANLayer, self).__init__()
        self.inter_dim = round(out_dim * expand_ratio)
        self.input_proj  = BasicConv(in_dim, self.inter_dim * 2, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.output_proj = BasicConv((2 + num_blocks) * self.inter_dim, out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.module = nn.ModuleList([YoloBottleneck(self.inter_dim,
                                                    self.inter_dim,
                                                    kernel_size  = [3, 3],
                                                    expand_ratio = 1.0,
                                                    shortcut     = shortcut,
                                                    act_type     = act_type,
                                                    norm_type    = norm_type,
                                                    depthwise    = depthwise)
                                                    for _ in range(num_blocks)])

    def forward(self, x):
        # Input proj
        x1, x2 = torch.chunk(self.input_proj(x), 2, dim=1)
        out = list([x1, x2])

        # Bottlenecl
        out.extend(m(out[-1]) for m in self.module)

        # Output proj
        out = self.output_proj(torch.cat(out, dim=1))

        return out
 
   
# --------------------- RTCNet ---------------------
class RTCBackbone(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0, num_classes=1000, act_type='silu', norm_type='BN', depthwise=False):
        super(RTCBackbone, self).__init__()
        # ---------------- Basic parameters ----------------
        self.width_factor = width
        self.depth_factor = depth
        self.last_stage_factor = ratio
        self.use_pixel_statistic = False
        self.feat_dims = [round(64 * width),
                          round(128 * width),
                          round(256 * width),
                          round(512 * width),
                          round(512 * width * ratio)
                          ]
        # ---------------- Network parameters ----------------
        ## P1/2
        self.layer_1 = BasicConv(3, self.feat_dims[0],
                                 kernel_size=3, padding=1, stride=2,
                                 act_type=act_type, norm_type=norm_type)
        ## P2/4
        self.layer_2 = nn.Sequential(
            BasicConv(self.feat_dims[0], self.feat_dims[1],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[1],
                      out_dim    = self.feat_dims[1],
                      num_blocks = round(3*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P3/8
        self.layer_3 = nn.Sequential(
            BasicConv(self.feat_dims[1], self.feat_dims[2],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[2],
                      out_dim    = self.feat_dims[2],
                      num_blocks = round(6*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P4/16
        self.layer_4 = nn.Sequential(
            BasicConv(self.feat_dims[2], self.feat_dims[3],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[3],
                      out_dim    = self.feat_dims[3],
                      num_blocks = round(6*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        ## P5/32
        self.layer_5 = nn.Sequential(
            BasicConv(self.feat_dims[3], self.feat_dims[4],
                      kernel_size=3, padding=1, stride=2,
                      act_type=act_type, norm_type=norm_type, depthwise=depthwise),
            ELANLayer(in_dim     = self.feat_dims[4],
                      out_dim    = self.feat_dims[4],
                      num_blocks = round(3*depth),
                      shortcut   = True,
                      act_type   = act_type,
                      norm_type  = norm_type,
                      depthwise  = depthwise,
                      )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feat_dims[4], num_classes)


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        c5 = self.avgpool(c5)
        c5 = torch.flatten(c5, 1)
        c5 = self.fc(c5)

        return c5


# ------------------------ Model Functions ------------------------
def rtcnet_p(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=0.25,
                       depth=0.34,
                       ratio=2.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=True,
                       num_classes=num_classes
                       )

def rtcnet_n(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=0.25,
                       depth=0.34,
                       ratio=2.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def rtcnet_s(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=0.50,
                       depth=0.34,
                       ratio=2.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def rtcnet_m(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=0.75,
                       depth=0.67,
                       ratio=1.5,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def rtcnet_l(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=1.0,
                       depth=1.0,
                       ratio=1.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def rtcnet_x(num_classes=1000) -> RTCBackbone:
    return RTCBackbone(width=1.25,
                       depth=1.34,
                       ratio=1.0,
                       act_type='silu',
                       norm_type='BN',
                       depthwise=False,
                       num_classes=num_classes
                       )

def build_rtcnet(model_name):
    # build vit model
    if   model_name == 'rtcnet_p':
        model = rtcnet_p()
    elif model_name == 'rtcnet_n':
        model = rtcnet_n()
    elif model_name == 'rtcnet_s':
        model = rtcnet_s()
    elif model_name == 'rtcnet_m':
        model = rtcnet_m()
    elif model_name == 'rtcnet_l':
        model = rtcnet_l()
    elif model_name == 'rtcnet_x':
        model = rtcnet_x()
    
    return model


if __name__ == '__main__':
    import torch
    from thop import profile

    # build model
    model = rtcnet_p()

    x = torch.randn(1, 3, 224, 224)
    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
