import torch
import torch.nn as nn


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'gelu':
        return nn.GELU()
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
            assert c1 == c2
            if s == 1:
                convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding='same', dilation=d, groups=c1, bias=False))
            else:
                convs.append(nn.Conv2d(c2, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=c1, bias=False))
            if act_type is not None:
                convs.append(get_activation(act_type))
            convs.append(get_norm(norm_type, c2))

        else:
            # normal group conv
            if s == 1:
                convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding='same', dilation=d, groups=1, bias=False))
            else:
                convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=1, bias=False))
            if act_type is not None:
                convs.append(get_activation(act_type))
            convs.append(get_norm(norm_type, c2))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# Patch-Embed Layer
class PatchEmbedLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 patch_size=16,
                 act_type='relu',
                 norm_type='BN'):
        super(PatchEmbedLayer, self).__init__()
        self.patch_embed = Conv(in_dim, out_dim, k=patch_size, s=patch_size, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.patch_embed(x)

        return x


# Conv-Mixer Block
class ConvMixerBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel_size=9,
                 act_type='relu',
                 norm_type='BN'):
        super(ConvMixerBlock, self).__init__()
        assert in_dim == out_dim
        self.spatial_mixer = Conv(in_dim, out_dim, k=kernel_size, depthwise=True, act_type=act_type, norm_type=norm_type)
        self.channel_mixer = Conv(out_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = x + self.spatial_mixer(x)
        x = self.channel_mixer(x)

        return x


# ConvMixer
class ConvMixer(nn.Module):
    def __init__(self, patch_size=16, kernel_size=9, nblocks=20, d_model=1024, act_type='relu', norm_type='BN', num_classes=1000):
        super(ConvMixer, self).__init__()
        self.nblocks = nblocks
        self.feat_dim = d_model

        # Patch-Embed Layer
        self.patch_embed = PatchEmbedLayer(3, d_model, patch_size, act_type, norm_type)
        
        # Conv-Mixer Blocks
        self.cm_blocks = nn.Sequential(
            *[ConvMixerBlock(d_model, d_model, kernel_size, act_type, norm_type)
              for _ in range(nblocks)]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(d_model, num_classes)


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.cm_blocks(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.flatten(1)
        x = self.fc(x)

        return x


def build_convmixer(model_name='convmixer_large', pretrained=False): 
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if model_name == 'convmixer_huge':
        model = ConvMixer(patch_size=16, kernel_size=9, nblocks=36, d_model=960, act_type='silu', norm_type='BN')
    elif model_name == 'convmixer_large':
        model = ConvMixer(patch_size=16, kernel_size=9, nblocks=32, d_model=768, act_type='silu', norm_type='BN')
    elif model_name == 'convmixer_medium':
        model = ConvMixer(patch_size=16, kernel_size=9, nblocks=28, d_model=512, act_type='silu', norm_type='BN')
    elif model_name == 'convmixer_small':
        model = ConvMixer(patch_size=16, kernel_size=9, nblocks=24, d_model=384, act_type='silu', norm_type='BN')
    elif model_name == 'convmixer_nano':
        model = ConvMixer(patch_size=16, kernel_size=9, nblocks=20, d_model=256, act_type='silu', norm_type='BN')

    return model


if __name__ == '__main__':
    import time
    from thop import profile
    model = build_convmixer(model_name='convmixer_nano')
    x = torch.randn(1, 3, 640, 640)
    t0 = time.time()
    y = model(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    print('==============================')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))
