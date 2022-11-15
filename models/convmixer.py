import torch
import torch.nn as nn


model_urls = {
    "convmixer_base": None,
}

class Mixer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Spatial Mixer
        self.spatial_convs = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=out_dim, bias=False)
        self.spatial_norm = nn.BatchNorm2d(in_dim)
        self.spatial_act = nn.SiLU(inplace=True)

        # Channel Mixer
        self.channel_convs = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.channel_norm = nn.BatchNorm2d(out_dim)
        self.channel_act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = x + self.spatial_act(self.spatial_norm(self.spatial_convs(x)))
        return self.channel_act(self.channel_norm(self.channel_convs(x)))


class ConvMixer(nn.Module):
    def __init__(self, dim, depth=20, stride=16, num_classes=1000):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=stride, stride=stride, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        mixers = [Mixer(dim, dim) for _ in range(depth)]
        self.mixers = nn.Sequential(*mixers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.mixers(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.flatten(1)
        x = self.fc(x)

        return x
            

# build elannet
def build_convmixer(model_name='convmixer_base', pretrained=False):
    # config
    model_size = model_name[-5:]
    model_dim = 768
    model_depth = 24

    # ConvMixer
    model = ConvMixer(model_dim, model_depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['convmixer_{}'.format(model_size)]
        # check
        if url is None:
            print('No pretrained weight for backbone ...')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print(k)

            model.load_state_dict(checkpoint_state_dict)

    return model


if __name__ == '__main__':
    import time
    net = build_convmixer(model_name='convmixer_base', pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)