import torch
import torch.nn as nn
import torch.nn.functional as F
import os


model_urls = {
    "cspd-s": "https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet53.pth",
}


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)

        return out


class CSPStage(nn.Module):
    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))

        return self.cv3(torch.cat([y1, y2], dim=1))


# CSPDarknet
class CSPDarkNet(nn.Module):
    """
    CSPDarknet_53.
    """
    def __init__(self, width=1.0, depth=1.0, num_classes=1000):
        super(CSPDarkNet, self).__init__()
        # init w&d cfg
        basic_w_cfg = [32, 64, 128, 256, 512, 1024]
        basic_d_cfg = [1, 3, 9, 9, 6]
        # init w&d cfg
        w_cfg = [int(w*width) for w in basic_w_cfg]
        d_cfg = [int(d*depth) for d in basic_d_cfg]
        d_cfg[0] = 1
        print('=================================')
        print('Width: ', w_cfg)
        print('Depth: ', d_cfg)
        print('=================================')

        self.layer_1 = nn.Sequential(
            Conv(3, w_cfg[0], k=3, p=1),      
            Conv(w_cfg[0], w_cfg[1], k=3, p=1, s=2),
            CSPStage(c1=w_cfg[1], n=d_cfg[0])                       # p1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(w_cfg[1], w_cfg[2], k=3, p=1, s=2),             
            CSPStage(c1=w_cfg[2], n=d_cfg[1])                      # P2/4
        )
        self.layer_3 = nn.Sequential(
            Conv(w_cfg[2], w_cfg[3], k=3, p=1, s=2),             
            CSPStage(c1=w_cfg[3], n=d_cfg[2])                      # P3/8
        )
        self.layer_4 = nn.Sequential(
            Conv(w_cfg[3], w_cfg[4], k=3, p=1, s=2),             
            CSPStage(c1=w_cfg[4], n=d_cfg[3])                      # P4/16
        )
        self.layer_5 = nn.Sequential(
            Conv(w_cfg[4], w_cfg[5], k=3, p=1, s=2),             
            CSPStage(c1=w_cfg[5], n=d_cfg[4])                     # P5/32
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_cfg[5], num_classes)


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


def build_cspd(model_name='cspd-l', pretrained=False):
    if model_name == 'cspd-s':
        return cspdarknet_s(pretrained)

    elif model_name == 'cspd-m':
        return cspdarknet_m(pretrained)

    elif model_name == 'cspd-l':
        return cspdarknet_m(pretrained)

    elif model_name == 'cspd-x':
        return cspdarknet_m(pretrained)


def cspdarknet_s(pretrained=False, width=0.5, depth=0.34):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-s']
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


def cspdarknet_m(pretrained=False, width=0.75, depth=0.67):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-m']
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


def cspdarknet_l(pretrained=False, width=1.0, depth=1.0):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-l']
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


def cspdarknet_x(pretrained=False, width=1.25, depth=1.34):
    # model
    model = CSPDarkNet(width=width, depth=depth)

    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        url = model_urls['cspd-x']
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
    net = cspdarknet_x()
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    y = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)