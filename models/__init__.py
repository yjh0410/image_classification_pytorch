import torch

from .resnet import build_resnet
from .darknet19 import build_darknet19
from .darknet53 import build_darknet53
from .cspdarknet import build_cspd
from .elannet import build_elannet
from .convmixer import build_convmixer


def build_model(model_name='resnet18',
                pretrained=False,
                num_classes=1000,
                resume=None):
    if 'resnet' in model_name:
        model = build_resnet(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes
            )

    elif model_name in ['cspd-n', 'cspd-t', 'cspd-s', 'cspd-m', 'cspd-l', 'cspd-x']:
        model = build_cspd(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name in ['convmixer_base', 'convmixer_huge', 'convmixer_tiny']:
        model = build_convmixer(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name == 'darknet19':
        model = build_darknet19(pretrained=pretrained)

    elif model_name == 'darknet53':
        model = build_darknet53(pretrained=pretrained)

    elif model_name == 'convmixer':
        model = build_darknet53(pretrained=pretrained)

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
