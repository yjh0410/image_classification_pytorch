import torch

from .darknet19 import build_darknet19
from .darknet53 import build_darknet53
from .darknet import build_darknet53_silu
from .elannet import build_elannet


def build_model(model_name='resnet18',
                pretrained=False,
                num_classes=1000,
                resume=None):
    if model_name in ['elannet_nano',  'elannet_tiny',
                      'elannet_small', 'elannet_medium',
                      'elannet_large', 'elannet_huge']:
        model = build_elannet(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name == 'darknet19':
        model = build_darknet19(pretrained=pretrained)

    elif model_name == 'darknet53':
        model = build_darknet53(pretrained=pretrained)

    elif model_name == 'darknet53_silu':
        model = build_darknet53_silu(csp_block=False, pretrained=pretrained)

    elif model_name == 'cspdarknet53_silu':
        model = build_darknet53_silu(csp_block=True, pretrained=pretrained)

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
