import torch

from .darknet19 import build_darknet19
from .darknet53 import build_darknet53
from .darknet import build_darknet
from .elannet import build_elannet
from .elannetv2 import build_elannetv2
from .elan_cspnet import build_elan_cspnet
from .cspdarknet import build_cspdarknet
from .convmixer import build_convmixer
from .mcnet import build_mcnet


def build_model(model_name='resnet18',
                pretrained=False,
                num_classes=1000,
                resume=None):
    if model_name in ['elannet_pico',  'elannet_nano',   'elannet_tiny',
                      'elannet_small', 'elannet_medium',
                      'elannet_large', 'elannet_huge',
                      'elannet_p6_large', 'elannet_p6_huge',
                      'elannet_p7_large', 'elannet_p7_huge']:
        model = build_elannet(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name in ['elannet_v2_pico', 'elannet_v2_nano', 'elannet_v2_tiny', 'elannet_v2_small', 'elannet_v2_medium', 'elannet_v2_large', 'elannet_v2_huge']:
        model = build_elannetv2(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name in ['elan_cspnet_nano',  'elan_cspnet_small', 'elan_cspnet_medium',
                        'elan_cspnet_large', 'elan_cspnet_huge']:
        model = build_elan_cspnet(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name in ['cspdarknet_nano',  'cspdarknet_small', 'cspdarknet_medium',
                        'cspdarknet_large', 'cspdarknet_huge']:
        model = build_cspdarknet(
            model_name=model_name,
            pretrained=pretrained
        )

    elif model_name == 'darknet19':
        model = build_darknet19(pretrained=pretrained)

    elif model_name == 'darknet53':
        model = build_darknet53(pretrained=pretrained)

    elif model_name in ['darknet53_silu', 'darknet_tiny']:
        model = build_darknet(model_name, csp_block=False, pretrained=pretrained)

    elif model_name in ['cspdarknet53_silu', 'cspdarknet_tiny']:
        model = build_darknet(model_name, csp_block=True, pretrained=pretrained)

    elif model_name in ['convmixer_nano',  'convmixer_small', 'convmixer_medium',
                        'convmixer_large', 'convmixer_huge']:
        model = build_convmixer(
            model_name=model_name,
            pretrained=pretrained
        )

    if model_name in ['mcnet_pico', 'mcnet_nano', 'mcnet_tiny', 'mcnet_small', 'mcnet_medium', 'mcnet_large', 'mcnet_huge']:
        model = build_mcnet(model_name, pretrained)

        
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
