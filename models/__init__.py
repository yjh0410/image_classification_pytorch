import torch

from .darknet19 import build_darknet19
from .darknet import build_darknet
from .elannetv2 import build_elannetv2
from .cspdarknet import build_cspdarknet
from .rtcnet import build_rtcnet


def build_model(model_name='resnet18', resume=None):
    if model_name in ['elannet_v2_pico', 'elannet_v2_nano', 'elannet_v2_tiny', 'elannet_v2_small', 'elannet_v2_medium', 'elannet_v2_large', 'elannet_v2_huge']:
        model = build_elannetv2(model_name)

    elif model_name in ['cspdarknet_nano',  'cspdarknet_small', 'cspdarknet_medium', 'cspdarknet_large', 'cspdarknet_huge']:
        model = build_cspdarknet(model_name)

    elif model_name in ['darknet19']:
        model = build_darknet19()

    elif model_name in ['darknet53_silu', 'darknet_tiny']:
        model = build_darknet(model_name, csp_block=False)

    elif model_name in ['cspdarknet53_silu', 'cspdarknet_tiny']:
        model = build_darknet(model_name, csp_block=True)

    elif model_name in ['rtcnet_p', 'rtcnet_n', 'rtcnet_t', 'rtcnet_s', 'rtcnet_m', 'rtcnet_l', 'rtcnet_x']:
        model = build_rtcnet(model_name)

    else:
        raise NotImplementedError("Unknown model: {}".format(model_name))
        
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
