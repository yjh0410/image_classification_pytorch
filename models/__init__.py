import torch

from .darknet19 import build_darknet19
from .darknet import build_darknet
from .elannetv2 import build_elannetv2
from .cspdarknet import build_cspdarknet
from .rtcnet import build_rtcnet


def build_model(args):
    if args.model in ['elannet_v2_pico', 'elannet_v2_nano', 'elannet_v2_tiny', 'elannet_v2_small', 'elannet_v2_medium', 'elannet_v2_large', 'elannet_v2_huge']:
        model = build_elannetv2(args.model)

    elif args.model in ['cspdarknet_nano',  'cspdarknet_small', 'cspdarknet_medium', 'cspdarknet_large', 'cspdarknet_huge']:
        model = build_cspdarknet(args.model)

    elif args.model in ['darknet19']:
        model = build_darknet19()

    elif args.model in ['darknet53_silu', 'darknet_tiny']:
        model = build_darknet(args.model, csp_block=False)

    elif args.model in ['cspdarknet53_silu', 'cspdarknet_tiny']:
        model = build_darknet(args.model, csp_block=True)

    elif args.model in ['rtcnet_p', 'rtcnet_n', 'rtcnet_s', 'rtcnet_m', 'rtcnet_l', 'rtcnet_x']:
        model = build_rtcnet(args.model)

    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))
        
    if args.resume and args.resume != "None":
        print('keep training: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
        del checkpoint, checkpoint_state_dict
      
    return model
