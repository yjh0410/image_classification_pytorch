import torch

from .resnet import build_resnet
from .cspdarknet import build_cspd


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

    elif 'cspd' in model_name:
        model = build_cspd(
            model_name=model_name,
            pretrained=pretrained
        )

    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)
                        
    return model
