from .resnet import build_resnet


def build_model(model_name='resnet18', pretrained=False, norm_type='BN'):
    if 'resnet' in model_name:
        model = build_resnet(model_name=model_name, 
                             pretrained=pretrained, 
                             norm_type=norm_type)

    return model
