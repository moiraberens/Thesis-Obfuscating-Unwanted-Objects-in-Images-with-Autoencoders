#load models

from model import conv_deconv
from model import unet
from model import unet_propAdd

def load_model(settings):
    if settings['model'] == 'conv_deconv':
        model = conv_deconv.get_models(settings)
    elif settings['model'] == 'unet':
        model = unet.get_models(settings)
    elif settings['model'] == 'unet_propAdd':
        model = unet_propAdd.get_models(settings)
    return model

if __name__ is "__main__":
    pass