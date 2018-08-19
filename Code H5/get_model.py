#load models


from model import unet_GRL_approach3
from model import unet_GRL_approach1_2
from model import unet_prop_blind
from model import unet_prop_semi_blind


def load_model(settings):
    if settings['model'] == 'unet_GRL_approach3':
        model = unet_GRL_approach3.get_models(settings)      
    elif settings['model'] == 'unet_GRL_approach1_2':
        model = unet_GRL_approach1_2.get_models(settings)
    elif settings['model'] == 'unet_prop_blind':
        model = unet_prop_blind.get_models(settings)
    elif settings['model'] == 'unet_prop_semi_blind':
        model = unet_prop_semi_blind.get_models(settings)
    return model

if __name__ is "__main__":
    pass