from keras.models import Model
from keras.layers import Input
from utils import model_utils


def get_models(settings):
    inputs = Input((settings['img_size'], settings['img_size'],settings['num_channels']))
    
    #get obfuscator
    if settings['extra_block'] == 'skip':
        conv, layer_list, num_filters = model_utils.residual_encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        deconv = model_utils.residual_decoder_block_unet(settings, conv, layer_list, num_filters, 3)
    else:
        conv, layer_list, num_filters = model_utils.encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        deconv = model_utils.decoder_block_unet(settings, conv, layer_list, num_filters, 3)
    
    grl = model_utils.GradientReversal(1.0)(deconv)
    
    attacker_layer = model_utils.get_attacker(settings, grl, 64, 4)
    
    obfuscator = Model(inputs=inputs, outputs=deconv)
    comb = Model(inputs=inputs, outputs = [deconv, attacker_layer])
    #obfuscator.summary()
    
    return obfuscator, comb
    

if __name__ == '__main__':
    pass