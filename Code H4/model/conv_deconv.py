from keras.models import Model
from keras.layers import Input
from utils import model_utils

def get_models(settings):
    inputs = Input((settings['img_size'], settings['img_size'],settings['num_channels']))
    
    
    
    #obfuscator
    if settings['extra_block'] == 'skip':
        conv, num_filter = model_utils.residual_encoder_block(settings, inputs,  settings['initial_filters'], settings['depth'])
        deconv = model_utils.residual_decoder_block(settings, conv, num_filter, 3, settings['depth']) 
    else:
        conv, num_filter = model_utils.encoder_block(settings, inputs,  settings['initial_filters'], settings['depth'])
        deconv = model_utils.decoder_block(settings, conv, num_filter, 3, settings['depth']) 
    
    #create model
    obfuscator = Model(inputs=inputs, outputs=deconv)
    
    #obfuscator.summary()

    
    return obfuscator
    
if __name__ == '__main__':
    pass