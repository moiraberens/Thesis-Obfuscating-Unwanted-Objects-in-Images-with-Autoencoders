from keras.models import Model
from utils import model_utils
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization, Add, concatenate, LeakyReLU, Flatten, Dense, Lambda

# code for attacker        
def attacker_block(settings, layer_A, layer_B, start_filters, depth):
    num_filters = start_filters
    
    for i in range(depth):
        layer = Conv2D(num_filters, 3, strides = 2, padding = 'same')
        layer_A = layer(layer_A)
        layer_B = layer(layer_B)
        act = LeakyReLU()
        layer_A = act(layer_A)
        layer_B = act(layer_B)
        num_filters *= 2
    return layer_A, layer_B

def get_attacker(settings, layer_A, layer_B, start_filters, depth):
        
    layer_A, layer_B = attacker_block(settings, layer_A, layer_B, start_filters, depth)
    
    flat = Flatten()
    layer_A = flat(layer_A)
    layer_B = flat(layer_B)

    dens = Dense(1, activation='sigmoid')
    layer_A = dens(layer_A)
    layer_B = dens(layer_B)

    return layer_A, layer_B

def get_models(settings):
    inputs = Input((settings['img_size'], settings['img_size'],settings['num_channels']))
    
    if settings['extra_block'] == 'skip':
        conv, layer_list, num_filters = model_utils.residual_encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        image, mask = model_utils.residual_decoder_block_unet_prop(settings, conv, layer_list, num_filters)
    else:
        conv, layer_list, num_filters = model_utils.encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        image, mask = model_utils.decoder_block_unet_prop(settings, conv, layer_list, num_filters)
    
    mask_real = concatenate([mask, mask, mask])
    deconv = Lambda(model_utils.PropAdd, output_shape=(settings['img_size'], settings['img_size'], settings['num_channels']))([inputs, image, mask_real])
    
    grl = model_utils.GradientReversal(1.0)(deconv)
    
    #
    extra_inputs = Input((settings['img_size'], settings['img_size'],settings['num_channels']))
        
    attacker_A, attacker_B = get_attacker(settings, grl, extra_inputs, 64, 4)
    
    obfuscator = Model(inputs=inputs, outputs=deconv)
    obfuscator_part = Model(inputs=inputs, outputs=[image, mask])
    comb = Model(inputs=[inputs, extra_inputs], outputs = [image, mask, attacker_A, attacker_B])
    #obfuscator.summary()
    
    return obfuscator, obfuscator_part, comb
    

if __name__ == '__main__':
    pass