from keras.models import Model
from keras.layers import Input, Lambda, concatenate
from utils import model_utils

def get_models(settings):
    inputs = Input((settings['img_size'], settings['img_size'],settings['num_channels']))
    
    #get obfuscator
    if settings['extra_block'] == 'skip':
        conv, layer_list, num_filters = model_utils.residual_encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        image, mask = model_utils.residual_decoder_block_unet_prop(settings, conv, layer_list, num_filters)
    else:
        conv, layer_list, num_filters = model_utils.encoder_block_unet(settings, inputs, settings['initial_filters'], settings['depth'])
        image, mask = model_utils.decoder_block_unet_prop(settings, conv, layer_list, num_filters)
    
    #trek layers uit elkaar
    #mask = Lambda(lambda x : x[:,:,:,0:1])(deconv)
    #image = Lambda(lambda x : x[:,:,:,1:4])(deconv)   
    
    #mask2 = Concatenate([mask, mask, mask])
    #mask_real = Lambda(lambda x : x[:,:,:,:])(mask2)
    
    mask_real = concatenate([mask, mask, mask])
    
    
    output = Lambda(model_utils.PropAdd, output_shape=(settings['img_size'], settings['img_size'], settings['num_channels']))([inputs, image, mask_real])
    
    #gradient reverse layer
    #grl = model_utils.GradientReversal(1.0)(output)
    
    #attacker
    #attacker_layer = model_utils.get_attacker(settings, grl, 8, settings['depth'])
    
    #comb = Model(inputs=inputs, outputs=attacker_layer)
    obfuscator_whole = Model(inputs=inputs, outputs=output)
    obfuscator_part = Model(inputs=inputs, outputs=[mask, image])
    
    #model_mask = Model(inputs=inputs, outputs=mask)
    #model_image = Model(inputs=inputs, outputs=image)
    
    #obfuscator_whole.summary()
    #plot_model(obfuscator, to_file='unet_inpaint.png', show_shapes=True)
    
    return obfuscator_whole, obfuscator_part 
    


if __name__ == '__main__':
    pass
