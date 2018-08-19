from keras import backend as K
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization, Add, concatenate, LeakyReLU, Flatten, Dense
import numpy as np
import tensorflow as tf
from keras.engine import Layer

# Implementation of gradient reversal layer
# Adapted from: Michele Tonutti  (https://github.com/michetonu/gradient_reversal_keras_tf)              
def reverse_gradient(X, hp_lambda):
    grad_name = 'GradientReversal {}'.format(np.random.random())
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
        return y

# Implementation of gradient reversal layer
# Adapted from: Michele Tonutti  (https://github.com/michetonu/gradient_reversal_keras_tf)
class GradientReversal(Layer):
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = K.variable(K.cast_to_floatx(hp_lambda))
    def build(self, input_shape):
        self.trainable_weights = []
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        return {'name': self.__class__.__name__,
                'hp_lambda': float(K.cast_to_floatx(self.hp_lambda.eval()))}

# code for attacker        
def attacker_block(settings, layer, start_filters, depth):
    num_filters = start_filters
    for i in range(depth):
        layer = Conv2D(num_filters, 3, strides = 2, padding = 'same')(layer)
        if settings['batchnormalisation'] == True:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU()(layer)
        num_filters *= 2
    return layer

def get_attacker(settings, layer, start_filters, depth):
        
    att_block = attacker_block(settings, layer, start_filters, depth)
    
    flat = Flatten()(att_block)

    dens = Dense(1, activation='sigmoid')(flat)

    return dens

# code for propAdd layer
def PropAdd(x):
    return x[0] * (K.abs(1-x[2])) + x[1] * x[2]


# code for convolution block with skip connection
def residual_conv_block(settings, layer_in, filters):
    layer_out = Conv2D(filters, 3, padding='same')(layer_in)
    if settings['batchnormalisation'] == True:
        layer_out = BatchNormalization()(layer_out)
    layer_out = Activation(settings['activation'])(layer_out)
    layer_out = Add()([layer_out, layer_in])
    #layer_out = BatchNormalization()(layer_out)
    layer_out = Activation(settings['activation'])(layer_out)
    return layer_out

# code for convolution block without skip connection
def conv_block(settings, layer_in, filters, strides=1, kernel_size=(3,3)):
    layer_out = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_in)
    if settings['batchnormalisation'] == True:
        layer_out = BatchNormalization()(layer_out)
    layer_out = Activation(settings['activation'])(layer_out)
    return layer_out

# code for deconv block 
def deconv_block(settings, layer, num_filters):
    layer = Conv2DTranspose(num_filters, 3, strides=(2, 2), padding='same')(layer)
    if settings['batchnormalisation'] == True:
        layer = BatchNormalization()(layer)
    layer = Activation(settings['activation'])(layer)
    return layer

# code for deconv block with a sigmoid
def final_deconv_block(settings, layer, filters):
    layer = Conv2DTranspose(filters, 3, strides=(2, 2), padding='same')(layer)
    layer = Activation('tanh')(layer)
    return layer

# code for deconv block with a sigmoid
def final_prop_block(settings, layer):
    layer1 = Conv2DTranspose(3, 3, strides=(2, 2), padding='same')(layer)
    layer1 = Activation('tanh')(layer1)
    layer2 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same')(layer)
    layer2 = Activation('sigmoid')(layer2)
    return layer1, layer2

# code for encoder and decoder block for the conv_deconv models
def encoder_block(settings, layer, num_filters, depth):
    for i in range(depth):
        layer = conv_block(settings, layer, num_filters, 2)
        if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
        num_filters *= 2
    return layer, num_filters//2

def residual_encoder_block(settings, layer, num_filters, depth):
    for i in range(depth):
        layer = conv_block(settings, layer, num_filters, 2)
        layer = residual_conv_block(settings, layer, num_filters)
        num_filters *= 2
    return layer, num_filters//2

def decoder_block(settings, layer, num_filters, final_filter, depth):
    #create interpolation and identity layer
    for i in range(depth-1):
        num_filters//=2
        layer = deconv_block(settings, layer, num_filters)
        if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
    layer = final_deconv_block(settings, layer, final_filter)
    return layer

def residual_decoder_block(settings, layer, num_filters, final_filter, depth):
    #create interpolation and identity layer
    for i in range(depth-1):
        num_filters//=2
        layer = deconv_block(settings, layer, num_filters)
        layer = residual_conv_block(settings, layer, num_filters)
    layer = final_deconv_block(settings, layer, final_filter)
    return layer


# code for encoder decoder block for unet models
def encoder_block_unet(settings, layer, num_filters, depth):
    layer_list = [] 
    for i in range(depth-1):
        layer = conv_block(settings, layer, num_filters, 2)
        if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
        num_filters *= 2
        layer_list.append(layer)
    layer = conv_block(settings, layer, num_filters, 2)
    if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
    return layer, layer_list, num_filters

def residual_encoder_block_unet(settings, layer, num_filters, depth):
    layer_list = [] 
    for i in range(depth-1):
        layer = conv_block(settings, layer, num_filters, 2)
        layer = residual_conv_block(settings, layer, num_filters)
        num_filters *= 2
        layer_list.append(layer)
    layer = conv_block(settings, layer, num_filters, 2)
    layer = residual_conv_block(settings, layer, num_filters)
    return layer, layer_list, num_filters

def decoder_block_unet(settings, layer, layer_list, num_filters, final_filter):
    for conv_layer in reversed(layer_list) :
        num_filters //= 2    
        layer = deconv_block(settings, layer, num_filters)
        layer = concatenate([layer, conv_layer])
        layer = conv_block(settings, layer, num_filters, kernel_size=(1,1)) # Downsample Features
        if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
    layer = final_deconv_block(settings, layer, final_filter)
    return layer

def residual_decoder_block_unet(settings, layer, layer_list, num_filters, final_filter):
    for conv_layer in reversed(layer_list) :
        num_filters //= 2    
        layer = deconv_block(settings, layer, num_filters)
        layer = concatenate([layer, conv_layer])
        layer = conv_block(settings, layer, num_filters, kernel_size=(1,1)) # Downsample Features
        layer = residual_conv_block(settings, layer, num_filters)
    layer = final_deconv_block(settings, layer, final_filter)
    return layer


def decoder_block_unet_prop(settings, layer, layer_list, num_filters):
    for conv_layer in reversed(layer_list) :
        num_filters //= 2    
        layer = deconv_block(settings, layer, num_filters)
        layer = concatenate([layer, conv_layer])
        layer = conv_block(settings, layer, num_filters, kernel_size=(1,1)) # Downsample Features
        if settings['extra_block'] == True:
            layer = conv_block(settings, layer, num_filters)
    layer1, layer2 = final_prop_block(settings, layer)
    return layer1, layer2

def residual_decoder_block_unet_prop(settings, layer, layer_list, num_filters):
    for conv_layer in reversed(layer_list) :
        num_filters //= 2    
        layer = deconv_block(settings, layer, num_filters)
        layer = concatenate([layer, conv_layer])
        layer = conv_block(settings, layer, num_filters, kernel_size=(1,1)) # Downsample Features
        layer = residual_conv_block(settings, layer, num_filters)
    layer1, layer2 = final_prop_block(settings, layer)
    return layer1, layer2


