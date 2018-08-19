#Set GPU Memory Allocation to Growth
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import numpy as np
import get_model
from utils import io_utils
import os
import matplotlib.pyplot as plt
import pylab 
import h5py
import random
import glob
from keras.layers import Input
from keras.models import Model
from utils import dataset_utils
import imageio

from tqdm import tqdm
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.utils.vis_utils import plot_model

def get_ImageNet(settings):
    image_paths_horse = glob.glob('ImageNet/Airplanes/*')
    image_paths_airplane = glob.glob('ImageNet/Horses/*')
    x_train_public = np.zeros((0, settings['img_size'], settings['img_size'], settings['num_channels']))
    x_train_private = np.zeros((0, settings['img_size'], settings['img_size'], settings['num_channels']))
    x_val_public = np.zeros((0, settings['img_size'], settings['img_size'], settings['num_channels']))
    x_val_private = np.zeros((0, settings['img_size'], settings['img_size'], settings['num_channels']))
    public_train_image_paths = []
    private_train_image_paths = []
    public_val_image_paths = []
    private_val_image_paths = []
    public_train_image_paths.extend(image_paths_horse[0:480])
    public_train_image_paths.extend(image_paths_airplane[0:480])
    private_train_image_paths.extend(image_paths_horse[480:960])
    private_train_image_paths.extend(image_paths_airplane[480:960])
    public_val_image_paths.extend(image_paths_horse[960:1180])
    public_val_image_paths.extend(image_paths_airplane[960:1180])
    private_val_image_paths.extend(image_paths_horse[1180:1400])
    private_val_image_paths.extend(image_paths_airplane[1180:1400])
    print("Load train images")
    for train_image_path in tqdm(public_train_image_paths):
        image = imageio.imread(train_image_path)
        image = dataset_utils.color_image(image, settings['num_channels'])
        image = dataset_utils.normalize_tahn(image)
        image = dataset_utils.square_image(image, settings['img_size'])
        image = np.expand_dims(image, axis=0)
        x_train_public = np.append(x_train_public, image, axis=0)
    for train_image_path in tqdm(private_train_image_paths):
        image = imageio.imread(train_image_path)
        image = dataset_utils.color_image(image, settings['num_channels'])
        image = dataset_utils.normalize_tahn(image)
        image = dataset_utils.square_image(image, settings['img_size'])
        image = np.expand_dims(image, axis=0)
        x_train_private = np.append(x_train_private, image, axis=0)
    print("Load val images")
    for val_image_path in tqdm(public_val_image_paths):
        image = imageio.imread(val_image_path)
        image = dataset_utils.color_image(image, settings['num_channels'])
        image = dataset_utils.normalize_tahn(image)
        image = dataset_utils.square_image(image, settings['img_size'])
        image = np.expand_dims(image, axis=0)
        x_val_public = np.append(x_val_public, image, axis=0)
    for val_image_path in tqdm(private_val_image_paths):
        image = imageio.imread(val_image_path)
        image = dataset_utils.color_image(image, settings['num_channels'])
        image = dataset_utils.normalize_tahn(image)
        image = dataset_utils.square_image(image, settings['img_size'])
        image = np.expand_dims(image, axis=0)
        x_val_private = np.append(x_val_private, image, axis=0)
    return x_train_public, x_train_private, x_val_public, x_val_private

def create_QR(settings):
    qr = np.ones((settings['obj_size'],settings['obj_size']))
    qr[1:-1, 1:-1] = (np.round(np.random.rand(settings['obj_size']-2,settings['obj_size']-2)) *2) -1
    qr = np.stack((qr, qr, qr), axis=-1).astype(np.float32)
    return qr

def make_private(settings, x):
    y = []
    position_list = []
    for image in x:
        position = np.random.randint(0,  x.shape[1] - settings['obj_size'], 2)
        image[position[0]:position[0] + settings['obj_size'], position[1]:position[1]+settings['obj_size'],:] = create_QR(settings)
        y.append(image)
        position_list.append(position)
    y = np.stack(y)
    return y, position_list

def batch_generator(settings, x, private=False):
    indices = np.arange(x.shape[0])
    while True:
        np.random.shuffle(indices)
        for i in range(int(np.floor(x.shape[0]/float(settings['batch_size'])))):
            batch_ids = indices[i*settings['batch_size']:i*settings['batch_size']+settings['batch_size']]
            flip_list = random.sample(range(0, 32), 16)
            if private:
                x_batch = x[batch_ids]
                x_batch[flip_list, :, :, :] = x_batch[flip_list, :, ::-1, :]
                x_batch, position_list = make_private(settings, x_batch)
                y_batch_image = x[batch_ids]
                y_batch_image[flip_list, :, :, :] = y_batch_image[flip_list, :, ::-1, :]
                y_batch_label = np.ones(settings['batch_size'], dtype=np.int32)
                yield x_batch, y_batch_image, y_batch_label, position_list
            else:
                x_batch = x[batch_ids]
                x_batch[flip_list, :, :, :] = x_batch[flip_list, :, ::-1, :]
                y_batch_label = np.zeros(settings['batch_size'], dtype=np.int32)
                yield x_batch, y_batch_label
                
def calculate_obj_loss(settings, x_batch_private_pred, y_batch_image_private, position_list):
    total = 0
    for index in range(x_batch_private_pred.shape[0]):
        small_image_x = x_batch_private_pred[index, position_list[index][0]:position_list[index][0] + settings['obj_size'], position_list[index][1]:position_list[index][1]+settings['obj_size'], 0:settings['num_channels']]
        small_image_y = y_batch_image_private[index, position_list[index][0]:position_list[index][0] + settings['obj_size'], position_list[index][1]:position_list[index][1]+settings['obj_size'], 0:settings['num_channels']]
        total += np.mean(np.abs(small_image_x - small_image_y))/settings['batch_size']
    return total



def save_plot(folder, metrics, epoch):
    plt.figure(1)
    plt.clf()
    
    fig, ax = plt.subplots(nrows=1, ncols=3, num=1)
    plt.tight_layout()
    
    ax[0].plot(metrics['train_public_loss'][:epoch+1], 'b')
    ax[0].plot(metrics['train_private_loss'][:epoch+1], 'g')
    ax[0].plot(metrics['val_public_loss'][:epoch+1], 'r')
    ax[0].plot(metrics['val_private_loss'][:epoch+1], 'k')
    ax[0].set_title('image loss')
    ax[0].legend(('train_public', 'train_private', 'val_public', 'val_private'))
    
    ax[1].plot(metrics['att_train_public_loss'][:epoch+1], 'b')
    ax[1].plot(metrics['att_train_private_loss'][:epoch+1], 'g')
    ax[1].plot(metrics['att_val_public_loss'][:epoch+1], 'r')
    ax[1].plot(metrics['att_val_private_loss'][:epoch+1], 'k')
    ax[1].set_title('att loss')
    ax[1].legend(('train_public', 'train_private', 'val_public', 'val_private'))
    
    ax[2].plot(metrics['train_obj_loss'][:epoch+1], 'g')
    ax[2].plot(metrics['val_obj_loss'][:epoch+1], 'k')
    ax[2].legend(('train_private', 'val_private'))
    ax[2].set_title('obj loss')
    
    plt.pause(0.001)    
    pylab.savefig('{0}/plots.png'.format(folder))
                
if __name__ is "__main__":
    settings = dict()
    #predefined
    settings['batch_size'] = 32
    settings['half_batch'] = settings['batch_size']//2
    settings['img_size'] = 128
    settings['num_channels'] = 3
    settings['obj_size'] = 40
    settings['epochs'] = 520
    settings['learning_rate_obf'] = 0.00005
    settings['learning_rate_att'] = 0.0001
    settings['final_activation'] = 'tanh'
    settings['activation'] = 'relu'
    settings['loss'] = 'mean_absolute_error'
    #changeble
    settings['batchnormalisation'] = False
    settings['same_batch'] = True
    settings['model'] = 'unet_prop_semi_blind'
    settings['depth'] = 5               #1,2,3,4,5
    settings['initial_filters'] = 32     #8,16,32
    settings['extra_block'] = 'Skip'       #True, False, Skip 
    settings['loss_weight_obf'] = 0.5    # value [0,1] range
    
    obf_weights = [0.4]
    
    for setting in obf_weights:
        settings['loss_weight_obf'] = setting
        for number in range(3):
    
    
            #get data
            x_train_public, x_train_private, x_val_public, x_val_private = get_ImageNet(settings)
            
            #create batch generators
            x_train_public_gen = batch_generator(settings, x_train_public)
            x_train_private_gen = batch_generator(settings, x_train_private, private=True)
            x_val_public_gen = batch_generator(settings, x_val_public)
            x_val_private_gen = batch_generator(settings, x_val_private, private=True)
            
            #determine training and validation steps
            train_steps_per_epoch = np.floor(x_train_private.shape[0]/settings['batch_size']).astype(np.int32)
            val_steps_per_epoch = np.floor(x_val_private.shape[0]/settings['batch_size']).astype(np.int32)
            
        
            #determine loss weights per loss
            obf_loss_weight = K.variable(settings['loss_weight_obf'])
            att_loss_weight = K.variable(1 - settings['loss_weight_obf'])
            
            folder = 'results/propAdd_semiblind_approach1_dataset2/weight_{}_number_{}'.format(settings['loss_weight_obf'], number)
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            obfuscator, obfuscator_part, comb = get_model.load_model(settings)
            plot_model(comb, to_file=folder + '\combination.png', show_shapes=True)
            plot_model(obfuscator, to_file=folder + '\obfuscator.png', show_shapes=True)
            obfuscator.compile(optimizer = Adam(lr=settings['learning_rate_obf']), loss = 'mean_absolute_error')
            comb.compile(optimizer=Adam(lr=settings['learning_rate_att']), loss = ['mean_absolute_error', 'mean_squared_error', 'binary_crossentropy', 'binary_crossentropy'], loss_weights=[obf_loss_weight, obf_loss_weight, att_loss_weight, att_loss_weight], metrics = ['mae'])
            
          
            #create metrics dictionary
            metrics = dict()
            metrics['train_public_loss'] = np.zeros(settings['epochs'])
            metrics['train_private_loss'] = np.zeros(settings['epochs'])
            metrics['val_public_loss'] = np.zeros(settings['epochs'])
            metrics['val_private_loss'] = np.zeros(settings['epochs'])
            metrics['att_train_public_loss'] = np.zeros(settings['epochs'])
            metrics['att_train_private_loss'] = np.zeros(settings['epochs'])
            metrics['att_val_public_loss'] = np.zeros(settings['epochs'])
            metrics['att_val_private_loss'] = np.zeros(settings['epochs'])
            metrics['train_obj_loss'] = np.zeros(settings['epochs'])
            metrics['val_obj_loss'] = np.zeros(settings['epochs'])
            
            mask = np.zeros((settings['batch_size'], settings['img_size'], settings['img_size'],1))
            
            #train the models
            for epoch in tqdm(range(settings['epochs'])):
            
                # TRAIN
                for step in range(train_steps_per_epoch):
                    # Get Batches
                    x_batch_public, y_batch_label_public = next(x_train_public_gen)
                    x_batch_private, y_batch_image_private, y_batch_label_private, position_list = next(x_train_private_gen)
                    
                    #train comb
                    metrics_public = comb.train_on_batch([x_batch_public, x_batch_public], [x_batch_public, mask, y_batch_label_public, y_batch_label_public])
                    metrics_private = comb.train_on_batch([x_batch_private, x_batch_private], [x_batch_private, mask, y_batch_label_private, y_batch_label_private])
                    
                    #metrics['train_public_loss'][epoch] += metrics_public[4]/train_steps_per_epoch #todo
                    #metrics['train_private_loss'][epoch] += metrics_private[4]/train_steps_per_epoch #todo
                    metrics['train_public_loss'][epoch] += obfuscator.test_on_batch(x_batch_public, x_batch_public)/train_steps_per_epoch
                    metrics['train_private_loss'][epoch] += obfuscator.test_on_batch(x_batch_private, x_batch_private)/train_steps_per_epoch
                    
                    metrics['att_train_public_loss'][epoch] += (metrics_public[3] + metrics_public[4])/(train_steps_per_epoch * 2)
                    metrics['att_train_private_loss'][epoch] += (metrics_private[3] + metrics_public[4])/(train_steps_per_epoch * 2)
                    
                    
                    
                    private_pred = obfuscator.predict(x_batch_private)
                    metrics['train_obj_loss'][epoch] += calculate_obj_loss(settings, private_pred, y_batch_image_private, position_list)/train_steps_per_epoch
                    
                # VAL
                for step in range(val_steps_per_epoch):
                    # Get Batches
                    x_batch_public, y_batch_label_public = next(x_val_public_gen)
                    x_batch_private, y_batch_image_private, y_batch_label_private, position_list = next(x_val_private_gen)
                    
                    metrics_public = comb.test_on_batch([x_batch_public, x_batch_public], [x_batch_public, mask, y_batch_label_public, y_batch_label_public])
                    metrics_private = comb.test_on_batch([x_batch_private, x_batch_private], [y_batch_image_private, mask, y_batch_label_private, y_batch_label_private])
                    
                    metrics['val_public_loss'][epoch] += obfuscator.test_on_batch(x_batch_public, x_batch_public)/val_steps_per_epoch
                    metrics['val_private_loss'][epoch] += obfuscator.test_on_batch(x_batch_private, y_batch_image_private)/val_steps_per_epoch
                    
                    metrics['att_val_public_loss'][epoch] += (metrics_public[3] + metrics_public[4])/(val_steps_per_epoch * 2)
                    metrics['att_val_private_loss'][epoch] += (metrics_private[3] + metrics_public[4])/(val_steps_per_epoch * 2)
                    
                    private_pred = obfuscator.predict(x_batch_private)
                    metrics['val_obj_loss'][epoch] += calculate_obj_loss(settings, private_pred, y_batch_image_private, position_list)/val_steps_per_epoch
                
                #save some images per epoch
                gen_public_images = obfuscator.predict(x_batch_public, batch_size=1) 
                gen_private_images = obfuscator.predict(x_batch_private, batch_size=1)
                
                #save some images per epoch
                inter_public, mask_public = obfuscator_part.predict(x_batch_public, batch_size=1) 
                inter_private, mask_private = obfuscator_part.predict(x_batch_private, batch_size=1) 
                
                #save image per epoch
                io_utils.save_epoch_result_mask(folder, epoch, x_batch_private, mask_private, inter_private, gen_private_images, y_batch_image_private, x_batch_public, mask_public, inter_public, gen_public_images, x_batch_public)
                
                
                #save plot per epoch
                save_plot(folder, metrics, epoch)
                
            #print metric results    
            print('val_public_loss {0:.4f}, val_private_loss {1:.4f}, val_obj_loss {2:.4f}'.format(
                    metrics['val_public_loss'][epoch],
                    metrics['val_private_loss'][epoch],
                    metrics['val_obj_loss'][epoch]))  
            
            #save metric restuls
            h5f = h5py.File('{0}/metrics_epoch_{1}.h5'.format(folder, epoch), 'w')
            for key in metrics:
                h5f.create_dataset(key, data=metrics[key])
            h5f.close()
    
