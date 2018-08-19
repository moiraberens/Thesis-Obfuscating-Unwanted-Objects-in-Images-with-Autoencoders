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

from tqdm import tqdm
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

def get_cifar10():
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    x_train = ((x_train/255).astype(np.float32) *2)-1
    x_val = ((x_val/255).astype(np.float32) * 2)-1
    
    x_train_public = []
    x_train_private = []
    x_val_public = []
    x_val_private = []
    
    for i in range(10):
        train_images = x_train[y_train.flatten()==i]
        x_train_public.append(train_images[:2500])
        x_train_private.append(train_images[2500:])
        
        val_images = x_val[y_val.flatten()==i]
        x_val_public.append(val_images[:500])
        x_val_private.append(val_images[500:])
    
    x_train_public = np.concatenate((x_train_public))
    x_train_private = np.concatenate((x_train_private))
    x_val_public = np.concatenate((x_val_public))
    x_val_private = np.concatenate((x_val_private))
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
    
    fig, ax = plt.subplots(nrows=1, ncols=2, num=1)
    plt.tight_layout()
    
    ax[0].plot(metrics['train_public_loss'][:epoch+1], 'b')
    ax[0].plot(metrics['train_private_loss'][:epoch+1], 'g')
    ax[0].plot(metrics['val_public_loss'][:epoch+1], 'r')
    ax[0].plot(metrics['val_private_loss'][:epoch+1], 'k')
    ax[0].set_title('image loss')
    ax[0].legend(('train_public', 'train_private', 'val_public', 'val_private'))
    
    ax[1].plot(metrics['train_obj_loss'][:epoch+1], 'g')
    ax[1].plot(metrics['val_obj_loss'][:epoch+1], 'k')
    ax[1].legend(('train_private', 'val_private'))
    ax[1].set_title('obj loss')
    
    plt.pause(0.001)    
    pylab.savefig('{0}/plots.png'.format(folder))
                
if __name__ is "__main__":
    settings = dict()
    #predefined
    settings['batch_size'] = 32
    settings['half_batch'] = settings['batch_size']//2
    settings['img_size'] = 32
    settings['num_channels'] = 3
    settings['obj_size'] = 10
    settings['epochs'] = 100
    settings['learning_rate'] = 0.001
    settings['final_activation'] = 'tanh'
    settings['activation'] = 'relu'
    settings['loss'] = 'mean_absolute_error'
    #changeble
    settings['batchnormalisation'] = True
    settings['same_batch'] = True
    settings['model'] = 'conv_deconv'
    settings['depth'] = 3               #1,2,3,4,5
    settings['initial_filters'] = 32     #8,16,32
    settings['extra_block'] = True       #True, False, Skip 
    
    #get data
    x_train_public, x_train_private, x_val_public, x_val_private = get_cifar10()
    
    #create batch generators
    x_train_public_gen = batch_generator(settings, x_train_public)
    x_train_private_gen = batch_generator(settings, x_train_private, private=True)
    x_val_public_gen = batch_generator(settings, x_val_public)
    x_val_private_gen = batch_generator(settings, x_val_private, private=True)
    
    #create setting list
    batchnorm = [True, False]
    #same_batch = [True, False]
    depth = [1,2,3]
    initial_filters = [8,16,32]
    extra_block = [True, False, 'skip']
    num_models = len(batchnorm) * len(depth) * len(initial_filters) * len(extra_block) 
    
    train_steps_per_epoch = np.floor(x_train_private.shape[0]/settings['batch_size']).astype(np.int32)
    val_steps_per_epoch = np.floor(x_val_private.shape[0]/settings['batch_size']).astype(np.int32)
    
    
    #train models
    current_model = 0
    for depth_setting in depth:
        settings['depth'] = depth_setting
        for initial_filter_setting in initial_filters:
            settings['initial_filters'] = initial_filter_setting
            for extra_block_setting in extra_block:
                settings['extra_block'] = extra_block_setting
                for batchnorm_setting in batchnorm:
                    settings['batchnormalisation'] = batchnorm_setting
                    current_model += 1
                    for i in range(3):
                    
                        folder = 'results/dataset1/MODEL_{0}_DEPTH_{1}_INITIALFILTER_{2}_EXTRABLOCK_{3}_BATCHNORM_{4}_num_{5}'.format(settings['model'], 
                                          str(settings['depth']), str(settings['initial_filters']), str(settings['extra_block']), str(settings['batchnormalisation']), str(i))
                        
                        obfuscator = get_model.load_model(settings)
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        plot_model(obfuscator, to_file=folder + '\model.png', show_shapes=True)
                        obfuscator.compile(optimizer = Adam(lr=settings['learning_rate']), loss = 'mean_absolute_error')
                        
                        #create metrics dictionary
                        metrics = dict()
                        metrics['train_public_loss'] = np.zeros(settings['epochs'])
                        metrics['train_private_loss'] = np.zeros(settings['epochs'])
                        metrics['val_public_loss'] = np.zeros(settings['epochs'])
                        metrics['val_private_loss'] = np.zeros(settings['epochs'])
                        metrics['train_obj_loss'] = np.zeros(settings['epochs'])
                        metrics['val_obj_loss'] = np.zeros(settings['epochs'])
                        
                        #train the models
                        print ('training model {0} from the {1}'.format(current_model, num_models))
                        print('MODEL: {0}_DEPTH_{1}_INITIALFILTER_{2}_EXTRABLOCK_{3}_BATCHNORM_{4}_SAMEBATCH_{5}'.format(settings['model'], 
                                          str(settings['depth']), str(settings['initial_filters']), str(settings['extra_block']), str(settings['batchnormalisation']), str(settings['same_batch'])))
                        for epoch in tqdm(range(settings['epochs'])):
                            # TRAIN
                            for step in range(train_steps_per_epoch):
                                # Get Batches
                                x_batch_public, y_batch_label_public = next(x_train_public_gen)
                                x_batch_private, y_batch_image_private, y_batch_label_private, position_list = next(x_train_private_gen)
                                
                                # TRAIN
                                if settings['same_batch'] == True:
                                    x_batch_1 = np.concatenate((x_batch_public[:settings['half_batch']], x_batch_private[:settings['half_batch']]))
                                    y_batch_1 = np.concatenate((x_batch_public[:settings['half_batch']], y_batch_image_private[:settings['half_batch']]))
                                    x_batch_2 = np.concatenate((x_batch_public[-settings['half_batch']:], x_batch_private[-settings['half_batch']:]))
                                    y_batch_2 = np.concatenate((x_batch_public[-settings['half_batch']:], y_batch_image_private[-settings['half_batch']:]))
                                    obfuscator.train_on_batch(x_batch_1, y_batch_1)
                                    obfuscator.train_on_batch(x_batch_2, y_batch_2)
                                    
                                    metrics['train_public_loss'][epoch] += obfuscator.test_on_batch(x_batch_public, x_batch_public)/train_steps_per_epoch
                                    metrics['train_private_loss'][epoch] += obfuscator.test_on_batch(x_batch_private, y_batch_image_private)/train_steps_per_epoch
                                    private_pred = obfuscator.predict(x_batch_private)
                                    metrics['train_obj_loss'][epoch] += calculate_obj_loss(settings, private_pred, y_batch_image_private, position_list)/train_steps_per_epoch
                                    
                                    
                                    
                                else:
                                    metrics['train_public_loss'][epoch] += obfuscator.train_on_batch(x_batch_public, x_batch_public)/train_steps_per_epoch
                                    metrics['train_private_loss'][epoch] += obfuscator.train_on_batch(x_batch_private, y_batch_image_private)/train_steps_per_epoch
                                    private_pred = obfuscator.predict(x_batch_private)
                                    metrics['train_obj_loss'][epoch] += calculate_obj_loss(settings, private_pred, y_batch_image_private, position_list)/train_steps_per_epoch
                                
                            # VAL
                            for step in range(val_steps_per_epoch):
                                # Get Batches
                                x_batch_public, y_batch_label_public = next(x_val_public_gen)
                                x_batch_private, y_batch_image_private, y_batch_label_private, position_list = next(x_val_private_gen)
                                
                                # val
                                metrics['val_public_loss'][epoch] += obfuscator.test_on_batch(x_batch_public, x_batch_public)/val_steps_per_epoch
                                metrics['val_private_loss'][epoch] += obfuscator.test_on_batch(x_batch_private, y_batch_image_private)/val_steps_per_epoch
                                private_pred = obfuscator.predict(x_batch_private)
                                metrics['val_obj_loss'][epoch] += calculate_obj_loss(settings, private_pred, y_batch_image_private, position_list)/val_steps_per_epoch
                            
                            #save some images per epoch
                            gen_public_images = obfuscator.predict(x_batch_public, batch_size=1) 
                            gen_private_images = obfuscator.predict(x_batch_private, batch_size=1)
                            
                            
                            #save image per epoch
                            io_utils.save_epoch_result(folder, epoch, x_batch_private, gen_private_images, y_batch_image_private, x_batch_public, gen_public_images, x_batch_public)
                            
                            
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