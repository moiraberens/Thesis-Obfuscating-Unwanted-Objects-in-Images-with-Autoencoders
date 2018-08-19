import h5py
import os

import gzip
import pickle

import imageio
import numpy as np
import matplotlib.pyplot as plt
import pylab

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def save_hdf5(path, x):
    if len(path.split('/')) > 1 and not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with h5py.File(path, 'w') as f:
        f.create_dataset('x', data=x, compression='gzip', compression_opts=9)

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        x = f['x'][:]
    return x

def save_dataset(path, x_train, y_train_label, x_val,  y_val_label):
    if len(path.split('/')) > 1 and not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with h5py.File(path, 'w') as f:
        print("save x_train")
        f.create_dataset('x_train', data=x_train, compression='gzip', compression_opts=9)
        print("save y_train_label")
        f.create_dataset('y_train_label', data=y_train_label, compression='gzip', compression_opts=9)
        print("save x_val")
        f.create_dataset('x_val', data=x_val, compression='gzip', compression_opts=9)
        print("save y_val_image")
        f.create_dataset('y_val_label', data=y_val_label, compression='gzip', compression_opts=9)

def load_dataset(path):
    with h5py.File(path, 'r') as f:
        x_train = f['x_train'][:]
        y_train_label = f['y_train_label'][:]
        x_val = f['x_val'][:]
        y_val_label = f['y_val_label'][:]
    return x_train, y_train_label, x_val, y_val_label

def save_obj(obj,path):
    with gzip.open(path,'w') as f:
        pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with gzip.open(path,'r') as f:
        obj = pickle.load(f)
    return obj

def save_img(settings, imgs, epoch, save_name):
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img2 = np.round(img * 255)
        img3 = img2.astype('uint8')    
        imageio.imwrite("results/{0}/{1}/epoch{2}_image{3}_{4}.jpg".format(settings['dataset'], settings['model'], epoch, i, save_name), img3)

def save_img_gray(settings, imgs, epoch, save_name):
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img * 255
        img = rgb2gray(img)
        img = img.astype('uint8')    
        img = np.stack((img, img, img), axis=2)
        imageio.imwrite("results/{0}/{1}/epoch{2}_image{3}_{4}.jpg".format(settings['dataset'], settings['model'], epoch, i, save_name), img)
        
# =============================================================================
# def save_epoch_result(folder, epoch, x_img_private, gen_img_private, mask_private, final_img_private, x_img_public, gen_img_public, mask_public, final_img_public):
#     num_images = 2
#     original = np.hstack(np.concatenate((x_img_private[0:num_images], x_img_public[0:num_images])))
#     
#     mask = rgb2gray(np.hstack(np.concatenate((mask_private[0:num_images], mask_public[0:num_images]))))
#     mask = np.stack((mask, mask, mask), axis=2)
#     gen_image = np.hstack(np.concatenate((gen_img_private[0:num_images], gen_img_public[0:num_images])))
#     result = np.hstack(np.concatenate((final_img_private[0:num_images], final_img_public[0:num_images])))
#     combined = np.vstack((original, mask, gen_image, result))
#     combined = combined * 255
#     combined = combined.astype('uint8')
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     imageio.imwrite("{0}/output_epoch{1}.jpg".format(folder, epoch), combined)
# =============================================================================
    
    
def save_epoch_result(folder, epoch, x_img_private, gen_img_private, y_img_private, x_img_public, gen_img_public, y_img_public):
    num_images = 2
    original = np.hstack(np.concatenate((x_img_private[0:num_images], x_img_public[0:num_images])))
    result = np.hstack(np.concatenate((gen_img_private[0:num_images], gen_img_public[0:num_images])))
    correct = np.hstack(np.concatenate((y_img_private[0:num_images], y_img_public[0:num_images])))
    combined = np.vstack((original, result, correct))
    combined = ((combined+1)/2) * 255
    combined = combined.astype('uint8')
    if not os.path.exists(folder):
        os.makedirs(folder)
    imageio.imwrite("{0}/output_epoch{1}.png".format(folder, epoch), combined)
    
def save_plot(folder, metrics, attacker, epoch):
    plt.figure(2)
    plt.clf()
    
    fig, ax = plt.subplots(nrows=4, ncols=2, num=2)
    plt.tight_layout()
    
    ax[0][0].plot(metrics['img_private_train_loss'][:epoch])
    ax[0][0].plot(metrics['img_private_val_loss'][:epoch])
    ax[0][0].set_title('image private loss')
    
    ax[0][1].plot(metrics['img_private_train_loss'][:epoch])
    ax[0][1].plot(metrics['img_private_val_loss'][:epoch])
    ax[0][1].set_title('image private loss')
    ax[0][1].set_ylim(np.min(metrics['img_private_train_loss'][:epoch]),2*np.min(metrics['img_private_train_loss'][:epoch]))
    
    ax[1][0].plot(metrics['img_public_train_loss'][:epoch])
    ax[1][0].plot(metrics['img_public_val_loss'][:epoch])
    ax[1][0].set_title('image public loss')
    
    ax[1][1].plot(metrics['img_public_train_loss'][:epoch])
    ax[1][1].plot(metrics['img_public_val_loss'][:epoch])
    ax[1][1].set_title('image public loss')
    ax[1][1].set_ylim(np.min(metrics['img_public_train_loss'][:epoch]),2*np.min(metrics['img_public_train_loss'][:epoch]))
    
    if attacker == True:
        ax[2][0].plot(metrics['comb_train_loss'][:epoch])
        ax[2][0].plot(metrics['comb_val_loss'][:epoch])
        ax[2][0].set_title('attacker loss in combination')
    
        ax[2][1].plot(metrics['comb_train_acc'][:epoch])
        ax[2][1].plot(metrics['comb_val_acc'][:epoch])
        ax[2][1].set_title('attacker accuracy in combination')
        
        ax[3][0].plot(metrics['att_train_loss'][:epoch])
        ax[3][0].plot(metrics['att_val_loss'][:epoch])
        ax[3][0].set_title('attacker loss')
    
        ax[3][1].plot(metrics['att_train_acc'][:epoch])
        ax[3][1].plot(metrics['att_val_acc'][:epoch])
        ax[3][1].set_title('attacker accuracy')
               
       
    if not os.path.exists(folder):
        os.makedirs(folder)
    pylab.savefig('{0}/plots.png'.format(folder))
    
    plt.pause(0.001)

if __name__ is '__main__':
    pass