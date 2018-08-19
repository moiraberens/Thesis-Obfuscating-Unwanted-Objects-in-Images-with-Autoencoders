import numpy as np
from skimage.transform import resize

def normalize_tahn(images):
    images = images.astype('float32')
    images = images/127.5
    images = images-1
    return images

def normalize_image(image):
    image = image.astype('float32')
    image = image/255
    return image

def square_image(image, size):
    """
    Preprocess images such that they become size by size with a reshape 
    """
    if image.shape[0] != image.shape[1]:
        min_size = min(image.shape[0], image.shape[1])
        start_y = (image.shape[0] - min_size)//2
        start_x = (image.shape[1] - min_size)//2
        image = image[start_y:start_y + min_size, start_x:start_x + min_size]
    if image.shape[0] != size:
        image = image.astype('float32')
        image = resize(image, (size, size))
    return image

def color_image(image, num_channels):
    if len(image.shape) != num_channels:
        image = np.stack([image, image, image], axis=2)
        if max(image.flatten()) <= 1:
            image = image * 255
    return image
        
if __name__ is '__main__':
    pass