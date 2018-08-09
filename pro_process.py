import tensorflow as tf
from os import listdir
from os.path import join
from scipy import ndimage
import numpy as np

class CONFIG:
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1, 1, 3))

def reshape_and_normalize_image(image):

    image = image-CONFIG.MEANS

    return image

def num(c):
    if c>='0' and c<='9':
        return ord(c)-ord('0')
    else:
        return ord(c)-ord('a')

def get_train(path):
    data_y=[]
    image_file = []
    for f in listdir(path):
        y = f.split('.')[0].lower()
        data = np.zeros(144)
        data[num(y[0])] = 1
        data[num(y[1])+36] = 1
        data[num(y[2])+72] = 1
        data[num(y[3])+108] = 1
        data_y.append(data)
        image_file.append(join(path,f))
    data_x = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_file))).astype(
        np.float32)
    #data_x = reshape_and_normalize_image(data_x)
    return data_x,np.array(data_y)




