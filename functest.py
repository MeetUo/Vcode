import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import my_model as model

image_file = []
path = "./image"
for f in listdir(path):
    image_file.append(join(path, f))
data_x = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_file))).astype(
        np.float32)

with tf.Session() as sess:
    conv = tf.placeholder('float32', shape=[None, 25, 100, 3], name="X");
    y_data = model.get_my_model(conv)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    saver.restore(sess,'./model/vcode-model.ckpt-done')
    result = sess.run(y_data,feed_dict={conv:data_x})

for idex,item in enumerate(result):
    ans = np.argmax(item.reshape((4,36)),axis=1);
    string = "";
    for i in ans:
        if (i<10):
            string = string + str(i)
        else :
            string =string + chr(i+ord('a'))
    plt.imshow(mpimg.imread(image_file[idex]))
    plt.axis('off')  # 不显示坐标轴
    plt.title(string)
    plt.show()
'''
plt.imshow(mpimg.imread(image_file[0]))
plt.axis('off') # 不显示坐标轴
plt.show()
'''