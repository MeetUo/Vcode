import  tensorflow as tf

def get_my_model(conv):
    MODEL_LAYERS = [
        'conv1','maxpool1','conv2','maxpool1','conv3','maxpool2','conv4','maxpool3'

    ]
    model_arg =[
        (3,32,3,1),(2,2),(32,64,3,1),(2,2),(64,128,3,2),(2,1),(128,144,3,2),(2,2)
    ]

    def instance_norm(x):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    def relu(layer):
        return tf.nn.relu(layer)

    def max_pool(layer,ksize,stride):
        return tf.nn.max_pool(layer,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding='SAME')

    def conv2d(x,input_filter,output_filter, kernal, strides,scale = 1):
        with tf.variable_scope('conv2d'):
            shape=[kernal,kernal*scale,input_filter,output_filter]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d(x,filter = weight,strides=[1,strides,strides*scale,1],
                                padding='SAME',name='conv')

    def reslayer(x,filter,kernel,strides):
        with tf.variable_scope('resnet'):
            conv1 = conv2d(x, filter, filter, kernel, strides)
            conv2 = conv2d(relu(conv1), filter, filter, kernel, strides)
            residual = x + conv2
            return residual

    net_con = [];
    for name,arg_num in zip(MODEL_LAYERS,model_arg):
        if name.startswith('c'):
            with tf.variable_scope(name):
                if (name=='conv1'):
                    conv = relu(instance_norm(conv2d(conv,arg_num[0],arg_num[1],arg_num[2],arg_num[3],scale=4)))
                else:
                    conv = relu(instance_norm(conv2d(conv, arg_num[0], arg_num[1], arg_num[2], arg_num[3], scale=1)))
        elif name.startswith('r'):
            with tf.variable_scope(name):
                conv = relu(instance_norm(reslayer(conv,arg_num[0],arg_num[1],arg_num[2])))
        elif name.startswith('m'):
            with tf.variable_scope(name):
                conv = max_pool(conv,arg_num[0],arg_num[1])
        elif name.startswith('d'):
            with tf.variable_scope(name):
                conv = tf.nn.dropout(conv,keep_prob=0.75);
        net_con.append(tf.shape(conv))

    #return net_con
    conv = tf.contrib.layers.flatten(conv)
    #conv = tf.contrib.layers.fully_connected(conv,1024)
    #conv = tf.contrib.layers.fully_connected(conv, 512)
    conv = tf.contrib.layers.fully_connected(conv, 144, activation_fn=None)
    return conv