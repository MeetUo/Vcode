import my_model as model0
import model_test as model1
from pro_process import *

X,Y = get_train("E:\IdeaProjects\produceVcode\image");
test_X,test_Y = get_train("E://IdeaProjects//produceVcode//newcode");
print("数据加载成功")
test_num = 100;
train_num = 12000;

random_test = np.random.permutation(test_Y.shape[0])
true_test_index = random_test[0:test_num]
true_train_index = random_test[test_num:]

random = np.random.permutation(Y.shape[0])
train_index = random[0:train_num]
val_index = random[train_num:]

train_x = X[train_index]
train_y = Y[train_index]
val_x = X[val_index]
val_y = Y[val_index]


test_train_x = test_X[true_train_index][0:800]
test_train_y = test_Y[true_train_index][0:800]
test_train_x_1 = test_X[true_train_index]
test_train_y_1 = test_Y[true_train_index]
test_x = test_X[true_test_index]
test_y = test_Y[true_test_index]

learning_rate = 1e-3
epochs = 20
batch_size = 512
n_iters_per_epoch = int(np.ceil(float(train_num) / batch_size))
num_iter = int(np.ceil(float(test_train_x.shape[0])/ batch_size))
num_iter_test = int(np.ceil(float(test_train_x_1.shape[0])/ batch_size))

with tf.Session() as sess:
    conv = tf.placeholder('float32', shape=[None, 25, 100, 3], name="X");
    y_data = model0.get_my_model(conv)

    label_y = tf.placeholder(dtype=tf.float32, shape=[None, 144], name="Y")
    loss = 0;
    #for i in range(4):
    #   loss+=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_data[:,i*36:(i+1)*36],labels=label_y[:,i*36:(i+1)*36]))
    loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_y, logits=y_data),axis=1))
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #print(sess.run(y_data, feed_dict={conv: test_train_x, label_y: test_train_y}))

    predict_op = tf.reshape(y_data, [-1, 4, 36])
    predict_op = tf.argmax(predict_op, axis=2)
    Y_op = tf.reshape(label_y, [-1, 4, 36])
    Y_op = tf.argmax(Y_op, axis=2)
    correct_prediction = tf.equal(predict_op, Y_op)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


    for e in range(70):
        current_loss = 0;
        for i in range(n_iters_per_epoch):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]
            batch_y = train_y[i*batch_size:(i+1)*batch_size]
            l,_ = sess.run([loss,train_op],feed_dict={conv:batch_x,label_y:batch_y})
            current_loss += l
        #randomx = np.random.permutation(train_num)[:2000]
        #train_ac = accuracy.eval({conv:train_x[randomx],label_y:train_y[randomx]})
        print("%d train_loss：" % e, current_loss)
        #print("train_accuracy：",train_ac)
        current_loss_test = 0;
        for num in range(4):
            current_loss_test = 0;
            for i in range(num_iter):
                batch_x = test_train_x[i * batch_size:(i + 1) * batch_size]
                batch_y = test_train_y[i * batch_size:(i + 1) * batch_size]
                l, _ = sess.run([loss, train_op], feed_dict={conv: batch_x, label_y: batch_y})
                current_loss_test += l
        print("%d test_loss：" % e, current_loss_test)

        #if (current_loss/10+current_loss_test<=1.0):break;


    for e in range(10):
        current_loss = 0;
        for i in range(num_iter_test):
            batch_x = test_train_x_1[i * batch_size:(i + 1) * batch_size]
            batch_y = test_train_y_1[i * batch_size:(i + 1) * batch_size]
            l, _ = sess.run([loss, train_op], feed_dict={conv: batch_x, label_y: batch_y})
            current_loss += l
        #randomx = np.random.permutation(test_train_x.shape[0])[:500]
        train_ac = accuracy.eval({conv: test_train_x_1, label_y: test_train_y_1})
        print("%d loss：" % e, current_loss)
        print("train_accuracy：", train_ac)

    saver.save(sess, './model/vcode-model.ckpt-done')

    val_ac = accuracy.eval({conv: val_x, label_y: val_y})
    print("val_accuracy：", val_ac)
    test_ac = accuracy.eval({conv:test_x,label_y:test_y})
    print("test_accuracy：", test_ac)



