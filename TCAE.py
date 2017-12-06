  
import tensorflow as tf  
import numpy as np  
import predata as pdt
import os 
  
train_epochs = 20  ## int(1e5+1)  
  
INPUT_HEIGHT = 5  
INPUT_WIDTH = 1440
  
batch_size = 360

# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string('log_dir', 
    os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS
  
## 原始输入是5*1440*1
input_x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT , INPUT_WIDTH], name='input_x')  
input_matrix = tf.reshape(input_x, shape=[-1, INPUT_HEIGHT, INPUT_WIDTH, 1])  
input_y = tf.placeholder(tf.float32, shape=[None, INPUT_HEIGHT, INPUT_WIDTH], name='input_y')  
  
## 1 conv layer
## 输入n*5*1440*1 
weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 144, 1, 90], stddev=0.1, name = 'weight_1'))  
bias_1 = tf.Variable(tf.constant(0.0, shape=[90], name='bias_1'))  
conv1 = tf.nn.conv2d(input=input_matrix, filter=weight_1, strides=[1, 2, 3, 1], padding='SAME')   #i don't know how to set the padding 
conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')  
acti1 = tf.nn.relu(conv1, name='acti_1')  
# pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')  
  
## 2 conv layer  
## 输入n*3*480*90 
weight_2 = tf.Variable(tf.truncated_normal(shape=[2, 48, 90, 75], stddev=0.1, name='weight_2'))  
bias_2 = tf.Variable(tf.constant(0.0, shape=[75], name='bias_2'))  
conv2 = tf.nn.conv2d(input=acti1, filter=weight_2, strides=[1, 2, 2, 1], padding='SAME')  
conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')  
acti2 = tf.nn.relu(conv2, name='acti_2')  
# pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')  
  
## 3 conv layer  
## 输入n*2*240*75 
weight_3 = tf.Variable(tf.truncated_normal(shape=[1, 12, 75, 60], stddev=0.1, name='weight_3'))  
bias_3 = tf.Variable(tf.constant(0.0, shape=[60]))  
conv3 = tf.nn.conv2d(input=acti2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')  
conv3 = tf.nn.bias_add(conv3, bias_3)  
acti3 = tf.nn.relu(conv3, name='acti_3')  
# pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')  
  
## 1 deconv layer  
## 输入n*2*240*60
## 经过反卷积，输出n*2*240*75                          #height width  outchannel inchannel
deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[1, 12, 75, 60], stddev=0.1), name='deconv_weight_1')  
deconv1 = tf.nn.conv2d_transpose(value=acti3, filter=deconv_weight_1, output_shape=[batch_size, 2, 240, 75], strides=[1, 1, 1, 1], padding='SAME', name='deconv_1')  
  
## 2 deconv layer  
## 输入2*240*75
## 经过反卷积，输出3*480*90 
deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[2, 48, 90, 75], stddev=0.1), name='deconv_weight_2')  
deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 3, 480, 90], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')  
  
## 3 deconv layer  
## 输入3*480*90 
## 经过反卷积，输出5*1440*90
deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 144, 90, 90], stddev=0.1, name='deconv_weight_3'))  
deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 5, 1440, 90], strides=[1, 2, 3, 1], padding='SAME', name='deconv_3')  
  
## conv layer  
## 输入5*1440*90
## 经过卷积，输出为5*1440*1
weight_final = tf.Variable(tf.truncated_normal(shape=[5, 480, 90, 1], stddev=0.1, name = 'weight_final'))  
bias_final = tf.Variable(tf.constant(0.0, shape=[1], name='bias_final'))  
conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')  
conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')  
  
## output  
## 输入5*1440*1
## reshape为5*1440
output = tf.reshape(conv_final, shape=[-1, INPUT_HEIGHT, INPUT_WIDTH])  
  
## loss and optimizer  
loss = tf.reduce_mean(tf.pow(tf.subtract(output, input_y), 2.0))  
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1) # 声明tf.train.Saver类用于保存模型
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)  
  
with tf.Session() as sess:  
    # writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    all_data = pdt.loadData()
    train_test_pivot = int(len(all_data)*0.8)
    train_data = all_data[0: train_test_pivot]
    test_data = all_data[train_test_pivot:]    
    n_samples = len(train_data)  
    print('train samples: %d' % n_samples)  
    print('batch size: %d' % batch_size)  
    total_batch = int(n_samples / batch_size)  
    print('total batchs: %d' % total_batch)  
    init = tf.global_variables_initializer()  
    sess.run(init)  
    for epoch in range(train_epochs):  
        for batch_index in range(total_batch):  
            batch_data = train_data[batch_index*batch_size:(batch_index+1)*batch_size]  #mnist.train.next_batch(batch_size)  
            batch_data = np.array(batch_data)
            batch_x = batch_data[:,0]
            batch_y = batch_data[:,1]
            _, train_loss = sess.run([optimizer, loss], feed_dict={input_x: batch_x, input_y: batch_y})  
            print('epoch: %04d\tbatch: %04d\ttrain loss: %.9f' % (epoch + 1, batch_index + 1, train_loss))  
  
    n_test_samples = len(test_data) 
    test_total_batch = int(n_test_samples / batch_size) 
    saver_path = saver.save(sess, "./model.ckpt")  # 将模型保存到save/model.ckpt文件
    print("Model saved in file:", saver_path) 
    for batch_index in range(test_total_batch):  
        batch_data = test_data[batch_index*batch_size:(batch_index+1)*batch_size]  #mnist.train.next_batch(batch_size)  
        batch_data = np.array(batch_data)
        batch_test_x = batch_data[:,0]
        batch_test_y = batch_data[:,1] 
        test_loss, output_result = sess.run([loss, output], feed_dict={input_x: batch_test_x, input_y: batch_test_y})  
        print('test batch index: %d\ttest loss: %.9f' % (i + 1, test_loss)) 

        # for index in range(batch_size):  
        #     array = np.reshape(pred_result[index], newshape=[INPUT_HEIGHT, INPUT_WIDTH])  
        #     array = array * 255  
        #     image = Image.fromarray(array)  
        #     if image.mode != 'L':  
        #         image = image.convert('L')  
        #     image.save('./pred/' + str(i * batch_size + index) + '.png')  
        #     array_raw = np.reshape(noise_test_x[index], newshape=[INPUT_HEIGHT, INPUT_WIDTH])  
        #     array_raw = array_raw * 255  
        #     image_raw = Image.fromarray(array_raw)  
        #     if image_raw.mode != 'L':  
        #         image_raw = image_raw.convert('L')  
        #     image_raw.save('./pred/' + str(i * batch_size + index) + '_raw.png')  

writer.close()
sess.close()