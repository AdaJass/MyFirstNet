
import tensorflow as tf  
import util
import numpy as np  
import predata as pdt
import os 
  
train_epochs = 20  ## int(1e5+1)  
GRID_HIGH = pdt.GRID_HIGH
PERIOD = pdt.PERIOD 
INPUT_HEIGHT = 5   
INPUT_WIDTH = 552
batch_size = 1

# The default pa
# 
# th for saving event files is the same folder of this python file.
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
## 输入n*5*552*1 
weight_1 = tf.Variable(tf.truncated_normal(shape=[3, 48, 1, 90], stddev=0.1, name = 'weight_1'))  
bias_1 = tf.Variable(tf.constant(0.0, shape=[90], name='bias_1'))  
conv1 = tf.nn.conv2d(input=input_matrix, filter=weight_1, strides=[1, 2, 3, 1], padding='SAME')   #i don't know how to set the padding 
conv1 = tf.nn.bias_add(conv1, bias_1, name='conv_1')  
acti1 = tf.nn.relu(conv1, name='acti_1')  
# pool1 = tf.nn.max_pool(value=acti1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')  
  
## 2 conv layer  
## 输入n*3*184*90 
weight_2 = tf.Variable(tf.truncated_normal(shape=[2, 12, 90, 75], stddev=0.1, name='weight_2'))  
bias_2 = tf.Variable(tf.constant(0.0, shape=[75], name='bias_2'))  
conv2 = tf.nn.conv2d(input=acti1, filter=weight_2, strides=[1, 2, 2, 1], padding='SAME')  
conv2 = tf.nn.bias_add(conv2, bias_2, name='conv_2')  
acti2 = tf.nn.relu(conv2, name='acti_2')  
# pool2 = tf.nn.max_pool(value=acti2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')  
  
## 3 conv layer  
## 输入n*2*92*75 
weight_3 = tf.Variable(tf.truncated_normal(shape=[1, 3, 75, 60], stddev=0.1, name='weight_3'))  
bias_3 = tf.Variable(tf.constant(0.0, shape=[60]))  
conv3 = tf.nn.conv2d(input=acti2, filter=weight_3, strides=[1, 1, 1, 1], padding='SAME')  
conv3 = tf.nn.bias_add(conv3, bias_3)  
acti3 = tf.nn.relu(conv3, name='acti_3')  
# pool3 = tf.nn.max_pool(value=acti3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_3')  
  
## 1 deconv layer  
## 输入n*2*92*60
## 经过反卷积，输出n*2*184*75                          #height width  outchannel inchannel
deconv_weight_1 = tf.Variable(tf.truncated_normal(shape=[1, 3, 75, 60], stddev=0.1), name='deconv_weight_1')  
deconv1 = tf.nn.conv2d_transpose(value=acti3, filter=deconv_weight_1, output_shape=[batch_size, 2, 92, 75], strides=[1, 1, 1, 1], padding='SAME', name='deconv_1')  
  
## 2 deconv layer  
## 输入2*240*75
## 经过反卷积，输出3*480*90 
deconv_weight_2 = tf.Variable(tf.truncated_normal(shape=[2, 12, 90, 75], stddev=0.1), name='deconv_weight_2')  
deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv_weight_2, output_shape=[batch_size, 3, 184, 90], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')  
  
## 3 deconv layer  
## 输入3*480*90 
## 经过反卷积，输出5*1440*90
deconv_weight_3 = tf.Variable(tf.truncated_normal(shape=[3, 48, 1, 90], stddev=0.1, name='deconv_weight_3'))  
deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv_weight_3, output_shape=[batch_size, 5, 552, 1], strides=[1, 2, 3, 1], padding='SAME', name='deconv_3')  
  
# ## conv layer  
# ## 输入5*1440*90
# ## 经过卷积，输出为5*1440*1
# weight_final = tf.Variable(tf.truncated_normal(shape=[5, 480, 90, 1], stddev=0.1, name = 'weight_final'))  
# bias_final = tf.Variable(tf.constant(0.0, shape=[1], name='bias_final'))  
# conv_final = tf.nn.conv2d(input=deconv3, filter=weight_final, strides=[1, 1, 1, 1], padding='SAME')  
# conv_final = tf.nn.bias_add(conv_final, bias_final, name='conv_final')  
  
## output  
## 输入5*1440*1
## reshape为5*1440
output = tf.reshape(deconv3, shape=[-1, INPUT_HEIGHT, INPUT_WIDTH])  
  
## loss and optimizer  
loss = tf.reduce_mean(tf.pow(tf.subtract(output, input_y), 2.0))  
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型

current_iter = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01,current_iter,
                                        decay_steps = train_epochs,
                                        decay_rate=0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  


def tidyInput(indt=0):
    """
    input:
        dict of high low close volume  time
    output:
        the input data to the network.
    """    
    if not indt and len(indt['close'])<INPUT_WIDTH:
        print('net work input data invalid.')
        return

    data=pdt.DataFrame()
    data['h'] = indt['high'][-INPUT_WIDTH:]
    data['l'] = indt['low'][-INPUT_WIDTH:]
    data['c'] = indt['close'][-INPUT_WIDTH:]
    data['v'] = indt['volume'][-INPUT_WIDTH:]
    data['t'] = indt['time'][-INPUT_WIDTH:]
    max_v = max(data['h'])
    min_v = min(data['l'])
    f_k = lambda x: (pdt.GRID_HIGH*(x-min_v)/(max_v-min_v))
    data['h'] = data['h'].apply(f_k)
    data['l'] = data['l'].apply(f_k)
    data['c'] = data['c'].apply(f_k)
    data['t'] = data['t'].apply(pdt.makeTime)
    data['v'].astype('float')
    matrix = data.as_matrix()
    matrix = matrix.transpose()
    matrix = matrix/pdt.GRID_HIGH
    return [matrix], min_v, max_v
    pass

def tidyOutput(odt, min_v, max_v):
    odt = odt * pdt.GRID_HIGH
    data = odt[0]
    data = np.array(data)
    data = data.transpose()
    data = pdt.DataFrame(data)
    mid_v = (max_v - min_v)/pdt.GRID_HIGH
    f_k = lambda x: mid_v *x + min_v
    data[0] = data[0].apply(f_k)
    data[1] = data[1].apply(f_k)
    data[2] = data[2].apply(f_k)
    return df

def PredictNext(currentX):
    sess = tf.Session()
    print('i have been here!!')
    model_file=tf.train.latest_checkpoint('./models552/')
    saver.restore(sess,model_file)
    output_result = sess.run(output, feed_dict={input_x: currentX})  
    sess.close()
    return output_result


if __name__ == '__main__':
    import predata as pt
    data=pt.loadData()
    data=data[-batch_size:]
    data = np.array(data)
    currentX = data[:,0]
    print(currentX*pdt.GRID_HIGH)
    print(PredictNext(currentX)*pdt.GRID_HIGH)
