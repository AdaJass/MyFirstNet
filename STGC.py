  
import tensorflow as tf  
from datetime import datetime
import math
import numpy as np

train_epochs = 20 
PERIOD = 5  # time period is 5 minutes, it will process 5 minite period data.
INPUT_HEIGHT = 5   
INPUT_WIDTH = 1440
GRID_HIGH = 1000    #it means that all price data will in the interval of [0,1000]
batch_size = 360

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

current_iter = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01,current_iter,
                                        decay_steps = train_epochs,
                                        decay_rate=0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  

def PredictNext(currentX):
    sess = tf.Session()
    model_file=tf.train.latest_checkpoint('./models/')
    saver.restore(sess,model_file)
    output_result = sess.run(output, feed_dict={input_x: currentX})  
    sess.close()
    return output_result


if __name__ == '__main__':
    # with open('./XTIUSD5.csv','r') as f:
    #     rawdata = f.readlines()
    # rawdata = rawdata[-1440:]

    # def makeTime(dt):
    #     w = dt.weekday()    
    #     w_s = w*24*60/PERIOD
    #     h = dt.hour
    #     h_s = h*60/PERIOD
    #     m = dt.minute
    #     m_s = math.ceil(m/PERIOD)
    #     return int(w_s + h_s + m_s)    
    # middata = []
    # for line in rawdata:
    #     cell = line.split(',')
    #     time = cell[0]+' '+cell[1]
    #     time = datetime.strptime(time,'%Y.%m.%d %H:%M')
    #     highest = float(cell[3])
    #     lowest = float(cell[4])
    #     close = float(cell[5])
    #     volume = int(cell[6])
    #     time = makeTime(time)
    #     middata.append([highest, lowest, close, volume, time])
    # # print(middata)
    # middata = np.array(middata)
    # middata = middata.transpose()
    # max_v = max(middata[0])
    # min_v = min(middata[1])
    # f_k = lambda x: int(math.floor(GRID_HIGH*(x-min_v)/(max_v-min_v)))
    # for ii in range(3):
    #     for i,v in enumerate(middata[ii]):
    #         middata[ii][i] = f_k(v)    
    # # print(middata.shape)
    # middata = middata.reshape((1,5,1440))
    # print(PredictNext(middata))
    import predata as pt
    data=pt.loadData()
    data=data[-batch_size:]
    data = np.array(data)
    currentX = data[:,0]
    # print(currentX)
    print(PredictNext(currentX))

