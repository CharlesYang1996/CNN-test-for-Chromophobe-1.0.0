# -*- coding: utf-8 -*-
"""
Created on Fri Mar 8 21:43:06 2019
@author: Feary
"""
from skimage import io, transform
# import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

path = 'E:/python_work/skinclassify/image/'
max_step = 500  # 训练论述
batch_size = 32
rate = 0.8  # 训练集和验证集的比率
resize_w = 100
resize_h = 100


# 读取图片
def get_files(filename):
      imgs = []

  labels = []
  labels_name = []
  labels_index = []
  i = 0
  for label_name in os.listdir(filename):  # 该位置的每个文件夹
        labels_name.append(label_name)  # 文件夹名称就是类别
    labels_index.append(i)
   
    for pic in glob.glob(filename + label_name + '/*.jpg'):  # 每个文件夹里面的图片
           # pic=filename+label_name+'/'+pic
          print('reading the images:%s' % (pic))
          pic = io.imread(pic)
          pic = transform.resize(pic, (100, 100))
          imgs.append(pic) 
          labels.append(i)
           # 主要区别在于 np.array （默认情况下）将会copy该对象，而 np.asarray 除非必要，否则不会copy该对象。
        i = i + 1
      return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
    image_list, label_list = get_files(path)


# 打乱所有图片顺序后 将所有数据分为训练集和验证集
def create_train_val(image_list, label_list, rate):
      num_example = len(image_list)  # 数据个数

  arr = np.arange(num_example)  # 0~num_example-1之间的编号 代表每张图片
  np.random.shuffle(arr)  # 图片顺序打乱
  image_list = image_list[arr]
  label_list = label_list[arr]
  s = np.int(num_example * rate)
  x_train = image_list[:s]
  y_train = label_list[:s]
  x_val = image_list[s:]
  y_val = label_list[s:]
  return x_train, y_train, x_val, y_val
x_train, y_train, x_val, y_val = create_train_val(image_list, label_list, rate)

# 定义一个函数，按批次取数据
def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)  # 使用tf.cast转化为tensorflow数据格式
    # image = tf.image.decode_jpeg(image,channels = 3)#decode_jpeg函数为jpeg（jpg）图片解码的过程
    # resize_image_with_crop_or_pad 剪裁或填充处理，会根据原图像的尺寸和指定的目标图像的尺寸选择剪裁还是填充，如果原图像尺寸大于目标图像尺寸，则在中心位置剪裁，反之则用黑色像素填充。
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # image = tf.image.resize_image(image,resize_w,resize_h)
    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
    # image = np.asarray(image.eval(), dtype='uint8')
    # plt.imshow(image)
    # plt.show()
    # (x - mean) / adjusted_stddev 标准化，即减去所有图片的均值，方便训练。
    image = tf.image.per_image_standardization(image)
    # 使用tf.train.batch函数产生训练的批次。
    image_batch, label_batch = tf.train.batch([image, label],
                      batch_size = batch_size,
                      num_threads = 64,  # 用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
                      capacity = capacity)  # 用来设置队列中元素的最大数量
    images_batch = tf.cast(image_batch, tf.float32)  # 产生的批次做数据类型的转换和shape的处理
    # labels_batch = tf.reshape(label_batch,[batch_size])
    labels_batch = tf.cast(label_batch, tf.int32)
    return images_batch, labels_batch


# 训练参数的定义及初始化
def weight_variable(shape):
      return tf.Variable(tf.truncated_normal(shape, stddev=0.01))  # 截断的正态分布噪声


def bias_variable(shape):
      return tf.Variable(tf.constant(0.01, shape=shape))  # 截断的正态分布噪声


# 卷积层、池化层的定义 （方便重复使用）
def conv2d(x, w):  # 卷积层
      x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
      return x

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 池化层 2x2最大池化

x = tf.placeholder(tf.float32, shape=[None, resize_w, resize_h, 3], name='x')
# 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一层卷积层
w_conv1 = weight_variable([5, 5, 3, 32])  # 卷积尺寸5x5,3颜色通道，32个不同的卷积核
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)  # 使用relu激活函数进行非线性处理
h_pool1 = max_pool_2x2(h_conv1)  # 池化
print(h_pool1)
# 第2层卷积层
w_conv2 = weight_variable([5, 5, 32, 64])  #
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # 使用relu激活函数进行非线性处理
h_pool2 = max_pool_2x2(h_conv2)  # 池化
print(h_pool2)
# 第3层卷积层
w_conv3 = weight_variable([3, 3, 64, 128])  #
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)  # 使用relu激活函数进行非线性处理
h_pool3 = max_pool_2x2(h_conv3)  # 池化
print(h_pool3)
# 第4层卷积层
w_conv4 = weight_variable([3, 3, 128, 128])  #
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)  # 使用relu激活函数进行非线性处理
h_pool4 = max_pool_2x2(h_conv4)  # 池化
print(h_pool4)
# 全连接层1 （通过四层卷积层后图片尺寸变成6 * 6 * 128）
h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 128])  # 对第四层卷积层输出进行编写，转成ID向量
w_fc1 = weight_variable([7 * 7 * 128, 1024])  # 隐含节点数1024
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)
# 全连接层2
w_fc2 = weight_variable([1024, 512])  # 隐含节点数1024
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
# 为减轻过拟合，使用一个Dropout层
# keep_prob=tf.placeholder(tf.float32)
# h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)
# Dropout层的输出连接一个softmax层得到最后概率输出
w_fc3 = weight_variable([512, 7])  # 隐含节点数1024
b_fc3 = bias_variable([7])
# logits = tf.nn.softmax(tf.matmul(h_fc2,w_fc3)+b_fc3)
logits = tf.add(tf.matmul(h_fc2, w_fc3), b_fc3)

# 定义损失函数cross_entropy
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=logits)  # softmax 的交叉熵
cost = tf.reduce_mean(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)  # 优化算法
# 定义评测准确率
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)   # tf.equal判断分类的结果是否正确
# tf.argmax就是返回最大的那个数值所在的下标,axis = 1:表示行 概率最大的那个
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 先将bool值转化为float32 再求平均
top_3_op = tf.nn.in_top_k(logits, y_, 3)  # 输出前三个类别

# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
      assert len(inputs) == len(targets)  # 断定此处是对的,如果表达式不成立（False），则抛出异常

  if shuffle:
        indices = np.arange(len(inputs))
    np.random.shuffle(indices)  # 打乱顺序
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
              excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]

# 训练数据
sess = tf.InteractiveSession()  # 创建默认的session
tf.global_variables_initializer().run()  # 初始化所有参数
# sess.run(tf.global_variables_initializer())
for step in range(max_step):
      start_time = time.time()
  train_loss, train_acc, n_batch = 0, 0, 0
  for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_step, cost, acc], feed_dict={x: x_train_a, y_: y_train_a})
    train_loss += err;
train_acc += ac;
n_batch += 1
  if ((step + 1) % 100 == 0):
        print("  trian iter:%d" % (step + 1))
        print("  train loss: %f" % (train_loss / n_batch))
        print("  train acc: %f" % (train_acc / n_batch))
        print("---------------------")

    # validation
    val_loss, val_acc, n_batchv = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
          err, ac = sess.run([cost, acc], feed_dict={x: x_val_a, y_: y_val_a})
      val_loss += err;
    val_acc += ac;
    n_batchv += 1
    print("validation result:")
    print()
    print("  validation loss: %f" % (val_loss / n_batchv))
    print("  validation acc: %f" % (val_acc / n_batchv))

    sess.close()