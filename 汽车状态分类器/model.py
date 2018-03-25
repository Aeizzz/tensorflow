

# from data_processing import load_data, convert2onehot
import data_processing
import numpy as np
import tensorflow as tf

# 获取数据，数据处理
data = data_processing.load_data(download=False)
new_data = data_processing.convert2onehot(data)

new_data = new_data.values.astype(np.float32)
np.random.shuffle(new_data)

# 分成训练集和测试机
sep = int(0.7*len(new_data))
train_data = new_data[:sep]
test_data = new_data[sep:]


tf_input = tf.placeholder(tf.float32,[None,25],'input') # 25列
# 分开x和y
tfx = tf_input[:,:21]
tfy = tf_input[:,21:]

# 两个隐藏层 一个输出层
l1 = tf.layers.dense(tfx,128,tf.nn.relu,name='l1')
l2 = tf.layers.dense(l1,128,tf.nn.relu,name='l2')
out = tf.layers.dense(l2,4,tf.nn.relu,name='l3')

prediction = tf.nn.softmax(out,name='pred')

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy,logits=out)

accuracy = tf.metrics.accuracy(    # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy,axis=1),predictions=tf.argmax(out,axis=1),
)[1]

opt = tf.train.GradientDescentOptimizer(0.1)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
for t in range(4000):
    batch_index = np.random.randint(len(train_data),size=32)
    sess.run(train_op,{tf_input:train_data[batch_index]})
    if t%50 == 0:
        acc_, pred_, loss_ = sess.run([accuracy,prediction,loss],{tf_input:test_data})
        print("Step: %i" % t,"| Accurate: %.2f"%acc_,"| Loss: %.2f"%loss_,)
