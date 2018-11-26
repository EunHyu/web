import tensorflow as tf
import numpy as np

data = np.loadtxt('data.csv', delimiter=',',dtype=np.float32)

x1_data = data[:, 0]
x2_data = data[:, 1]
x3_data = data[:, 2]
y_data = data[:, 3]

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

normed_x1_data = (x1_data - sess.run(tf.reduce_mean(x1_data))) / (max(x1_data) - min(x1_data))
normed_x2_data = (x2_data - sess.run(tf.reduce_mean(x2_data))) / (max(x2_data) - min(x2_data))
normed_x3_data = (x3_data - sess.run(tf.reduce_mean(x3_data))) / (max(x3_data) - min(x3_data))

hypothesis = tf.sigmoid(w1 * normed_x1_data + w2 * normed_x2_data + w3 * normed_x3_data +b)

cost = -tf.reduce_mean(y_data * tf.log(hypothesis) + (1 - y_data) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

for step in range(2001) :
    sess.run(train)

    if step % 20 == 0 :
        print(step, "cost : ", sess.run(cost))

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), dtype=tf.float32))

print("hypothesis = ", sess.run(hypothesis))
print("accuracy = ", sess.run(accuracy))