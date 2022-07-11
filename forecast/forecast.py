import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

today = datetime.datetime.now()

file_path = 'forecast/data.csv'
data = pd.read_csv(file_path, delimiter=',', header = 0, skipinitialspace = True)
data.head(24)

temperature = np.array(data['temperature'])

num_periods = 24
f_horizon = 1
x_train = temperature[:(len(temperature)-num_periods)]
x_batches = x_train.reshape(-1, num_periods, 1)

y_train = temperature[1:(len(temperature)-num_periods) + 1]
y_batches = y_train.reshape(-1, num_periods, 1)



def test_data(series, forecast, num):
    testX = temperature[-(num + forecast):][:num].reshape(-1, num_periods, 1)
    testY = temperature[-(num):].reshape(-1, num_periods, 1)
    return testX, testY
X_test, Y_test = test_data(temperature, f_horizon, 24*2)

tf.compat.v1.reset_default_graph()
rnn_size = 100
learning_rate=0.001

tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, [None, num_periods, 1])
Y = tf.compat.v1.placeholder(tf.float32, [None, num_periods, 1])

rnn_cells=tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=rnn_size, activation=tf.nn.relu)
rnn_output, states = tf.compat.v1.nn.dynamic_rnn(rnn_cells, X, dtype=tf.float32)

output=tf.reshape(rnn_output, [-1, rnn_size])
logit=tf.compat.v1.layers.dense(output, 1, name="softmax")

outputs=tf.reshape(logit, [-1, num_periods, 1])
print(logit)

loss = tf.reduce_sum(input_tensor=tf.square(outputs - Y))

accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(tf.argmax(input=logit, axis=1), tf.cast(Y, tf.int64)), tf.float32))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step=optimizer.minimize(loss)

epochs = 1000

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

for epoch in range(epochs):
    train_dict = {X: x_batches, Y: y_batches}
    sess.run(train_step, feed_dict=train_dict)

saver = tf.compat.v1.train.Saver()
save_path = saver.save(sess, "models/model.ckpt")

with tf.compat.v1.Session() as sess:
  # Restore variables from disk.
    saver.restore(sess, "models/model.ckpt")
    y_pred=sess.run(outputs, feed_dict={X: X_test})
    print ('Y predict:')
    y_pred = y_pred.reshape(2, 24)
    hourList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    for i in range (0, 24):
        y_pred[1][i] = round((y_pred[1][i]), 2)
    
    print (y_pred[1])

    df = pd.DataFrame({"Hour": hourList, "temperature": y_pred[1]})    
    df.to_csv('forecast/csv/246.csv')


'''plt.title("Compare Predict vs Actual Temperature", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=5, label="Actual")
plt.plot(pd.Series(np.ravel(y_pred[1])), "r.", markersize=5, label="Predict")
plt.legend(loc="upper left")
plt.xlabel("Time line")
plt.savefig('src/public/img/' + str(today.day) + str(today.month) + '.png')'''
