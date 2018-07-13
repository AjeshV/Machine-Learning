import input_data
#downloads dataset, splits and formats it.

mnist = input_data.read_data_sets("/Users/Ajesh/Desktop/UNH/Courses/Project1/dataset/", one_hot=True)
import tensorflow as tf

#learning rate determines how fast to update weights, if value is >, it skips optimal solution, value is <, might need many iterations.

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

#reducing dimensions by unstacking rows and lining them.
#output is 2d 10 dimentional vector.

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

#-----------------Model definitions----------------------

#lowering variance by increasing bias.

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

#summary to visualize weights and biases.
	
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)

#cost function to minimize error during training.

with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.scalar_summary("cost_function", cost_function)

#gradient descent to improve model using learning rate to pace.
	
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
init = tf.initialize_all_variables()

merged_summary_op = tf.merge_all_summaries()

#graph to visualize in tensorboard.

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.train.SummaryWriter('/Users/Ajesh/Desktop/UNH/Courses/PProject/logs', graph_def=sess.graph_def)

#model training.
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

#evaluating using accuracy.
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
