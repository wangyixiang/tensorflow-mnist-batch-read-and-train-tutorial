from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import math

import tensorflow as tf

FLAGS = None
batch_size = 100

def main(_):
    #
    # 1. read training data
    #
    
    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
    # label - digit (0, 1, ..., 9)
    train_queue = tf.train.string_input_producer(
        [FLAGS.train_file],
        num_epochs = 10) # when all data is read, it raises OutOfRange
    train_reader = tf.TFRecordReader()
    _, train_serialized_exam = train_reader.read(train_queue)
    train_exam = tf.parse_single_example(
        train_serialized_exam,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    train_image = tf.decode_raw(train_exam['image_raw'], tf.uint8)
    train_image.set_shape([784])
    train_image = tf.cast(train_image, tf.float32) * (1. / 255)
    train_label = tf.cast(train_exam['label'], tf.int32)
    train_batch_image, train_batch_label = tf.train.batch(
        [train_image, train_label],
        batch_size=batch_size)

    #
    # 2. read test data
    #
    
    test_queue = tf.train.string_input_producer(
        [FLAGS.test_file],
        num_epochs = 1) # when all data is read, it raises OutOfRange
    test_reader = tf.TFRecordReader()
    _, test_serialized_exam = test_reader.read(test_queue)
    test_exam = tf.parse_single_example(
        test_serialized_exam,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    test_image = tf.decode_raw(test_exam['image_raw'], tf.uint8)
    test_image.set_shape([784])
    test_image = tf.cast(test_image, tf.float32) * (1. / 255)
    test_label = tf.cast(test_exam['label'], tf.int32)
    test_batch_image, test_batch_label = tf.train.batch(
        [test_image, test_label],
        batch_size=batch_size) # simply enqueue/dequeue_many with tf.FIFOQueue

#     # for debugging... (check input)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         tf.train.start_queue_runners(sess=sess)
#         for i in range(2):
#             debug_image, debug_label = sess.run([train_batch_image, train_batch_label])
#             tf.summary.image('images', debug_image)
#             print(debug_label)
    
    #
    # 2. define graph
    #
    
    # define input
    plchd_image = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size, 784)) # here we use fixed dimension with batch_size. (Please use undefined dimension with None in production.)
    plchd_label = tf.placeholder(
        dtype=tf.int32,
        shape=(batch_size)) # here we use fixed dimension with batch_size. (Please use undefined dimension with None in production.)
        
    # define network and inference
    # (simple 2 fully connected hidden layer : 784->128->64->10)
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal(
                [784, 128],
                stddev=1.0 / math.sqrt(float(784))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([128]),
            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(plchd_image, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal(
                [128, 64],
                stddev=1.0 / math.sqrt(float(128))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([64]),
            name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal(
                [64, 10],
                stddev=1.0 / math.sqrt(float(64))),
            name='weights')
        biases = tf.Variable(
            tf.zeros([10]),
            name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    
    # define optimization (training)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0.07)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=plchd_label,
        logits=logits)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    
    # define testing
    array_correct = tf.nn.in_top_k(logits, plchd_label, 1)
    test_op = tf.reduce_sum(tf.cast(array_correct, tf.int32))
    
    #
    # 3. run session
    #
        
    with tf.Session() as sess: # use tf.train.MonitoredTrainingSession for more advanced features ...
        sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # for data batching
        
        # train !!!
        try:
            step = 0
            while not coord.should_stop():
                array_image, array_label = sess.run(
                    [train_batch_image, train_batch_label])
                feed_dict = {
                    plchd_image: array_image,
                    plchd_label: array_label
                }
                _, loss_value = sess.run(
                    [train_op, loss],
                    feed_dict=feed_dict)
                if step % 100 == 0:
                    print("Worker: Step %d (Loss: %.2f)" % (step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Training done !')

        # test (evaluate) !!!
        num_true = 0
        try:
            num_test = 0
            while not coord.should_stop():
                array_image, array_label = sess.run(
                    [test_batch_image, test_batch_label])
                feed_dict = {
                    plchd_image: array_image,
                    plchd_label: array_label
                }
                num_true += sess.run(
                    test_op,
                    feed_dict = feed_dict)
                num_test += batch_size
        except tf.errors.OutOfRangeError:
            print('Scoring done !')
        precision = float(num_true) / num_test
        print('Accuracy: %0.04f (Num of samples: %d)' %
              (precision, num_test))
            
        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file',
        type=str,
        default='train.tfrecords',
        help='File path for the training data.')
    parser.add_argument(
        '--test_file',
        type=str,
        default='test.tfrecords',
        help='File path for the test data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
