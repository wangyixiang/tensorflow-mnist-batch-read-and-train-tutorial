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
    # define graph (to be restored !)
    #

    # define input
    plchd_image = tf.placeholder(
        dtype=tf.float32,
        shape=(batch_size, 784))
    plchd_label = tf.placeholder(
        dtype=tf.int32,
        shape=(batch_size))

    # define network and inference
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

    #
    # Restore and Testing
    #

    ckpt = tf.train.get_checkpoint_state(FLAGS.out_dir)
    idex = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore graph
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        # add to graph - read test data
        test_queue = tf.train.string_input_producer(
            [FLAGS.test_file],
            num_epochs=1)  # when data is over, it raises OutOfRange
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
            batch_size=batch_size)

        # add to graph - test (evaluate) graph
        array_correct = tf.nn.in_top_k(logits, plchd_label, 1)
        test_op = tf.reduce_sum(tf.cast(array_correct, tf.int32))

        # run
        sess.run(tf.initialize_local_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # for data batching
        num_test = 0
        num_true = 0
        array_image, array_label = sess.run(
            [test_batch_image, test_batch_label])
        try:
            while True:
                feed_dict = {
                    plchd_image: array_image,
                    plchd_label: array_label
                }
                batch_num_true, array_image, array_label = sess.run(
                    [test_op, test_batch_image, test_batch_label],
                    feed_dict=feed_dict)
                num_true += batch_num_true
                num_test += batch_size
        except tf.errors.OutOfRangeError:
            print('Scoring done !')
        precision = float(num_true) / num_test
        print('Accuracy: %0.04f (Num of samples: %d)' %
              (precision, num_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_file',
        type=str,
        default='test.tfrecords',
        help='File path for the test data.')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='yxckpt',
        help='Dir path for the model and checkpoint output.')
    FLAGS, unparsed = parser.parse_known_args()

    main(unparsed)
