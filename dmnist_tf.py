from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import math

import tensorflow as tf

FLAGS = None
batch_size = 100

cluster = None
server = None
is_chief = None


def main(_):
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:%s/task:%d" % (FLAGS.job_name, FLAGS.task_index),
            ps_device="/job:ps/cpu:0",
            cluster=cluster)):
        #
        # 1. read training data
        #

        # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
        # label - digit (0, 1, ..., 9)
        train_queue = tf.train.string_input_producer(
            [FLAGS.train_file],
            num_epochs=200)  # when all data is read, it raises OutOfRange
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
            shape=(batch_size,
                   784))  # here we use fixed dimension with batch_size. (Please use undefined dimension with None in production.)
        plchd_label = tf.placeholder(
            dtype=tf.int32,
            shape=(
                batch_size))  # here we use fixed dimension with batch_size. (Please use undefined dimension with None in production.)

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
        global_step = tf.train.get_or_create_global_step()
        # define optimization (training)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.07)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=plchd_label,
            logits=logits)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

        #
        # 3. run session
        #

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                checkpoint_dir=FLAGS.out_dir,
                is_chief=is_chief,
        ) as sess:  # use tf.train.MonitoredTrainingSession for more advanced features ...

            # train !!!
            local_step_value = 0
            array_image, array_label = sess.run(
                [train_batch_image, train_batch_label])
            while not sess.should_stop():
                feed_dict = {
                    plchd_image: array_image,
                    plchd_label: array_label
                }
                _, global_step_value, loss_value, array_image, array_label = sess.run(
                    [train_op, global_step, loss, train_batch_image, train_batch_label],
                    feed_dict=feed_dict)
                local_step_value += 1
                if local_step_value % 100 == 0:
                    print("Worker: Local Step %d, Global Step %d (Loss: %.2f)" % (
                        local_step_value, global_step_value, loss_value))
    print('Training done !')


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
    parser.add_argument(
        '--out_dir',
        type=str,
        default='yxckpt')
    parser.add_argument(
        '--job_name',
        type=str,
        required=True)
    parser.add_argument(
        '--task_index',
        type=int,
        required=True)
    FLAGS, unparsed = parser.parse_known_args()
    cluster = tf.train.ClusterSpec({
        'ps': [
            '192.168.50.209:2222'
        ],
        'worker': [
            '192.168.50.209:2223',
            '192.168.50.208:2223',
            '192.168.50.208:2224'
        ]})

    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index
    )
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        is_chief = (FLAGS.task_index == 0)
        main(unparsed)
