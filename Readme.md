# MNIST Tutorial with TFRecord batch reading

Many TensorFlow's tutorials encapsulate how to read TFRecord with batch reading, but this sample (mnist_tf.py) shows end-to-end example with well-known MNIST (hand-writing digits) dataset from scratch.
(This code doesn't use high-level Estimator or Experiment.)

To simplify our example, here I used brief fully-connected neural networks without any modularity by functions and without detailed exception handling.

Please change this code to fit more advanced TensorFlow scenarios like benchmarking for more complicated networks, distributed TensorFlow (also with Google Cloud ML, Azure Batch AI, etc), TPU benchmarking, etc, etc. (This sample uses only standard functions.)

```bash
python mnist_tf --train_file /yourdatapath/train.tfrecords --test_file /yourdatapath/test.tfrecords
```

- This code reads TFRecords (train.tfrecords, test.tfrecords) with batching. When you set num_epochs, the data is read num_epochs times and you can catch the end of data by OutOfRange exception. (When you don't specify num_epochs, data is read unlimited times and you must set the number of steps.)

```python
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
```

- When you see data for debugging purpose, please uncomment the source code.

```python
# To see original data
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  thread = tf.train.start_queue_runners(sess=sess)
  for i in range(3):
    debug_image, debug_label = sess.run([train_image, train_label])
    print(debug_label)
```

```python
# To see batch data
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  tf.train.start_queue_runners(sess=sess)
  for i in range(2):
    debug_image, debug_label = sess.run([train_batch_image, train_batch_label])
    print(debug_label)
```

- This code runs the data reading operation and training operation separately. You can also connect these operations and finish with only one sess.run(), but here we enable to use logits for both training data and testing (scoring) data by separating these operations.

```python
with tf.Session() as sess:
  sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  array_image, array_label = sess.run(
    [train_batch_image, train_batch_label])
  feed_dict = {
    plchd_image: array_image,
    plchd_label: array_label
  }
  
  _, loss_value = sess.run(
    [train_op, loss],
    feed_dict=feed_dict)

  ...

  coord.request_stop()
  coord.join(threads)
```
