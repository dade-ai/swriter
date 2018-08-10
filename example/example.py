import tensorflow as tf
from swriter import SubdirWriter
import random


def example_log_subdir():
    # writer
    writer = SubdirWriter('./logs')

    p = tf.placeholder(tf.float32, shape=(), name='p')

    s = tf.summary.scalar('p', p)
    sess = tf.Session()  # get_default_session()

    for i in range(100):
        # log event in logs/
        out = sess.run(s, feed_dict={p: random.random()})
        # the default log destination is `logdir`
        writer.add_summary(out, global_step=i)

        # log event in logs/task1/validation
        out = sess.run(s, feed_dict={p: random.random()})
        writer.add_summary(out, global_step=i, subdir='task1/validation')

        # log event in logs/task1/test
        out = sess.run(s, feed_dict={p: random.random()})
        writer.add_summary(out, global_step=i, subdir='task1/test')

        # log event in logs/task2/validation
        out = sess.run(s, feed_dict={p: random.random()})
        writer.add_summary(out, global_step=i, subdir='task2/validation')

        # log event in logs/task2/test
        out = sess.run(s, feed_dict={p: random.random()})
        writer.add_summary(out, global_step=i, subdir='task2/test')


if __name__ == '__main__':
    example_log_subdir()



