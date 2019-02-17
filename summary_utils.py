import os
import tensorflow as tf

def add_scalar_summaries(metrics, postfix=None):
    sums = []
    for name, op in metrics.items():
        if postfix is not None:
            name = '{}_{}'.format(name, postfix)
        sums.append(tf.summary.scalar(name, op))
    return tf.summary.merge(sums)

def get_summary_writers(sess, logs_path, modes=('train', 'val')):
    writers = [tf.summary.FileWriter(os.path.join(logs_path, mode), graph=sess.graph)
                for mode in modes]
    return writers