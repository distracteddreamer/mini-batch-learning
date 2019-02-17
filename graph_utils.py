import tensorflow as tf

def get_train_op(loss, optimizer, **optim_kwargs):
    optimizer = getattr(tf.train, optimizer)(**optim_kwargs)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = optimizer.minimize(loss)
    return train_step

def add_metric_avg_ops(metrics):
    step = tf.assign_add(tf.Variable(initial_value=0., trainable=False), 1)
    step_reset = tf.assign(step, 0)
    avgs = {}
    resets = {'step':step_reset}
    for name, metric in metrics.items():
        avg = tf.Variable(initial_value=0., trainable=False)
        avg = tf.assign(avg, (avg*(step-1) + metric)/step, name=name)
        avg_reset = tf.assign(avg, 0)
        avgs[name] = avg
        resets[name] = avg_reset
    return avgs, resets
