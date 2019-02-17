import tensorflow as tf
import json
import numpy as np
import os
import datetime

class Config(object):

    compare_fns = dict([(str(fn), fn) for fn in [np.less, np.greater]])
    def __init__(self, name=None, json_file=None):
        if json_file is not None:
            with open(json_file) as f:
                config_dict = json.load(f)
            for k,v in config_dict.items():
                setattr(self, k, v)
            self.metric_compare = self.compare_fns[self.metric_compare]
        else:
            self.model_name = name
            self.unique_name = '{}-{:%Y%m%dT%H%M}'.format(name, datetime.datetime.now())
            self.ckpt = None
            self.iters_done = 0
    

    def make_paths(self):
        self.logs_path = 'models/{}/logs'.format(self.unique_name)
        self.save_path = 'models/{}/ckpts'.format(self.unique_name)
        
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
            
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
    
    def set_train_stats(self, batch_size, 
                        num_train=None, num_val=None, n_epochs=None,
                        n_iters=None, valid_iters=None, valid_every=None):
        self.batch_size = batch_size
        if n_iters is None:
            self.num_train = num_train
            self.num_val = num_val
            self.n_epochs = n_epochs
            self.steps_per_epoch = int(np.ceil(num_train/(self.batch_size)))
            self.valid_every = int(self.steps_per_epoch)
            self.n_iters = int(self.steps_per_epoch*self.n_epochs)
            self.valid_iters = int(np.ceil(num_val/(self.batch_size)))
        else:
            self.n_iters = n_iters
            self.valid_iters = valid_iters
            self.valid_every = valid_every
    
    def set_loss_attrs(self):
        self.best_metric = np.inf
        self.metric_compare = np.less
        
    def set_metric_attrs(self):
        self.best_metric = 0.
        self.metric_compare = np.greater
        
    def update_best(self, metric, itr):
        self.best_metric = metric
        self.best_iter = itr
        
    def save_json(self, iters_done=None):
        if iters_done is not None:
            self.iters_done=iters_done
        self_dict = dict(self.__dict__)
        self_dict['metric_compare'] = str(self.metric_compare)
        with open('models/{}/config.json'.format(self.unique_name), 'w') as f:
            json.dump(obj=self_dict, fp=f)

class DataPipeline(object):

    def __init__(self, batch_size, map_fn):
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.iterators = {}
        self.handles = {}

    def create_data_pipeline(self, arrs, mode='train'):
        datasets = tuple(map(tf.data.Dataset.from_tensor_slices, arrs))
        dataset = tf.data.Dataset.zip(datasets)
        if mode=='train':
            dataset = dataset.shuffle(len(arrs[0]))
        
        dataset = dataset.map(lambda *x: self.map_fn(*x, mode))
        dataset = dataset.batch(self.batch_size).repeat()
        self.iterators[mode] = dataset.make_initializable_iterator()

    def preproc(self, arrs):
        dataset_handle = tf.placeholder(tf.string, shape=[])

        for mode in ['train', 'valid']:
            self.create_data_pipeline(arrs[mode], mode)
        
        train_itr = self.iterators['train']
        iterator = tf.data.Iterator.from_string_handle(dataset_handle, 
                                                    output_types=train_itr.output_types,
                                                    output_shapes=train_itr.output_shapes)
        return iterator.get_next(), dataset_handle

    def prepare(self, sess):
        sess.run({k: v.initializer for k,v in self.iterators.items()})
        self.handles = sess.run({k: v.string_handle() for k,v in self.iterators.items()})