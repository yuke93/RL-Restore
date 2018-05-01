import os
import pprint
import inspect
import tensorflow as tf


def class_vars(obj):
    return {k:v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


class BaseModel(object):
    """Abstract object representing an Reader model."""
    def __init__(self, config):
        self._saver = None
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        pp = pprint.PrettyPrinter().pprint
        pp(self._attrs)

        self.config = config

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))


    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)


    def load_model(self):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.play_model)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.play_model, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.play_model)
            return False


    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)


    @property
    def model_dir(self):
        return 'model/'


    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
