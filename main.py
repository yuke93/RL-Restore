import random
import tensorflow as tf
from dqn.agent import Agent
from dqn.environment import MyEnvironment
from config import get_config
import sys

# Parameters
flags = tf.app.flags
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_boolean('is_train', False, 'Whether to do training or testing')
# test
flags.DEFINE_boolean('is_save', True, 'Whether to save results')
flags.DEFINE_string('dataset', 'moderate', 'Select a dataset from mild/moderate/severe')
flags.DEFINE_string('play_model', 'models/', 'Path for testing model')
# training
flags.DEFINE_string('save_dir', 'models/save/', 'Path for saving models')
flags.DEFINE_string('log_dir', 'logs/', 'Path for logs')
FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        config = get_config(FLAGS)
        env = MyEnvironment(config)
        agent = Agent(config, env, sess)

        if FLAGS.is_train:
            agent.train()
        else:
            if FLAGS.dataset == 'mine':
                agent.play_mine()
            else:
                agent.play()


if __name__ == '__main__':
    tf.app.run()
