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
flags.DEFINE_boolean('is_save', True, 'Whether to save results')
flags.DEFINE_string('dataset', 'moderate', 'Select a dataset from mild/moderate/severe')
FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        config = get_config(FLAGS) or FLAGS
        env = MyEnvironment(config)
        agent = Agent(config, env, sess)

        if FLAGS.is_train:
            # agent.train()
            print('To be released.')
        else:
            agent.play()


if __name__ == '__main__':
    tf.app.run()
