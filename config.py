import tensorflow as tf

class AgentConfig(object):
    # train / test
    is_train = False

    # LSTM
    h_size = 50
    lstm_in = 32

    # test model
    play_model = 'models/'
    is_save = True

    # train model
    save_dir = 'models/save/'
    log_dir = 'logs/'
    memory_size = 500000
    learn_start = 5000
    test_step = 1000
    save_step = 50000
    max_step = 2000000
    target_q_update_step = 10000
    batch_size = 32
    train_frequency = 4
    discount = 0.99
    # learning rate
    learning_rate = 0.0001
    learning_rate_minimum = 0.000025
    learning_rate_decay = 0.5
    learning_rate_decay_step = 1000000
    # experience replay
    ep_start = 1.  # 1: fully random; 0: no random
    ep_end = 0.1
    ep_end_t = 1000000

    # debug
    # learn_start = 500
    # test_step = 500
    # save_step = 1000
    # target_q_update_step = 1000

class EnvironmentConfig(object):
    # params for environment
    screen_width  = 63
    screen_height = 63
    screen_channel = 3
    dataset = 'moderate'  # mild / moderate / severe
    test_batch = 2048  # test how many patches at a time
    stop_step = 3
    reward_func = 'step_psnr_reward'

    # data path
    train_dir = 'data/train/'
    val_dir = 'data/valid/'
    test_dir = 'data/test/'


class DQNConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(FLAGS):
    config = DQNConfig

    # TF version
    tf_version = tf.__version__.split('.')
    if int(tf_version[0]) >= 1 and int(tf_version[1]) > 4:  # TF version > 1.4
        for k in FLAGS:
            v = FLAGS[k].value
            if hasattr(config, k):
                setattr(config, k, v)
    else:
        for k, v in FLAGS.__dict__['__flags'].items():
            if hasattr(config, k):
                setattr(config, k, v)

    return config
