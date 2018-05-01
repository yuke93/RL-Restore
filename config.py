class AgentConfig(object):
    # LSTM
    _h_size = 50
    _lstm_in = 32

    # test model
    _play_model = 'model/'
    is_save = True


class EnvironmentConfig(object):
    # changeable params for environment
    screen_width  = 63
    screen_height = 63
    screen_channel = 3
    dataset = 'moderate'  # mild / moderate / severe
    test_batch = 2048  # test how many patches at a time
    stop_step = 3
    reward_func = 'step_psnr_reward'


class DQNConfig(AgentConfig, EnvironmentConfig):
    pass


def get_config(FLAGS):
    config = DQNConfig

    for k, v in FLAGS.__dict__['__flags'].items():
        if hasattr(config, k):
            setattr(config, k, v)

    return config
