import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import cv2
import scipy as sci
from .base import BaseModel
from .ops import linear, conv2d


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.weight_dir = 'weights'
        self.env = environment
        self.cnn_format = 'NHWC'
        self.build_dqn()
        self.action_size = self.env.get_action_size()


    def predict_test(self, count_step = 0):
        if count_step == 0:
            imgs = self.env.get_data_test()
            env_steps = np.zeros(len(imgs), dtype=int)
            # initialize LSTM states
            self.state_test = (np.zeros([len(imgs), self.h_size]), np.zeros([len(imgs), self.h_size]))
            self.sess_test = tf.get_default_session()
        else:
            imgs = self.env.get_test_imgs()
            env_steps = self.env.get_test_steps()

        action_in = np.zeros([len(imgs), self.action_size - 1])
        if count_step > 0:
            for k in range(len(imgs)):
                if self.pre_action_test[k] < self.action_size - 1:
                    action_in[k, self.pre_action_test[k]] = 1.

        actions_vec, self.state_test = self.sess_test.run([self.q, self.rnn_state],
                                       {self.s_t: imgs, self.action_in: action_in, self.state_in: self.state_test,
                                        self.batch: len(imgs), self.length: 1})
        actions = actions_vec.argmax(axis=1)

        # choose the last action if pre-action is the last
        if count_step > 0:
            actions[self.pre_action_test==self.action_size - 1] = self.action_size - 1
        actions[env_steps == self.stop_step] = self.action_size - 1  # already stopped before

        return actions


    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        # training network
        activation_fn = tf.nn.relu
        with tf.variable_scope('prediction'):
            # input batch and length for recurrent training
            self.batch = tf.placeholder(tf.int32, shape=[])
            self.length = tf.placeholder(tf.int32)
            self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.screen_channel],
                                      name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [9, 9], [2, 2],
                                                             activation_fn, self.cnn_format, name='l1')
            self.l1_out = self.l1
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l2')
            self.l2_out = self.l2
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l3')
            self.l3_out = self.l3
            self.l4, self.w['l4_w'], self.w['l4_b'] = conv2d(self.l3_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l4')
            shape = self.l4.get_shape().as_list()
            self.l6_flat = tf.reshape(self.l4, [-1, reduce(lambda x, y: x * y, shape[1:])])

            ### Add action as input
            self.action_in = tf.placeholder('float32', [None, self.env.get_action_size() - 1], name='action_in')
            self.action_out = self.action_in
            self.l7, self.w['l7_w'], self.w['l7_b'] = linear(self.l6_flat, self.lstm_in,
                                                             activation_fn=activation_fn, name='l7')
            self.l7_action = tf.concat([self.l7, self.action_out], 1)

            # add LSTM with dynamic steps
            self.rnn_input = tf.reshape(self.l7_action, [self.batch, self.length,
                                                         self.l7_action.get_shape().as_list()[-1]])
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            self.state_in = self.rnn_cell.zero_state(self.batch, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=self.rnn_cell, dtype=tf.float32,
                                                         initial_state=self.state_in, scope='prediction_rnn')
            self.rnn = tf.reshape(self.rnn, shape=[-1, self.h_size])
            self.w['rnn_w'], self.w['rnn_b'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                 scope='prediction/prediction_rnn')
            # end LSTM

            self.q, self.w['q_w'], self.w['q_b'] = linear(self.rnn, self.env.action_size, name='q')
            self.q_action = tf.argmax(self.q, axis=1)

        tf.global_variables_initializer().run()
        self.load_model()


    def play(self):  # test
        rewards = []
        actions = []
        psnrs = []
        diction = {}
        # create folder if needed
        if self.is_save:
            names = []
            save_path = 'results/' + self.dataset + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # loop for test batch
        test_update = True
        total_base_psnr = 0.
        while test_update:
            img_num, batch_size = self.env.get_test_info()  # current image No. and batch size
            for m in range(self.stop_step):
                # predict action
                action_test = self.predict_test(count_step=m)
                self.pre_action_test = action_test.copy()
                reward_all, psnr_all, base_psnr = self.env.act_test(action_test, step=m)
                if m == 0:
                    total_base_psnr += base_psnr * batch_size
                    cur_img = self.env.get_data_test()
                    cur_img_temp = cur_img.copy()
                    cur_img[:, :, :, 0] = cur_img_temp[:, :, :, 2]
                    cur_img[:, :, :, 2] = cur_img_temp[:, :, :, 0]
                    # initialize names
                    if self.is_save:
                        for k in range(batch_size):
                            names.append(save_path + str(k + img_num + 1))

                # store reward, psnr, action
                rewards.append(reward_all)
                actions.append(action_test)
                psnrs.append(psnr_all)

                # construct dictionary for log
                diction['reward' + str(m + 1)] = rewards[-1] if 'reward' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['reward' + str(m + 1)], rewards[-1]], axis=0)
                diction['action' + str(m + 1)] = actions[-1] if 'action' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['action' + str(m + 1)], actions[-1]], axis=0)
                diction['psnr' + str(m + 1)] = psnrs[-1] if 'psnr' + str(m + 1) not in diction.keys() \
                    else np.concatenate([diction['psnr' + str(m + 1)], psnrs[-1]], axis=0)

                # print results
                print(('reward' + str(m + 1) + ': %.4f, psnr' + str(m + 1) + ': %.4f' +
                       ', tested images: %d, total tested images: %d') % (reward_all.mean(), psnr_all.mean(),
                                                                          batch_size, img_num + batch_size))

                # save images
                if self.is_save:
                    cur_img = self.env.get_test_imgs()
                    cur_img_temp = cur_img.copy()
                    cur_img[:,:,:,0] = cur_img_temp[:,:,:,2]
                    cur_img[:,:,:,2] = cur_img_temp[:,:,:,0]
                    save_img = np.swapaxes(cur_img, 1, 2)  # swap H, W
                    for k in range(batch_size):
                        names[k + img_num] += '_' + str(action_test[k] + 1)
                        cv2.imwrite(names[k + img_num] + '.png', 255 * save_img[k, ...])

            test_update = self.env.update_test_data()

        # print final results
        print('This is the final result:')
        for m in range(self.stop_step):
            print(('reward' + str(m + 1) + ': %.4f, psnr' + str(m + 1) + ': %.4f' +
                   ', toal tested images: %d') % (diction['reward' + str(m + 1)].mean(),
                                                  diction['psnr' + str(m + 1)].mean(),
                                                  img_num + batch_size))
        mean_base_psnr = total_base_psnr / (img_num + batch_size)
        print('base_psnr: %.4f' % mean_base_psnr)


