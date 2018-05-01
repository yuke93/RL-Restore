import numpy as np
import tensorflow as tf
import os
from utils import psnr_cal, load_imgs, step_psnr_reward

class MyEnvironment(object):
    def __init__(self, config):

        screen_width, screen_height = config.screen_width, config.screen_height
        self.dims = (screen_width, screen_height)
        self.test_batch = config.test_batch
        self.test_in = 'test_images/' + config.dataset + '_in/'
        self.test_gt = 'test_images/' + config.dataset + '_gt/'
        self._screen = None
        self.reward = 0
        self.terminal = True
        self.stop_step = config.stop_step
        self.reward_func = config.reward_func

        # test data
        list_in = [self.test_in + name for name in os.listdir(self.test_in)]
        list_in.sort()
        list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
        list_gt.sort()
        self.data_all, self.label_all = load_imgs(list_in, list_gt)
        self.test_total = len(list_in)
        self.test_cur = 0

        # BGR --> RGB, swap H and W
        # This is because the data for tools training are in a different format
        # You don't need to do so with your own tools
        temp = self.data_all.copy()
        self.data_all[:, :, :, 0] = temp[:, :, :, 2]
        self.data_all[:, :, :, 2] = temp[:, :, :, 0]
        self.data_all = np.swapaxes(self.data_all, 1, 2)
        temp = self.label_all.copy()
        self.label_all[:, :, :, 0] = temp[:, :, :, 2]
        self.label_all[:, :, :, 2] = temp[:, :, :, 0]
        self.label_all = np.swapaxes(self.label_all, 1, 2)

        self.data_test = self.data_all[0 : min(self.test_batch, self.test_total), ...]
        self.label_test = self.label_all[0 : min(self.test_batch, self.test_total), ...]

        # reward functions
        self.rewards = {'step_psnr_reward': step_psnr_reward}
        self.reward_function = self.rewards[self.reward_func]

        # base_psnr (input psnr)
        self.base_psnr = 0.
        for k in range(len(self.data_all)):
            self.base_psnr += psnr_cal(self.data_all[k, ...], self.label_all[k, ...])
        self.base_psnr /= len(self.data_all)

        self.data = np.array([[[[0]]]])
        self._data_index = 0
        self._data_len = len(self.data)

        # build toolbox
        self.action_size = 12 + 1
        toolbox_path = 'toolbox/'
        self.graphs = []
        self.sessions = []
        self.inputs = []
        self.outputs = []
        for idx in range(12):
            g = tf.Graph()
            with g.as_default():
                # load graph
                saver = tf.train.import_meta_graph(toolbox_path + 'tool%02d' % (idx + 1) + '.meta')
                # input data
                input_data = g.get_tensor_by_name('Placeholder:0')
                self.inputs.append(input_data)
                # get the output
                output_data = g.get_tensor_by_name('sum:0')
                self.outputs.append(output_data)
                # save graph
                self.graphs.append(g)
            sess = tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True))
            with g.as_default():
                with sess.as_default():
                    saver.restore(sess, toolbox_path + 'tool%02d' % (idx + 1))
                    self.sessions.append(sess)


    def act_test(self, action, step = 0):
        reward_all = np.zeros(action.shape)
        psnr_all = np.zeros(action.shape)
        if step == 0:
            self.test_imgs = self.data_test.copy()
            self.test_temp_imgs = self.data_test.copy()
            self.test_pre_imgs = self.data_test.copy()
            self.test_steps = np.zeros(len(action), dtype=int)
        for k in range(len(action)):
            img_in = self.data_test[k:k+1,...].copy() if step == 0 else self.test_imgs[k:k+1,...].copy()
            img_label = self.label_test[k:k+1,...].copy()
            self.test_temp_imgs[k:k+1,...] = img_in.copy()
            psnr_pre = psnr_cal(img_in, img_label)
            if action[k] == self.action_size - 1 or self.test_steps[k] == self.stop_step: # stop action or already stop
                img_out = img_in.copy()
                self.test_steps[k] = self.stop_step # terminal flag
            else:
                feed_dict = {self.inputs[action[k]]: img_in}
                with self.graphs[action[k]].as_default():
                    with self.sessions[action[k]].as_default():
                        with tf.device('/gpu:0'):
                            img_out = self.sessions[action[k]].run(self.outputs[action[k]], feed_dict=feed_dict)
                self.test_steps[k] += 1
            self.test_pre_imgs[k:k+1,...] = self.test_temp_imgs[k:k+1,...].copy()
            self.test_imgs[k:k+1,...] = img_out.copy()  # keep intermediate results
            psnr = psnr_cal(img_out, img_label)
            reward = self.reward_function(psnr, psnr_pre=psnr_pre)
            psnr_all[k] = psnr
            reward_all[k] = reward

        return reward_all, psnr_all, self.base_psnr


    def get_test_imgs(self):
        return self.test_imgs.copy()


    def get_test_steps(self):
        return self.test_steps.copy()


    def get_data_test(self):
        return self.data_test.copy()


    def get_action_size(self):
        return self.action_size


    def get_test_info(self):
        return self.test_cur, len(self.data_test) # current image number & batch size


    def update_test_data(self):
        self.test_cur = self.test_cur + len(self.data_test)
        test_end = min(self.test_total, self.test_cur + self.test_batch)
        if self.test_cur >= test_end:
            return False #failed
        else:
            self.data_test = self.data_all[self.test_cur: test_end, ...]
            self.label_test = self.label_all[self.test_cur: test_end, ...]
            # swap axes if shape is not right (for mixed data)
            if self.data_test.shape[-1] > 3:
                self.data_test = np.swapaxes(self.data_test, 1, 2)
                self.data_test = np.swapaxes(self.data_test, 2, 3)
                self.label_test = np.swapaxes(self.label_test, 1, 2)
                self.label_test = np.swapaxes(self.label_test, 2, 3)
            # update base psnr
            self.base_psnr = 0.
            for k in range(len(self.data_test)):
                self.base_psnr += psnr_cal(self.data_test[k, ...], self.label_test[k, ...])
            self.base_psnr /= len(self.data_test)
            return True #successful
