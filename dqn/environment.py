import numpy as np
import tensorflow as tf
import os
import h5py
import cv2
from .utils import psnr_cal, load_imgs, step_psnr_reward, data_reformat

class MyEnvironment(object):
    def __init__(self, config):
        self.reward = 0
        self.terminal = True
        self.stop_step = config.stop_step
        self.reward_func = config.reward_func
        self.is_train = config.is_train
        self.count = 0  # count restoration step
        self.psnr, self.psnr_pre, self.psnr_init = 0., 0., 0.

        if self.is_train:
            # training data
            self.train_list = [config.train_dir + file for file in os.listdir(config.train_dir) if file.endswith('.h5')]
            self.train_cur = 0
            self.train_max = len(self.train_list)
            f = h5py.File(self.train_list[self.train_cur], 'r')
            self.data = f['data'].value
            self.label = f['label'].value
            f.close()
            self.data_index = 0
            self.data_len = len(self.data)

            # validation data
            f = h5py.File(config.val_dir + os.listdir(config.val_dir)[0], 'r')
            self.data_test = f['data'].value
            self.label_test = f['label'].value
            f.close()
            self.data_all = self.data_test
            self.label_all = self.label_test
        else:
            if config.dataset == 'mine':
                self.my_img_dir = config.test_dir + 'mine/'
                self.my_img_list = os.listdir(self.my_img_dir)
                self.my_img_list.sort()
                self.my_img_idx = 0

            elif config.dataset in ['mild', 'moderate', 'severe']:
                # test data
                self.test_batch = config.test_batch
                self.test_in = config.test_dir + config.dataset + '_in/'
                self.test_gt = config.test_dir + config.dataset + '_gt/'
                list_in = [self.test_in + name for name in os.listdir(self.test_in)]
                list_in.sort()
                list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
                list_gt.sort()
                self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
                self.data_all, self.label_all = load_imgs(list_in, list_gt)
                self.test_total = len(list_in)
                self.test_cur = 0
    
                # data reformat, because the data for tools training are in a different format
                self.data_all = data_reformat(self.data_all)
                self.label_all = data_reformat(self.label_all)
                self.data_test = self.data_all[0 : min(self.test_batch, self.test_total), ...]
                self.label_test = self.label_all[0 : min(self.test_batch, self.test_total), ...]
            else:
                raise ValueError('Invalid dataset!')

        if self.is_train or config.dataset!='mine':
            # input PSNR
            self.base_psnr = 0.
            for k in range(len(self.data_test)):
                self.base_psnr += psnr_cal(self.data_test[k, ...], self.label_test[k, ...])
            self.base_psnr /= len(self.data_test)

            # reward functions
            self.rewards = {'step_psnr_reward': step_psnr_reward}
            self.reward_function = self.rewards[self.reward_func]

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


    def new_image(self):
        self.terminal = False
        while self.data_index < self.data_len:
            self.img = self.data[self.data_index: self.data_index + 1, ...]
            self.img_gt = self.label[self.data_index: self.data_index + 1, ...]
            self.psnr = psnr_cal(self.img, self.img_gt)
            if self.psnr > 50:  # ignore too smooth samples and rule out 'inf'
                self.data_index += 1
            else:
                break

        # update training file
        if self.data_index >= self.data_len:
            if self.train_max > 1:
                self.train_cur += 1
                if self.train_cur >= self.train_max:
                    self.train_cur = 0

                # load new file
                print('loading file No.%d' % (self.train_cur + 1))
                f = h5py.File(self.train_list[self.train_cur], 'r')
                self.data = f['data'].value
                self.label = f['label'].value
                self.data_len = len(self.data)
                f.close()

            # start from beginning
            self.data_index = 0
            while True:
                self.img = self.data[self.data_index: self.data_index + 1, ...]
                self.img_gt = self.label[self.data_index: self.data_index + 1, ...]
                self.psnr = psnr_cal(self.img, self.img_gt)
                if self.psnr > 50:  # ignore too smooth samples and rule out 'inf'
                    self.data_index += 1
                else:
                    break

        self.reward = 0
        self.count = 0
        self.psnr_init = self.psnr
        self.data_index += 1
        return self.img, self.reward, 0, self.terminal


    def act(self, action):
        self.psnr_pre = self.psnr
        if action == self.action_size - 1:  # stop
            self.terminal = True
        else:
            feed_dict = {self.inputs[action]: self.img}
            with self.graphs[action].as_default():
                with self.sessions[action].as_default():
                    im_out = self.sessions[action].run(self.outputs[action], feed_dict=feed_dict)
            self.img = im_out
        self.psnr = psnr_cal(self.img, self.img_gt)

        # max step
        if self.count >= self.stop_step - 1:
            self.terminal = True

        # stop if too bad
        if self.psnr < self.psnr_init:
            self.terminal = True

        # calculate reward
        self.reward = self.reward_function(self.psnr, self.psnr_pre)
        self.count += 1

        return self.img, self.reward, self.terminal


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

        if self.is_train:
            return reward_all.mean(), psnr_all.mean(), self.base_psnr
        else:
            return reward_all, psnr_all, self.base_psnr


    def update_test_data(self):
        self.test_cur = self.test_cur + len(self.data_test)
        test_end = min(self.test_total, self.test_cur + self.test_batch)
        if self.test_cur >= test_end:
            return False #failed
        else:
            self.data_test = self.data_all[self.test_cur: test_end, ...]
            self.label_test = self.label_all[self.test_cur: test_end, ...]

            # update base psnr
            self.base_psnr = 0.
            for k in range(len(self.data_test)):
                self.base_psnr += psnr_cal(self.data_test[k, ...], self.label_test[k, ...])
            self.base_psnr /= len(self.data_test)
            return True #successful


    def act_test_mine(self, my_img_cur, action):
        if action == self.action_size - 1:
            return my_img_cur.copy()
        else:
            if my_img_cur.ndim == 4:
                feed_img_cur = my_img_cur
            else:
                feed_img_cur = my_img_cur.reshape((1,) + my_img_cur.shape)
            my_img_next = self.sessions[action].run(self.outputs[action], feed_dict={self.inputs[action]: feed_img_cur})
            return my_img_next[0, ...]


    def update_test_mine(self):
        """
        :return: (image, image name) or (None, None)
        """
        if self.my_img_idx >= len(self.my_img_list):
            return None, None
        else:
            img_name = self.my_img_list[self.my_img_idx]
            base_name, _ = os.path.splitext(img_name)
            my_img = cv2.imread(self.my_img_dir + img_name)
            my_img = my_img[:,:,::-1] / 255.
            self.my_img_idx += 1
            return my_img, base_name


    def get_test_imgs(self):
        return self.test_imgs.copy()


    def get_test_steps(self):
        return self.test_steps.copy()


    def get_data_test(self):
        return self.data_test.copy()


    def get_test_info(self):
        return self.test_cur, len(self.data_test) # current image number & batch size
