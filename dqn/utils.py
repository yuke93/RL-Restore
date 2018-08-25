import math
import numpy as np
import cv2


def psnr_cal(im_input, im_label):
    loss = (im_input - im_label) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr


def load_imgs(list_in, list_gt, size = 63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = cv2.imread(list_in[k]) / 255.
        imgs_gt[k, ...] = cv2.imread(list_gt[k]) / 255.
    return imgs_in, imgs_gt


def img2patch(my_img, size=63):
    height, width, _ = np.shape(my_img)
    assert height >= size and width >= size
    patches = []
    for k in range(0, height - size + 1, size):
        for m in range(0, width - size + 1, size):
            patches.append(my_img[k: k+size, m: m+size, :].copy())
    return np.array(patches)


def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1]
    out = np.swapaxes(out, 1, 2)
    return out


def step_psnr_reward(psnr, psnr_pre):
    reward = psnr - psnr_pre
    return reward