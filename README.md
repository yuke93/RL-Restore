## RL-Restore [[project page](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/)][[paper](https://arxiv.org/abs/1804.03312)]

:triangular_flag_on_post: Support arbitrary input size. Aug 25<br/>
:triangular_flag_on_post: Add Python3 compatibility. Aug 6<br/>
:triangular_flag_on_post: Training code is ready! Jun 15

### Overview

- Framework
<img src='imgs/framework.png' align="center">

- Synthetic & real-world results
![](imgs/restore.gif)
<p align="center">
    <img src='imgs/real_world.png' width = "100%">
</p>

### Prerequisite

- [Anaconda](https://www.anaconda.com/download/) is highly recommended as you can easily adjust the environment setting.
    ```
    pip install opencv-python scipy tqdm h5py
    ```

- We have tested our code under the following settings:<br/>

    | Python | TensorFlow | CUDA | cuDNN |
    | :----: | :--------: | :--: | :---: |
    |  2.7   |  1.3       | 8.0  |  5.1  |
    |  3.5   |  1.4       | 8.0  |  5.1  |
    |  3.6   |  1.10      | 9.0  |  7.0  |

### Test
- Start testing on synthetic dataset
    ```
    python main.py --dataset moderate
    ```
    > `dataset`: choose a test set among `mild`, `moderate` and `severe`

- :heavy_exclamation_mark: Start testing on real-world data (support arbitrary input size)
    ```
    python main.py --dataset mine
    ```

    - You may put your own test images in `data/test/mine/`

- Dataset

    - All test sets can be downloaded at [Google Drive](https://drive.google.com/open?id=19z2s1e3zT8_1J9ZtsCOrzUSsrQahuINo) or [Baidu Cloud](https://pan.baidu.com/s/1RXTcfI-mne5YZh3myQcjzQ).

    - Replace `test_images/` with the downloaded data and play with the whole dataset.

- Naming rules

    - Each saved image name refers to a selected toolchain. Please refer to my second reply in this [issue](https://github.com/yuke93/RL-Restore/issues/1).

### Train
- Download training images
    - Download training images (down-sampled DIV2K images) at [Google Drive](https://drive.google.com/file/d/146mmYHcZeWnklQ_Sg7ltCrJVqjL_yB3K/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1CD-E5dUMsMswvCVQhe5PeQ).

    - Move the downloaded file to `data/train/` and unzip.

-  Generate training data
    - Run `data/train/generate_train.m` to generate training data in HDF5 format.

    - You may generate multiple `.h5` files in `data/train/`

- Let's train!

    ```
    python main.py --is_train True
    ```

    - When you observe `reward_sum` is increasing, it indicates training is going well.

    - You can visualize reward increasing by TensorBoard.

    <img src='imgs/tensorboard.png' align="center">

### Acknowledgement
The DQN algorithm is modified from [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow).

### Citation

    @inproceedings{yu2018crafting,
     author = {Yu, Ke and Dong, Chao and Lin, Liang and Loy, Chen Change},
     title = {Crafting a Toolchain for Image Restoration by Deep Reinforcement Learning},
     booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition},
     pages={2443--2452},
     year = {2018} 
    }
