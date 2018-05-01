## RL-Restore [[project page](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/)][[paper](https://arxiv.org/abs/1804.03312)]
We run this code under [TensorFlow](https://www.tensorflow.org/) 1.3.

Test code is ready. Training code is coming...

### Overview

- Framework
<img src='http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/support/framework.png' align="center">

- Results
![](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/support/restore.gif)

### Run
- Start testing
```
python main.py --dataset moderate
```
> `dataset`: choose a test set among `mild`, `moderate` and `severe`

- Dataset

  - All test sets can be downloaded at [Google Drive](https://drive.google.com/open?id=19z2s1e3zT8_1J9ZtsCOrzUSsrQahuINo) or [Baidu Cloud](https://pan.baidu.com/s/1RXTcfI-mne5YZh3myQcjzQ). 

  - Replace `test_images/` with the downloaded data and play with the whole dataset.

### Citation

    @inproceedings{yu2018crafting,
     author = {Ke Yu, Chao Dong, Liang Lin, and Chen Change Loy},
     title = {Crafting a Toolchain for Image Restoration by Deep Reinforcement Learning},
     booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
     month = {June},
     year = {2018} 
    }