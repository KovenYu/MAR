## MAR: Multilabel Reference Learning
![](img/framework.png)

This repo contains the source code for our CVPR'19 work
[Unsupervised person re-identification by soft multilabel learning](
https://kovenyu.com/publication/2019-cvpr-mar/).
Our implementation is based on [Pytorch](https://pytorch.org/).
In the following is an instruction to use the code
to train and evaluate the MAR model on the [Market-1501](
http://www.liangzheng.org/Project/project_reid.html) dataset.

### Prerequisites
1. [Pytorch 1.0.0](https://pytorch.org/)
2. Python 3.6+
3. Python packages: numpy, scipy, pyyaml/yaml, h5py
4. [Optional] MATLAB, if you need to customize used datasets.

### Data preparation
(If you simply want to run the demo code without further modification,
you might skip this step by downloading all required data from
[BaiduPan](https://pan.baidu.com/s/1O0s_dJcbkku6T0MwlLQecw) with
password "tih8",
and put all of them into */data/*)

1. Pretrained model

    Please find the pretrained model (pretrained using softmax loss on MSMT17) 
[here](https://pan.baidu.com/s/1O0s_dJcbkku6T0MwlLQecw) (password: tih8).
After downloading *pretrained.pth*, please put it into */data/*.

2. Target dataset

    Download the [Market-1501](
http://www.liangzheng.org/Project/project_reid.html) dataset,
and unzip it into */data*. After this step, you should have
a folder structure:
    - data
        - Market-1501-v15.09.15
            - bounding_box_test
            - bounding_box_train
            - query

    Then run [/data/construct_dataset_Market.m](/data/construct_dataset_Market.m)
    in MATLAB. If you prefer to use another dataset, just modify the MATLAB code accordingly.
Again, the processed Market-1501 and DukeMTMC-reID are available [here](https://pan.baidu.com/s/1O0s_dJcbkku6T0MwlLQecw).

3. Auxiliary (source) dataset

    Download the [MSMT17](https://http://www.pkuvmc.com/publications/msmt17.html) 
dataset, and unzip it into */data*. After this step, you should have a folder structure:
    - data
        - MSMT17_V1
            - train
            - test
            - list_train.txt
            - list_query.txt
            - list_gallery.txt

    Then run [/data/construct_dataset_MSMT17.m](/data/construct_dataset_MSMT17.m) in MATLAB.
If you prefer to use another dataset, just modify the MATLAB code accordingly.
Again, the processed MSMT17 is available 
[here](https://pan.baidu.com/s/1O0s_dJcbkku6T0MwlLQecw).
     

### Run the code

Please enter the main folder, and run
```bash
python src/main.py --gpu 0,1,2,3 --save_path runs/debug
```
where "0,1,2,3" specifies your gpu IDs.
If you are using gpus with 12G memory, you need 4 gpus to run 
in the default setting (batchsize=368).
Please also note that since I load the whole datasets into cpu memory
to cut down IO overhead,
you need at least 40G cpu memory. Hence I recommend you run it on a server.

### Reference

If you find our work helpful in your research,
please kindly cite our paper:

Hong-Xing Yu, Wei-Shi Zheng, Ancong Wu, Xiaowei Guo, Shaogang Gong
and Jian-Huang Lai, "Unsupervised person re-identification by soft multilabel learning",
In CVPR, 2019.

bib:
```
@inproceedings{yu2019unsupervised,
  title={Unsupervised Person Re-identification by Soft Multilabel Learning},
  author={Yu, Hong-Xing and Zheng, Wei-Shi and Wu, Ancong and Guo, Xiaowei and Gong, Shaogang and Lai, Jianhuang},
  year={2019},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

If you have any problem/question, please feel free to contact me at xKoven@gmail.com
or open an issue.
