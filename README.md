# Readme

## 环境安装
1. 创建环境  执行命令：`conda create -n tkhmr python=3.10`
2. 安装pytorch，cuda版本为11.8, Pytorch版本为2.1.0：
`pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
3. 安装TokenHMR依赖：`pip install -r requirements.txt`
4. 安装Detectron2以运行图片Demo
`pip install git+https://github.com/facebookresearch/detectron2
5. 安装Sapiens模型依赖，进入sapiens文件夹，运行install.sh
6. 下载模型：运行fetch_demo_data.sh；在网址https://huggingface.co/facebook/sapiens中下载sapiens-depth-2b模型，并将其放入TokenHMR\tokenhmr\lib\models\sapiens文件夹。
7. 复制SMPL模型`cp data/body_models/smpl/SMPL_NEUTRAL.pkl $HOME/.cache/phalp/3D/models/smpl/`

## 运行Demo
```shell
python tokenhmr/demo.py \
    --img_folder demo_sample/images/ \
    --batch_size=1 \
    --full_frame \
    --checkpoint data/checkpoints/tokenhmr_model_latest.ckpt \
    --model_config data/checkpoints/model_config.yaml
```

## 训练模型
在https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=bedlam.tar.gz 中下载bedlam数据集，
在https://www.dropbox.com/scl/fo/vp8v9wxw46n63w94xxnmo/AOpIPovpwNU6ucNBamGrLg8?rlkey=lmbd7cpce009gzmc41081gesi&e=1 中下载4DHuman训练的数据集。

数据存放位置
```shell
TokenHMR/
├── tokenhmr/
│   └── dataset_dir/
│       └── training_data/                      # Training data
│           └── dataset_tars/
│               └── coco-train-2014-pruned/
│               └── aic-train-vitpose/
│               └── bedlam/
|               ...                          
│           ...
│       └── evaluation_data/                    # Evaluation data
│           └── 3DPW/
│           └── EMDB/
│           └── emdb.npz
│           └── 3dpw_test.npz
└── ...
```
作者使用4块A100显卡训练了四天，运行下面代码进行训练：
`python tokenhmr/train.py datasets=mix_all experiment=tokenhmr_release`

若在测试集上测试，需要下载3DPW和EMDB数据集：
https://virtualhumans.mpi-inf.mpg.de/3DPW/
https://eth-ait.github.io/emdb/
并下载元数据：
https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=test.tar.gz
运行下面代码进行测试：
```shell
python tokenhmr/eval.py  \
    --dataset EMDB,3DPW-TEST \
    --batch_size 32 --log_freq 50 \
    --dataset_dir tokenhmr/dataset_dir/evaluation_data \
    --checkpoint data/checkpoints/tokenhmr_model.ckpt \
    --model_config data/checkpoints/model_config.yaml```