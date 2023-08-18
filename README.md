# Multi-Phase Liver-Specific DCE-MRI Translation via a Registration-Guided GAN

::large_orange_diamond:This repository is developing.

![intro](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/5PNkpg.png)

![arch](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/R0Vd3i.png)

This paper has been accepted by: [The 8th Simulation and Synthesis in Medical Imaging (SASHIMI) workshop, MICCAI 2023.](https://2023.sashimi-workshop.org/call_for_papers/)

## Prepare dataset and meta information

```
-dataset
    - training_path
        - id_1.npy
        - id_2.npy
        ...
    - test_path
        - id_1.npy
        - id_2.npy
        ...
```

You can use `./data/split.py` to get meta imformation of your dataset.

```
data
    - train.txt
    - val.txt
    - test.txt
```

## Modify config file


## Train and test

Pretrain Unet:

```python
python train_UNet.py --config './Yaml/UNet.yaml'
```

Start visdom：

```python
python -m visdom.server -p 6019
```

Train:
```python
python train.py --config './Yaml/mrgan.yaml'
```

Test:
```python
python train.py --config './Yaml/mrgan.yaml'
```