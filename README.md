

# 基于注意力机制的跨模态迁移攻击方法



## 环境
```
（尽量安装python3.9）
pip install -r requirements.txt
```
## GPU infos
```
NVIDIA GeForce RTX 3060TI
NVIDIA-SMI 545.37.02       Driver Version: 546.65       
CUDA Version: 12.1
```
## 预训练模型和数据集
```

### Video model
For Kinetics-400, download config files from [gluon](https://cv.gluon.ai/model_zoo/action_recognition.html).  Models include i3d_nl5_resnet50_v1_kinetics400, i3d_nl5_resnet101_v1_kinetics400, slowfast_8x8_resnet50_kinetics400, slowfast_8x8_resnet101_kinetics400, tpn_resnet50_f32s2_kinetics400, tpn_resnet101_f32s2_kinetics400.
After that, change the CONFIG_ROOT of utils.py into your custom path. We use pretrained models on Kinetics-400 from gluon to conduct experiments.

For UCF-101, we fine-tune these models on UCF-101. Download checkpoint files from [here](https://drive.google.com/open?id=10KOlWdi5bsV9001uL4Bn1T48m9hkgsZ2&authuser=weizhipeng1226%40gmail.com&usp=drive_fs) and specify UCF_CKPT_PATH of utils.py.
<!-- (due to the double blind review, we will provide the link after the paper is accepted)  -->

### Dataset
Download Kinetics-400 dataset and UCF-101 dataset and set OPT_PATH of utils.py to specify the output path.

For Kinetics-400, change cfg.CONFIG.DATA.VAL_DATA_PATH of utils.py into your validation path.

For UCF-101, split videos into images and change UCF_IMAGE_ROOT of utils.py into your images path of UCF-101.
```
# 运行代码
## 消融实验和性能比较
使用此代码可获得表3.1表3.2和图3.14图3.15的结果.
```python
python image_main.py/image_main_ucf --gpu {gpu}
```
## 生成对抗样本
将我们提出的CMTA-SAM方法需要用白盒视频模型生成对抗样本。

对于kinetics-400数据集,
```python
python attack.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```
* model: 白盒模型.
* attack_method: 例如 FGSM, BIM, MI, 等.更多攻击参见base_attack.py
* step: 迭代次数.

对于UCF101数据集,
```python
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type image --attack_method {image_method} --step {step} --batch_size {batch_size} 
python attack_ucf101.py --gpu {gpu} --model {model} --attack_type video --attack_method TemporalTranslation --step {step} --batch_size 1
```

这些生成的对抗样本将存储在utils.py的OPT_PATH中，可以直接用作后续命令中的“—used_ori”和“—used_adv”的参数。

## 与更强的基线进行比较
对现有对抗样本进行微调: 
```python
python image_fine_tune_attack.py --gpu {gpu} --attack_method ILAF --used_ori {path} --used_adv {path} --opt_path {path} --white_model {model} --dataset {dataset}
```
* used_ori: 原始样本路径.
* used_adv: 现有对抗样本的路径.
* opt_path: 输出路径.
* white_model: 白盒模型.
* dataset: Kinetics-400 or UCF-101

预测这些生成的对抗样本 
```python
# ucf101 reference
python reference_ucf101.py --gpu {gpu} --adv_path {adv_path}
# kinetics reference
python reference.py --gpu {gpu} --adv_path {adv_path}
```
* adv_path: 生成对抗样本的输出路径.   
