<div align="center">

<h1>
OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic
</h1>

<p align="center">
<!-- <a href=https://arxiv.org/abs/2412.09951><img src="https://img.shields.io/badge/ArXiv-2412.09951-%23840707.svg" alt="ArXiv"></a> -->

<!-- [![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2412.09951) -->

[![🤗 HuggingFace models](https://img.shields.io/badge/HuggingFace🤗-Models-orange)](https://huggingface.co/wyddmw/OpenREAD/tree/main)
[![🤗 HuggingFace datasets](https://img.shields.io/badge/HuggingFace🤗-Datasets-orange)](https://huggingface.co/datasets/wyddmw/OpenREAD)

</p>

Songyan Zhang<sup>1*</sup>, Wenhui Huang<sup>2*</sup>, Zhan Chen<sup>1</sup>, Chua Jiahao Collister<sup>1</sup>, Qihang Huang<sup>1</sup>, Chen Lv<sup>1†</sup>

Nanyang Technological University<sup>1</sup>, Harvard University<sup>2</sup>

*Equal Contributions, †Corresponding Author

<image src="./asset/framework.png"/><br>
An overview of the framework of our OpenREAD.
</div>

## ✨Capabilities

<image src="./asset/planning.png"/>
<image src="./asset/lingoqa.png"/>


An overview of the capability of our proposed OpenREAD, a vision-language model tailored for autonomous driving by reinforment learning with GRPO. Besides the trajectory planning, our OpenREAD is also capable of providing reasoning-enhanced response for open-ended scenario understanding, action analysis, *etc*.

## 🦙 Data & Model Zoo
Our OpenREAD is built upon the [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and finetuned on a mixture of datasets including LingoQA, OmniDrive, and NuScenes datasets. Our OpenREAD is now available at [huggingface](https://huggingface.co/wyddmw/WiseAD). Enjoy playing with it!

<image src="./asset/annotations.png"/>

To facalitate the learning of reasoning capability at the cold start stage, we construct a large scale of CoT annotations on the LingoQA and NuScenes datasets as shown above. We further extend the amount of annations for LingoQA from 7K to 11K. All the CoT annotations are available [here](https://huggingface.co/datasets/wyddmw/OpenREAD).

## 🛠️ Install

1. Clone this repository and navigate to OpenREAD folder
  ```bash
  git clone https://github.com/wyddmw/OpenREAD
  cd OpenREAD
  ```

2. Install ms-swift package
  ```Shell
  conda create -n openread python=3.10 -y
  conda activate openread
  pip install -e .
  ```

3. Install Flash-Attention.
  ```Shell
    pip install flash_attn=2.8.3 --no-build-isolation
  ```

  If the installation is not compatable for your device and environment, please refer to the [source code](https://github.com/Dao-AILab/flash-attention/releases) and install the suitable version.

4. Install Qwen3-VL dependicies.
  ```Shell
    pip install "transformers==4.57" "qwen_vl_utils==0.0.14"
  ```

## 🪜 Training & Evaluation

### Datasets
The datasets used to train OpenREAD are as follows:
* [NuScenes](https://www.nuscenes.org/nuscenes)
* [LingoQA](https://github.com/wayveai/LingoQA)
* [OmniDrive](https://github.com/NVlabs/OmniDrive)

Please download our pre-processed [Lidar-BEV](https://huggingface.co/datasets/wyddmw/NuScenes_LidarBev) images for the NuScenes dataset. For trajectory evaluation, we use the GT cache introduced in [GPT-Driver](https://github.com/PointsCoder/GPT-Driver). Please download the GT cache from [Google Drive](https://drive.google.com/drive/folders/1NCqPtdK8agPi1q3sr9-8-vPdYj08OCAE)
The datasets are organized in the following structure:
```
data
├── LingoQA
│   ├── action
│   │   └── images
│   ├── evaluation
│   │   │── images
│   │   └── val.parquet
│   ├── scenery
│   │   └── images
│   ├── training_data.json
│   └── evaluation_data.json
├── nuscenes
│   ├── samples
│   │   ├── CAM_FRONT
│   │   ├── LIDAR_BEV
│   ├── gt
│   │   │── vad_gt_seg.pkl
│   │   └── gt_traj_mask.pkl
│   traj_val_bev_ego_status.json
│   
```

It is recommended to symlink your dataset root to `data`:

### Evaluate on the LingoQA dataset.
Before running the evaluation script, please first download the pretrained [Lingo-Judge](wayveai/Lingo-Judge).
Check the path of LingoQA dataset and LingoJudge pretrained model in the eval/LingoQA/eval_lingo.sh.
```shell
sh eval/LingoQA/eval_lingo.sh
```
The predictions, Lingo-Judge, CIDEr, Meteor, and BLEU metrics will be saved to the eval/LingoQA/lingoqa_results_OpenREAD.json.

### Evaluation on the NuScenes Trajectory Benchmark
We also provide scripts to evaluate trajectory prediction quality on the NuScenes validation set using both STP-3 and UniAD metrics. Update the trained model path, eval_file path, training mode, and inference outputs path in the eval/Trajectory/infer_trajs_dist.sh, then run trajectory inference:
```shell
bash eval/Trajectory/infer_trajs_dist.sh
```
This script generates trajectory prediction JSON files under the directory specified by inference outputs path. Next, update the trajectory inference outputs path inside eval/Trajectory/eval_trajs.py, Then compute both STP-3 and UniAD metrics by running:
```shell
python eval/Trajectory/eval_trajs.py
```

## 🔨 TODO LIST
- [✓] Release hugging face model, inference and eval scripts.
- [✓] Release CoT data.
- [ ] Release training code.

## Acknowledgment

We appeciate the awesome open-source project of [ms-swift](https://github.com/modelscope/ms-swift), [OmniDrive](https://github.com/NVlabs/OmniDrive), and [GPT-Driver](https://github.com/PointsCoder/GPT-Driver).


## ✏️ Citation
Coming soon.
